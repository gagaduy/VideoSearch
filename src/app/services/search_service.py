import re
from collections import defaultdict

from sqlalchemy.orm import Session

from app.config import settings
from app.db.repositories.search import create_query_log, fetch_frame_media_map
from app.services.local_rerank import score_local_candidate
from app.services.query_expansion import expand_query
from app.services.query_rerank import rerank_structured_candidates
from app.services.query_understanding import ObjectFilter, StructuredQuery, TemporalStep, parse_structured_query
from app.services.retrieval_branches import collect_branch_candidates
from app.services.retrieval_fusion import apply_constraint_penalty, fuse_branch_rankings
from app.services.temporal_paths import find_best_temporal_paths
from worker.adapters.openclip_adapter import OpenClipAdapter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lexical_score(query_terms: list[str], text: str) -> float:
    if not query_terms:
        return 0.0
    haystack = set(_tokenize(text))
    if not haystack:
        return 0.0
    matched = sum(1 for term in query_terms if term in haystack)
    return matched / len(query_terms)


def _merge_object_filters(structured: StructuredQuery, object_labels: list[str]) -> list[ObjectFilter]:
    merged = list(structured.object_filters)
    seen = {(item.label.lower(), item.min_count, tuple(item.regions)) for item in merged}
    for label in object_labels:
        key = (label.lower(), 1, ())
        if key not in seen:
            merged.append(ObjectFilter(label=label, min_count=1))
            seen.add(key)
    return merged


def _row_display_counts(row: dict[str, object]) -> dict[str, int]:
    semantic_counts = {str(key).lower(): int(value) for key, value in dict(row.get("semantic_counts", {})).items()}
    if semantic_counts:
        return semantic_counts
    counts = {str(key).lower(): int(value) for key, value in dict(row["object_counts"]).items()}
    for label in [str(item).lower() for item in list(row["labels"])]:
        counts.setdefault(label, 1)
    return counts


def _row_display_labels(row: dict[str, object]) -> list[str]:
    semantic_counts = _row_display_counts(row)
    if semantic_counts:
        return sorted(semantic_counts)
    return sorted({str(item).lower() for item in list(row["labels"]) if str(item).strip()})


def _row_semantic_positions(row: dict[str, object]) -> dict[str, set[str]]:
    semantic_entities = list(row.get("semantic_entities", []))
    if semantic_entities:
        positions: dict[str, set[str]] = {}
    else:
        positions = {
            str(key).lower(): {str(region).lower() for region in value}
            for key, value in dict(row["object_positions"]).items()
        }
    for entity in semantic_entities:
        label = str(entity.get("label", "")).strip().lower()
        aliases = [str(alias).strip().lower() for alias in entity.get("aliases", []) if str(alias).strip()]
        regions = {str(region).strip().lower() for region in entity.get("regions", []) if str(region).strip()}
        if not regions:
            continue
        for name in [label, *aliases]:
            if name:
                positions.setdefault(name, set()).update(regions)
    return positions


def _row_object_score(row: dict[str, object], filters: list[ObjectFilter]) -> tuple[bool, float]:
    if not filters:
        return True, 0.0

    counts = _row_display_counts(row)
    positions = _row_semantic_positions(row)
    filter_scores: list[float] = []
    for object_filter in filters:
        label = object_filter.label.lower()
        count = counts.get(label, 0)
        if count < object_filter.min_count:
            return False, 0.0
        if object_filter.max_count is not None and count > object_filter.max_count:
            return False, 0.0

        score = min(count / max(object_filter.min_count, 1), 1.0)
        if object_filter.regions:
            available = positions.get(label, set())
            required = {region.lower() for region in object_filter.regions}
            matched_regions = len(required & available)
            if matched_regions < len(required):
                return False, 0.0
            score = (score + (matched_regions / len(required))) / 2.0
        filter_scores.append(score)
    return True, sum(filter_scores) / len(filter_scores)


def _step_score(row: dict[str, object], step: TemporalStep) -> float:
    terms = _tokenize(step.text)
    text_score = max(
        _lexical_score(terms, str(row["caption_text"])),
        _lexical_score(terms, str(row["ocr_text"])),
    )
    passed, object_score = _row_object_score(row, step.object_filters)
    if not passed:
        return 0.0
    if not step.object_filters:
        return text_score
    return max(text_score, object_score)


def _temporal_path_scores(rows: list[dict[str, object]], steps: list[TemporalStep]) -> dict[int, float]:
    if len(steps) < 2:
        return {}

    ordered_rows = sorted(rows, key=lambda item: (int(item["video_id"]), int(item["segment_index"]), int(item["segment_id"])))
    step_candidates: list[list[dict[str, object]]] = []
    for step in steps:
        scored = []
        for row in ordered_rows:
            score = _step_score(row, step)
            if score > 0:
                scored.append(
                    {
                        "segment_id": int(row["segment_id"]),
                        "video_id": int(row["video_id"]),
                        "segment_index": int(row["segment_index"]),
                        "score": score,
                    }
                )
        step_candidates.append(scored)

    path_scores: dict[int, float] = defaultdict(float)
    for path in find_best_temporal_paths(step_candidates, max_gap=6):
        segment_ids = list(path["segment_ids"])
        for ordinal, segment_id in enumerate(segment_ids, start=1):
            path_scores[int(segment_id)] = max(
                path_scores[int(segment_id)],
                float(path["score"]) * (ordinal / len(segment_ids)),
            )
    return dict(path_scores)


def _candidate_rankings(
    rows: list[dict[str, object]],
    expanded_queries: list[str],
    object_filters: list[ObjectFilter],
) -> tuple[dict[str, list[int]], dict[int, dict[str, float]], dict[int, bool]]:
    text_terms = _tokenize(" ".join(expanded_queries))
    diagnostics: dict[int, dict[str, float]] = {}
    passes_filters: dict[int, bool] = {}

    dense_ranking = [int(row["segment_id"]) for row in rows]
    text_scored = sorted(
        (
            (
                int(row["segment_id"]),
                max(
                    _lexical_score(text_terms, str(row["caption_text"])),
                    _lexical_score(text_terms, str(row["ocr_text"])),
                ),
            )
            for row in rows
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    object_scores: list[tuple[int, float]] = []
    for row in rows:
        segment_id = int(row["segment_id"])
        passed, score = _row_object_score(row, object_filters)
        object_scores.append((segment_id, score))
        passes_filters[segment_id] = passed

    object_scored = sorted(object_scores, key=lambda item: item[1], reverse=True)

    for row in rows:
        segment_id = int(row["segment_id"])
        diagnostics[segment_id] = {
            "vector_distance": float(row.get("vector_distance", 0.0)),
            "text_score": next(score for candidate_id, score in text_scored if candidate_id == segment_id),
            "object_score": next(score for candidate_id, score in object_scored if candidate_id == segment_id),
        }

    rankings: dict[str, list[int]] = {"dense": dense_ranking}
    text_ranking = [segment_id for segment_id, score in text_scored if score > 0]
    if text_ranking:
        rankings["text"] = text_ranking
    object_ranking = [segment_id for segment_id, score in object_scored if score > 0]
    if object_ranking:
        rankings["object"] = object_ranking
    return rankings, diagnostics, passes_filters


def _entity_score(row: dict[str, object], query_terms: list[str]) -> float:
    labels = set(_row_display_labels(row))
    aliases = {
        str(alias).lower()
        for values in dict(row.get("semantic_aliases", {})).values()
        for alias in values
    }
    haystack = labels | aliases
    if not haystack or not query_terms:
        return 0.0
    matched = sum(1 for term in query_terms if term in haystack)
    return matched / len(query_terms)


def run_search(db: Session, query: str, object_labels: list[str]) -> dict[str, object]:
    structured = parse_structured_query(query, api_key=settings.openai_api_key, model=settings.openai_model)
    object_filters = _merge_object_filters(structured, object_labels)
    expanded = structured.semantic_queries or [structured.semantic_query]
    if settings.openai_api_key:
        for item in expand_query(structured.semantic_query, api_key=settings.openai_api_key, model=settings.openai_model):
            if item not in expanded:
                expanded.append(item)

    branch_rows = collect_branch_candidates(
        db,
        semantic_query=structured.semantic_query,
        expanded_queries=expanded,
        object_filters=object_filters,
        temporal_steps=structured.temporal_steps,
        dense_encoder=OpenClipAdapter(),
    )
    if not any(branch_rows.values()):
        create_query_log(db, query, expanded, object_labels, [])
        db.commit()
        return {"query": query, "expanded_queries": expanded, "results": [], "parsed_query": structured.model_dump()}

    rows_by_segment: dict[int, dict[str, object]] = {}
    branch_rankings: dict[str, list[int]] = {}
    for branch_name, items in branch_rows.items():
        branch_rankings[branch_name] = [int(item["segment_id"]) for item in items]
        for item in items:
            segment_id = int(item["segment_id"])
            existing = rows_by_segment.setdefault(segment_id, dict(item))
            for key, value in item.items():
                if key not in existing or existing[key] in (None, "", [], {}):
                    existing[key] = value
    rows = list(rows_by_segment.values())

    rankings, diagnostics, passes_filters = _candidate_rankings(rows, expanded, object_filters)
    for branch_name, ranking in branch_rankings.items():
        if ranking:
            rankings[branch_name] = ranking
    fused_scores = fuse_branch_rankings(rankings)
    temporal_bonus = _temporal_path_scores(rows, structured.temporal_steps)
    frame_media = fetch_frame_media_map(
        db,
        [int(row["keyframe_id"]) for row in rows if row["keyframe_id"] is not None],
    )

    filtered_segment_ids = {segment_id for segment_id, passed in passes_filters.items() if passed}
    enforce_hard_filter = bool(object_filters and filtered_segment_ids)
    rerank_payload = [
        {
            "segment_id": int(row["segment_id"]),
            "caption_text": str(row["caption_text"]),
            "ocr_text": str(row["ocr_text"]),
            "object_labels": _row_display_labels(row),
            "object_counts": _row_display_counts(row),
            "object_positions": {key: sorted(value) for key, value in _row_semantic_positions(row).items()},
            "semantic_entities": list(row.get("semantic_entities", [])),
            "semantic_counts": dict(row.get("semantic_counts", {})),
            "start_timestamp_sec": float(row["start_timestamp_sec"]),
            "end_timestamp_sec": float(row["end_timestamp_sec"]),
        }
        for row in rows[:25]
    ]
    llm_alignment_scores = rerank_structured_candidates(
        structured,
        rerank_payload,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    ranked_segments: list[dict[str, object]] = []
    for row in rows:
        segment_id = int(row["segment_id"])
        if enforce_hard_filter and not passes_filters[segment_id]:
            continue
        keyframe_id = int(row["keyframe_id"]) if row["keyframe_id"] is not None else None
        media = frame_media.get(keyframe_id or -1, {})
        ranked_segments.append(
            {
                "segment_id": segment_id,
                "video_id": int(row["video_id"]),
                "segment_index": int(row["segment_index"]),
                "frame_id": keyframe_id,
                "timestamp_sec": float(row["start_timestamp_sec"]),
                "start_timestamp_sec": float(row["start_timestamp_sec"]),
                "end_timestamp_sec": float(row["end_timestamp_sec"]),
                "score": (
                    0.0
                ),
                "object_labels": _row_display_labels(row),
                "object_counts": _row_display_counts(row),
                "object_positions": {key: sorted(value) for key, value in _row_semantic_positions(row).items()},
                "caption": str(row["caption_text"]),
                "thumb_url": f"/media/frames/{keyframe_id}/thumb" if keyframe_id is not None and media else "",
                "image_url": f"/media/frames/{keyframe_id}/image" if keyframe_id is not None and media else "",
                "preview_url": f"/media/frames/{keyframe_id}/preview" if keyframe_id is not None and media else "",
                "diagnostics": {
                    **diagnostics.get(segment_id, {}),
                    "llm_alignment_score": llm_alignment_scores.get(segment_id, 0.0),
                    "constraint_mode": "hard" if enforce_hard_filter else "soft",
                },
            }
        )

    text_terms = _tokenize(" ".join(expanded))
    for item in ranked_segments:
        segment_id = int(item["segment_id"])
        base = {
            **item,
            "dense_score": fused_scores.get(segment_id, 0.0),
            "text_score": diagnostics.get(segment_id, {}).get("text_score", 0.0),
            "ocr_score": _lexical_score(text_terms, str(item["caption"])) if item["caption"] else 0.0,
            "object_score": diagnostics.get(segment_id, {}).get("object_score", 0.0),
            "entity_score": _entity_score(rows_by_segment[segment_id], text_terms),
            "temporal_score": temporal_bonus.get(segment_id, 0.0),
            "hard_constraints_passed": passes_filters.get(segment_id, True) if enforce_hard_filter else True,
        }
        item["score"] = apply_constraint_penalty(
            base,
            score_local_candidate(base) + (0.12 * llm_alignment_scores.get(segment_id, 0.0)),
        )

    ranked_segments = sorted(ranked_segments, key=lambda row: float(row["score"]), reverse=True)
    results = [item for item in ranked_segments if item.get("frame_id")]
    create_query_log(db, query, expanded, object_labels, results)
    db.commit()
    return {
        "query": query,
        "expanded_queries": expanded,
        "results": results,
        "parsed_query": structured.model_dump(),
    }
