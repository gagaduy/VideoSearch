import re
from collections import defaultdict
from time import perf_counter

from sqlalchemy.orm import Session

from app.config import settings
from app.db.repositories.search import create_query_log, fetch_frame_media_map, search_segment_candidates
from app.services.local_rerank import score_local_candidate
from app.services.openai_vision_rerank import (
    blend_rerank_score,
    run_openai_vision_rerank,
    select_rerank_candidates,
    should_run_openai_vision_rerank,
)
from app.services.object_refinement import refine_object_matches
from app.services.query_expansion import expand_query
from app.services.query_rerank import rerank_structured_candidates
from app.services.query_understanding import ObjectFilter, StructuredQuery, TemporalStep, parse_structured_query
from app.services.retrieval_branches import collect_branch_candidates
from app.services.retrieval_fusion import apply_constraint_penalty, fuse_branch_rankings
from app.services.temporal_paths import find_best_temporal_paths
from worker.retrieval_ontology import canonicalize_object_label
from worker.adapters.openclip_adapter import OpenClipAdapter

_SEARCH_DENSE_ENCODER: OpenClipAdapter | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_search_dense_encoder() -> OpenClipAdapter:
    global _SEARCH_DENSE_ENCODER
    if _SEARCH_DENSE_ENCODER is None:
        _SEARCH_DENSE_ENCODER = OpenClipAdapter()
    return _SEARCH_DENSE_ENCODER


def prewarm_search_runtime() -> None:
    try:
        _get_search_dense_encoder().embed_text("search runtime prewarm")
    except Exception:
        return


def _current_rss_mb() -> float:
    try:
        for line in open("/proc/self/status", "r", encoding="utf-8"):
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(float(parts[1]) / 1024.0, 1)
    except Exception:
        return 0.0
    return 0.0


def _record_search_stage(metrics: dict[str, object], stage: str, started_at: float) -> None:
    metrics.setdefault("stage_timings", {})[stage] = {
        "elapsed_sec": round(perf_counter() - started_at, 6),
        "rss_mb": _current_rss_mb(),
    }


def _normalize_score_map(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    positive_values = [float(value) for value in scores.values() if float(value) > 0.0]
    if not positive_values:
        return {int(key): 0.0 for key in scores}
    max_value = max(positive_values)
    if max_value == 0.0:
        return {int(key): 0.0 for key in scores}
    if all(float(value) == max_value for value in positive_values):
        return {int(key): (1.0 if float(value) > 0.0 else 0.0) for key, value in scores.items()}
    normalized: dict[int, float] = {}
    for key, value in scores.items():
        numeric = float(value)
        if numeric <= 0.0:
            normalized[int(key)] = 0.0
        else:
            normalized[int(key)] = numeric / max_value
    return normalized


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
    counts = {str(key).lower(): int(value) for key, value in dict(row["object_counts"]).items()}
    for label in [str(item).lower() for item in list(row["labels"])]:
        counts.setdefault(label, 1)
    return counts


def _row_display_labels(row: dict[str, object]) -> list[str]:
    return sorted({str(item).lower() for item in list(row["labels"]) if str(item).strip()})


def _row_entity_labels(row: dict[str, object]) -> set[str]:
    semantic_counts = {str(key).lower() for key in dict(row.get("semantic_counts", {})).keys() if str(key).strip()}
    if semantic_counts:
        return semantic_counts
    return set(_row_display_labels(row))


def _row_semantic_positions(row: dict[str, object]) -> dict[str, set[str]]:
    positions = {
        str(key).lower(): {str(region).lower() for region in value}
        for key, value in dict(row["object_positions"]).items()
    }
    semantic_entities = list(row.get("semantic_entities", []))
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
        canonical_label = canonicalize_object_label(label)
        count = max(
            counts.get(label, 0),
            counts.get(canonical_label, 0),
        )
        if count < object_filter.min_count:
            return False, 0.0
        if object_filter.max_count is not None and count > object_filter.max_count:
            return False, 0.0

        if object_filter.max_count is not None:
            score = 1.0
        elif object_filter.min_count > 1:
            score = 1.0 / (1.0 + abs(count - object_filter.min_count))
        else:
            score = min(count / max(object_filter.min_count, 1), 1.0)
        if object_filter.regions:
            available = positions.get(label, set()) | positions.get(canonical_label, set())
            required = {region.lower() for region in object_filter.regions}
            matched_regions = len(required & available)
            if matched_regions < len(required):
                return False, 0.0
            score = (score + (matched_regions / len(required))) / 2.0
        filter_scores.append(score)
    return True, sum(filter_scores) / len(filter_scores)


def _row_count_refine_score(row: dict[str, object], filters: list[ObjectFilter]) -> tuple[bool, float]:
    count_filters = [
        item for item in filters
        if item.max_count is None and item.min_count > 1
    ]
    if not count_filters:
        return False, 0.0

    counts = _row_display_counts(row)
    scores: list[float] = []
    for object_filter in count_filters:
        label = object_filter.label.lower()
        canonical_label = canonicalize_object_label(label)
        count = max(
            counts.get(label, 0),
            counts.get(canonical_label, 0),
        )
        scores.append(1.0 / (1.0 + abs(count - object_filter.min_count)))
    return True, sum(scores) / len(scores)


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
    labels = _row_entity_labels(row)
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


def _compute_result_component_scores(
    *,
    row: dict[str, object],
    text_terms: list[str],
    dense_score: float,
    text_score: float,
    object_score: float,
    temporal_score: float,
) -> dict[str, float]:
    return {
        "dense_score": dense_score,
        "text_score": text_score,
        "ocr_score": _lexical_score(text_terms, str(row.get("ocr_text", ""))) if row.get("ocr_text") else 0.0,
        "object_score": object_score,
        "entity_score": _entity_score(row, text_terms),
        "temporal_score": temporal_score,
    }


def _filter_display_results(rows: list[dict[str, object]], threshold: float, limit: int) -> list[dict[str, object]]:
    kept = [
        row for row in rows
        if float(row.get("score", 0.0) or 0.0) >= threshold
    ]
    return kept[:limit]


def filter_result_payloads(rows: list[dict[str, object]], threshold: float, limit: int | None = None) -> list[dict[str, object]]:
    return _filter_display_results(
        rows,
        threshold=threshold,
        limit=limit if limit is not None else settings.search_result_display_limit,
    )


def _embed_query_image(image_path) -> list[float]:
    return OpenClipAdapter().embed_image(str(image_path)).values


def _vector_similarity_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def build_visual_search_result(
    *,
    row: dict[str, object],
    keyframe_id: int,
    media: dict[str, str],
    score: float,
) -> dict[str, object]:
    return {
        "segment_id": int(row["segment_id"]),
        "video_id": int(row["video_id"]),
        "segment_index": int(row["segment_index"]),
        "frame_id": keyframe_id,
        "timestamp_sec": float(row["start_timestamp_sec"]),
        "start_timestamp_sec": float(row["start_timestamp_sec"]),
        "end_timestamp_sec": float(row["end_timestamp_sec"]),
        "score": score,
        "object_labels": _row_display_labels(row),
        "object_counts": _row_display_counts(row),
        "object_positions": {key: sorted(value) for key, value in _row_semantic_positions(row).items()},
        "caption": str(row.get("caption_text", "")),
        "thumb_url": f"/media/frames/{keyframe_id}/thumb" if media else "",
        "image_url": f"/media/frames/{keyframe_id}/image" if media else "",
        "preview_url": f"/media/frames/{keyframe_id}/preview" if media else "",
        "_image_path": str(media.get("image_path", "")) if media else "",
    }


def _apply_openai_vision_rerank(
    query: str,
    results: list[dict[str, object]],
    *,
    use_openai_rerank: bool | None = None,
) -> list[dict[str, object]]:
    if not should_run_openai_vision_rerank(
        enabled=(settings.openai_enabled and settings.openai_vision_rerank_enabled) if use_openai_rerank is None else (settings.openai_enabled and use_openai_rerank),
        api_key=settings.openai_api_key,
        candidates=results,
    ):
        return results

    top_candidates = select_rerank_candidates(results, settings.openai_vision_rerank_top_k)
    vision_scores = run_openai_vision_rerank(query, top_candidates)
    updated: list[dict[str, object]] = []
    for index, row in enumerate(results):
        frame_id = int(row.get("frame_id") or -1)
        if index < settings.openai_vision_rerank_top_k and frame_id in vision_scores:
            local_score = float(row.get("score", 0.0) or 0.0)
            row = {
                **row,
                "score": blend_rerank_score(
                    local_score=local_score,
                    vision_score=float(vision_scores[frame_id]),
                    local_weight=settings.openai_vision_rerank_local_weight,
                    vision_weight=settings.openai_vision_rerank_vision_weight,
                ),
            }
        updated.append(row)
    return sorted(updated, key=lambda item: float(item.get("score", 0.0)), reverse=True)


def run_image_search(db: Session, image_path) -> dict[str, object]:
    query_embedding = _embed_query_image(image_path)
    rows = search_segment_candidates(db, query_embedding, limit=80)
    frame_media = fetch_frame_media_map(
        db,
        [int(row["keyframe_id"]) for row in rows if row["keyframe_id"] is not None],
    )

    results: list[dict[str, object]] = []
    for row in rows:
        keyframe_id = int(row["keyframe_id"]) if row["keyframe_id"] is not None else None
        if keyframe_id is None:
            continue
        media = frame_media.get(keyframe_id, {})
        results.append(
            build_visual_search_result(
                row=row,
                keyframe_id=keyframe_id,
                media=media,
                score=_vector_similarity_score(float(row.get("vector_distance", 0.0) or 0.0)),
            )
        )

    results = filter_result_payloads(results, threshold=settings.image_result_score_threshold)
    results = [{key: value for key, value in item.items() if key != "_image_path"} for item in results]

    return {
        "mode": "image",
        "query": getattr(image_path, "name", "query-image"),
        "expanded_queries": [],
        "results": results,
        "parsed_query": None,
    }


def run_search(
    db: Session,
    query: str,
    object_labels: list[str],
    *,
    use_openai_rerank: bool | None = None,
) -> dict[str, object]:
    total_started = perf_counter()
    debug_metrics: dict[str, object] = {
        "rss_mb": _current_rss_mb(),
        "openai_enabled": bool(settings.openai_enabled),
        "openai_requested": bool(use_openai_rerank) if use_openai_rerank is not None else True,
        "stage_timings": {},
    }
    use_openai_features = (
        settings.openai_api_key
        if settings.openai_enabled and (use_openai_rerank is None or use_openai_rerank)
        else ""
    )
    parse_started = perf_counter()
    structured = parse_structured_query(query, api_key=use_openai_features, model=settings.openai_model)
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "structured_query", parse_started)
    object_filters = _merge_object_filters(structured, object_labels)
    expanded = structured.semantic_queries or [structured.semantic_query]
    expand_started = perf_counter()
    if use_openai_features:
        for item in expand_query(structured.semantic_query, api_key=use_openai_features, model=settings.openai_model):
            if item not in expanded:
                expanded.append(item)
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "query_expansion", expand_started)

    collect_started = perf_counter()
    branch_rows = collect_branch_candidates(
        db,
        semantic_query=structured.semantic_query,
        expanded_queries=expanded,
        object_filters=object_filters,
        temporal_steps=structured.temporal_steps,
        dense_encoder=_get_search_dense_encoder(),
    )
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "collect_branch_candidates", collect_started)
    if not any(branch_rows.values()):
        commit_started = perf_counter()
        create_query_log(db, query, expanded, object_labels, [])
        db.commit()
        if settings.enable_stage_timing:
            _record_search_stage(debug_metrics, "query_log_commit", commit_started)
            _record_search_stage(debug_metrics, "total", total_started)
            debug_metrics["rss_mb"] = _current_rss_mb()
        return {
            "query": query,
            "expanded_queries": expanded,
            "results": [],
            "parsed_query": structured.model_dump(),
            "debug_metrics": debug_metrics,
        }

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

    rankings_started = perf_counter()
    rankings, diagnostics, passes_filters = _candidate_rankings(rows, expanded, object_filters)
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "candidate_rankings", rankings_started)
    for branch_name, ranking in branch_rankings.items():
        if ranking:
            rankings[branch_name] = ranking
    fused_scores = fuse_branch_rankings(rankings)
    temporal_bonus = _temporal_path_scores(rows, structured.temporal_steps)
    normalized_dense_scores = _normalize_score_map(fused_scores)
    normalized_temporal_scores = _normalize_score_map(temporal_bonus)
    media_started = perf_counter()
    frame_media = fetch_frame_media_map(
        db,
        [int(row["keyframe_id"]) for row in rows if row["keyframe_id"] is not None],
    )
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "fetch_frame_media", media_started)
    refinement_scores: dict[int, float] = {}
    query_object_terms = [item.label for item in object_filters]
    refinement_started = perf_counter()
    if query_object_terms:
        candidate_frames = []
        for row in sorted(rows, key=lambda item: normalized_dense_scores.get(int(item["segment_id"]), 0.0), reverse=True)[:12]:
            keyframe_id = int(row["keyframe_id"]) if row["keyframe_id"] is not None else None
            media = frame_media.get(keyframe_id or -1, {})
            image_path = str(media.get("image_path", "")).strip()
            if keyframe_id is not None and image_path:
                candidate_frames.append({"frame_id": keyframe_id, "image_path": image_path})
        refinement_scores = refine_object_matches(candidate_frames, query_object_terms)
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "object_refinement", refinement_started)

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
    llm_alignment_started = perf_counter()
    llm_alignment_scores = rerank_structured_candidates(
        structured,
        rerank_payload,
        api_key=use_openai_features,
        model=settings.openai_model,
    )
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "llm_alignment", llm_alignment_started)

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
                "_image_path": str(media.get("image_path", "")) if keyframe_id is not None and media else "",
                "diagnostics": {
                    **diagnostics.get(segment_id, {}),
                    "object_refinement_score": refinement_scores.get(keyframe_id or -1, 0.0),
                    "llm_alignment_score": llm_alignment_scores.get(segment_id, 0.0),
                    "constraint_mode": "hard" if enforce_hard_filter else "soft",
                    "object_detector_family": str(row.get("object_detector_family", "")),
                    "object_detector_model": str(row.get("object_detector_model", "")),
                },
            }
        )

    local_scoring_started = perf_counter()
    text_terms = _tokenize(" ".join(expanded))
    for item in ranked_segments:
        segment_id = int(item["segment_id"])
        count_refine_active, count_refine_score = _row_count_refine_score(rows_by_segment[segment_id], object_filters)
        component_scores = _compute_result_component_scores(
            row=rows_by_segment[segment_id],
            text_terms=text_terms,
            dense_score=normalized_dense_scores.get(segment_id, 0.0),
            text_score=diagnostics.get(segment_id, {}).get("text_score", 0.0),
            object_score=max(
                diagnostics.get(segment_id, {}).get("object_score", 0.0),
                refinement_scores.get(int(item.get("frame_id") or -1), 0.0),
            ),
            temporal_score=normalized_temporal_scores.get(segment_id, 0.0),
        )
        base = {
            **item,
            **component_scores,
            "hard_constraints_passed": passes_filters.get(segment_id, True) if enforce_hard_filter else True,
            "count_refine_active": count_refine_active,
            "count_refine_score": count_refine_score,
        }
        item["score"] = apply_constraint_penalty(
            base,
            score_local_candidate(base) + (0.12 * llm_alignment_scores.get(segment_id, 0.0)),
        )
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "local_scoring", local_scoring_started)

    ranked_segments = sorted(ranked_segments, key=lambda row: float(row["score"]), reverse=True)
    results = [item for item in ranked_segments if item.get("frame_id")]
    display_started = perf_counter()
    results = _filter_display_results(
        results,
        threshold=settings.text_result_score_threshold,
        limit=settings.search_result_display_limit,
    )
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "display_filter", display_started)
    vision_started = perf_counter()
    results = _apply_openai_vision_rerank(query, results, use_openai_rerank=use_openai_rerank)
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "vision_rerank", vision_started)
    results = [{key: value for key, value in item.items() if key != "_image_path"} for item in results]
    commit_started = perf_counter()
    create_query_log(db, query, expanded, object_labels, results)
    db.commit()
    if settings.enable_stage_timing:
        _record_search_stage(debug_metrics, "query_log_commit", commit_started)
        _record_search_stage(debug_metrics, "total", total_started)
        debug_metrics["rss_mb"] = _current_rss_mb()
    return {
        "query": query,
        "expanded_queries": expanded,
        "results": results,
        "parsed_query": structured.model_dump(),
        "debug_metrics": debug_metrics,
    }
