from __future__ import annotations

from math import sqrt

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.db.models import Segment
from app.services.query_understanding import ObjectFilter, TemporalStep
from worker.retrieval_ontology import canonicalize_object_label


def _l2_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return float("inf")
    size = min(len(left), len(right))
    return sqrt(sum((left[index] - right[index]) ** 2 for index in range(size)))


def _segment_row_to_dict(row) -> dict[str, object]:
    raw_json = dict(row.raw_json or {})
    return {
        "segment_id": int(row.id),
        "video_id": int(row.video_id),
        "segment_index": int(row.segment_index),
        "start_timestamp_sec": float(row.start_timestamp_sec),
        "end_timestamp_sec": float(row.end_timestamp_sec),
        "keyframe_id": int(row.keyframe_id) if row.keyframe_id is not None else None,
        "caption_text": row.caption_text or "",
        "ocr_text": row.ocr_text or "",
        "ocr_tokens": list(row.ocr_tokens_json or []),
        "labels": list(row.object_labels_json or []),
        "object_counts": dict(row.object_counts_json or {}),
        "object_positions": {str(key): list(value) for key, value in dict(row.object_positions_json or {}).items()},
        "semantic_entities": list(row.semantic_entities_json or []),
        "semantic_aliases": dict(row.semantic_aliases_json or {}),
        "semantic_counts": dict(row.semantic_counts_json or {}),
        "object_detector_family": str(raw_json.get("object_detector_family", "")),
        "object_detector_model": str(raw_json.get("object_detector_model", "")),
    }


def _branch_segment_select():
    return select(
        Segment.id.label("segment_id"),
        Segment.video_id,
        Segment.segment_index,
        Segment.start_timestamp_sec,
        Segment.end_timestamp_sec,
        Segment.keyframe_id,
        Segment.caption_text,
        Segment.ocr_text,
        Segment.ocr_tokens_json,
        Segment.object_labels_json,
        Segment.object_counts_json,
        Segment.object_positions_json,
        Segment.semantic_entities_json,
        Segment.semantic_aliases_json,
        Segment.semantic_counts_json,
        Segment.raw_json,
    )


def _branch_row_to_dict(row) -> dict[str, object]:
    raw_json = dict(row.raw_json or {})
    return {
        "segment_id": int(row.segment_id),
        "video_id": int(row.video_id),
        "segment_index": int(row.segment_index),
        "start_timestamp_sec": float(row.start_timestamp_sec),
        "end_timestamp_sec": float(row.end_timestamp_sec),
        "keyframe_id": int(row.keyframe_id) if row.keyframe_id is not None else None,
        "caption_text": row.caption_text or "",
        "ocr_text": row.ocr_text or "",
        "ocr_tokens": list(row.ocr_tokens_json or []),
        "labels": list(row.object_labels_json or []),
        "object_counts": dict(row.object_counts_json or {}),
        "object_positions": {str(key): list(value) for key, value in dict(row.object_positions_json or {}).items()},
        "semantic_entities": list(row.semantic_entities_json or []),
        "semantic_aliases": dict(row.semantic_aliases_json or {}),
        "semantic_counts": dict(row.semantic_counts_json or {}),
        "object_detector_family": str(raw_json.get("object_detector_family", "")),
        "object_detector_model": str(raw_json.get("object_detector_model", "")),
    }


def search_dense_branch(
    db: Session,
    query_embedding: list[float],
    column_name: str,
    limit: int = 80,
) -> list[dict[str, object]]:
    column = getattr(Segment, column_name)
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        distance = column.l2_distance(query_embedding)
        rows = db.execute(
            select(Segment, distance.label("distance"))
            .where(column.is_not(None))
            .where(func.vector_dims(column) == len(query_embedding))
            .order_by(distance.asc(), Segment.id.asc())
            .limit(limit)
        ).all()
        return [{**_segment_row_to_dict(row.Segment), "vector_distance": float(row.distance), "branch": column_name} for row in rows]

    rows = db.execute(select(Segment).where(column.is_not(None))).scalars().all()
    candidates = []
    for row in rows:
        vector = list(getattr(row, column_name) or [])
        if len(vector) != len(query_embedding):
            continue
        candidates.append(
            {
                **_segment_row_to_dict(row),
                "vector_distance": _l2_distance(vector, query_embedding),
                "branch": column_name,
            }
        )
    return sorted(candidates, key=lambda item: (float(item["vector_distance"]), int(item["segment_id"])))[:limit]


def search_text_branch(db: Session, query_terms: list[str], limit: int = 80) -> list[dict[str, object]]:
    if not query_terms:
        return []
    rows = db.execute(
        _branch_segment_select().where(
            or_(*([Segment.caption_text.ilike(f"%{term}%") for term in query_terms] + [Segment.ocr_text.ilike(f"%{term}%") for term in query_terms]))
        )
    ).all()
    scored = []
    for row in rows:
        haystack = f"{row.caption_text or ''} {row.ocr_text or ''}".lower()
        score = sum(1 for term in query_terms if term.lower() in haystack)
        scored.append({**_branch_row_to_dict(row), "text_score": score / len(query_terms), "branch": "text"})
    return sorted(scored, key=lambda item: (float(item["text_score"]), -int(item["segment_id"])), reverse=True)[:limit]


def search_object_branch(db: Session, object_filters: list[ObjectFilter], limit: int = 80) -> list[dict[str, object]]:
    if not object_filters:
        return []
    rows = db.execute(_branch_segment_select()).all()
    matched: list[dict[str, object]] = []
    for row in rows:
        counts = {str(key).lower(): int(value) for key, value in dict(row.object_counts_json or {}).items()}
        aliases = {str(key).lower(): [str(item).lower() for item in value] for key, value in dict(row.semantic_aliases_json or {}).items()}
        score = 0.0
        passed = True
        for item in object_filters:
            label = item.label.lower()
            canonical_label = canonicalize_object_label(label)
            alias_hits = [name for name, values in aliases.items() if label == name or label in values]
            count = max(counts.get(label, 0), counts.get(canonical_label, 0))
            if alias_hits:
                count = max([count, *[counts.get(name, 0) for name in alias_hits]])
            if count < item.min_count:
                passed = False
                break
            if item.max_count is not None and count > item.max_count:
                passed = False
                break
            if item.max_count is not None:
                score += 1.0
            elif item.min_count > 1:
                score += 1.0 / (1.0 + abs(count - item.min_count))
            else:
                score += min(count / max(item.min_count, 1), 1.0)
        if passed:
            matched.append({**_branch_row_to_dict(row), "object_score": score / len(object_filters), "branch": "object"})
    return sorted(matched, key=lambda item: (float(item["object_score"]), -int(item["segment_id"])), reverse=True)[:limit]


def search_temporal_seed_branch(db: Session, temporal_steps: list[TemporalStep], limit: int = 80) -> list[dict[str, object]]:
    if not temporal_steps:
        return []
    query_terms = [term for step in temporal_steps for term in step.text.lower().split() if term]
    return search_text_branch(db, query_terms, limit=limit)
