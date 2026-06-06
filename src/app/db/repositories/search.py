from collections import defaultdict
from math import sqrt

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Frame, QueryLog, Segment


def _l2_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return float("inf")
    size = min(len(left), len(right))
    return sqrt(sum((left[index] - right[index]) ** 2 for index in range(size)))


def search_segment_candidates(
    db: Session,
    query_embedding: list[float],
    limit: int = 80,
) -> list[dict[str, object]]:
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        distance = Segment.embedding.l2_distance(query_embedding)
        rows = db.execute(
            select(
                Segment.id.label("segment_id"),
                Segment.video_id,
                Segment.segment_index,
                Segment.start_timestamp_sec,
                Segment.end_timestamp_sec,
                Segment.keyframe_id,
                Segment.caption_text,
                Segment.ocr_text,
                Segment.object_labels_json,
                Segment.object_counts_json,
                Segment.object_positions_json,
                Segment.semantic_entities_json,
                Segment.semantic_counts_json,
                distance.label("distance"),
            )
            .where(Segment.embedding.is_not(None))
            .where(func.vector_dims(Segment.embedding) == len(query_embedding))
            .order_by(distance.asc(), Segment.id.asc())
            .limit(limit)
        ).all()
        return [
            {
                "segment_id": int(row.segment_id),
                "video_id": int(row.video_id),
                "segment_index": int(row.segment_index),
                "start_timestamp_sec": float(row.start_timestamp_sec),
                "end_timestamp_sec": float(row.end_timestamp_sec),
                "keyframe_id": int(row.keyframe_id) if row.keyframe_id is not None else None,
                "caption_text": row.caption_text or "",
                "ocr_text": row.ocr_text or "",
                "labels": list(row.object_labels_json or []),
                "object_counts": dict(row.object_counts_json or {}),
                "object_positions": {str(key): list(value) for key, value in dict(row.object_positions_json or {}).items()},
                "semantic_entities": list(row.semantic_entities_json or []),
                "semantic_counts": dict(row.semantic_counts_json or {}),
                "vector_distance": float(row.distance),
            }
            for row in rows
        ]

    rows = db.execute(
        select(
            Segment.id,
            Segment.video_id,
            Segment.segment_index,
            Segment.start_timestamp_sec,
            Segment.end_timestamp_sec,
            Segment.keyframe_id,
            Segment.caption_text,
            Segment.ocr_text,
            Segment.object_labels_json,
            Segment.object_counts_json,
            Segment.object_positions_json,
            Segment.semantic_entities_json,
            Segment.semantic_counts_json,
            Segment.embedding,
        ).where(Segment.embedding.is_not(None))
    ).all()
    candidates = [
        {
            "segment_id": int(row.id),
            "video_id": int(row.video_id),
            "segment_index": int(row.segment_index),
            "start_timestamp_sec": float(row.start_timestamp_sec),
            "end_timestamp_sec": float(row.end_timestamp_sec),
            "keyframe_id": int(row.keyframe_id) if row.keyframe_id is not None else None,
            "caption_text": row.caption_text or "",
            "ocr_text": row.ocr_text or "",
            "labels": list(row.object_labels_json or []),
            "object_counts": dict(row.object_counts_json or {}),
            "object_positions": {str(key): list(value) for key, value in dict(row.object_positions_json or {}).items()},
            "semantic_entities": list(row.semantic_entities_json or []),
            "semantic_counts": dict(row.semantic_counts_json or {}),
            "vector_distance": _l2_distance(list(row.embedding or []), query_embedding),
        }
        for row in rows
        if len(list(row.embedding or [])) == len(query_embedding)
    ]
    return sorted(candidates, key=lambda item: (float(item["vector_distance"]), int(item["segment_id"])))[:limit]


def fetch_frame_media_map(db: Session, frame_ids: list[int]) -> dict[int, dict[str, str]]:
    if not frame_ids:
        return {}
    rows = db.execute(
        select(Frame.id, Frame.image_path, Frame.thumb_path).where(Frame.id.in_(frame_ids))
    ).all()
    return {
        int(row.id): {
            "image_path": str(row.image_path),
            "thumb_path": str(row.thumb_path),
        }
        for row in rows
    }


def fetch_segment_neighbors(db: Session, segment_ids: list[int], radius: int = 1) -> dict[int, list[dict[str, object]]]:
    if not segment_ids:
        return {}
    segments = {
        int(segment.id): segment
        for segment in db.execute(
            select(Segment).where(Segment.id.in_(segment_ids))
        ).scalars()
    }
    neighbor_map: dict[int, list[dict[str, object]]] = defaultdict(list)
    for segment_id, segment in segments.items():
        rows = db.execute(
            select(Segment)
            .where(Segment.video_id == segment.video_id)
            .where(Segment.segment_index >= segment.segment_index - radius)
            .where(Segment.segment_index <= segment.segment_index + radius)
            .where(Segment.id != segment.id)
            .order_by(Segment.segment_index.asc())
        ).scalars()
        for neighbor in rows:
            neighbor_map[segment_id].append(
                {
                    "segment_id": int(neighbor.id),
                    "video_id": int(neighbor.video_id),
                    "segment_index": int(neighbor.segment_index),
                    "start_timestamp_sec": float(neighbor.start_timestamp_sec),
                    "end_timestamp_sec": float(neighbor.end_timestamp_sec),
                    "keyframe_id": int(neighbor.keyframe_id) if neighbor.keyframe_id is not None else None,
                    "caption_text": neighbor.caption_text or "",
                    "ocr_text": neighbor.ocr_text or "",
                    "labels": list(neighbor.object_labels_json or []),
                    "object_counts": dict(neighbor.object_counts_json or {}),
                    "object_positions": {str(key): list(value) for key, value in dict(neighbor.object_positions_json or {}).items()},
                    "semantic_entities": list(neighbor.semantic_entities_json or []),
                    "semantic_counts": dict(neighbor.semantic_counts_json or {}),
                }
            )
    return neighbor_map


def create_query_log(
    db: Session,
    original_query: str,
    expanded_queries: list[str],
    object_labels: list[str],
    results: list[dict[str, object]],
) -> QueryLog:
    log = QueryLog(
        original_query=original_query,
        expanded_queries=expanded_queries,
        filters_json={"object_labels": object_labels},
        weights_json={"strategy": "segment_rrf_v1"},
        top_results_json=results,
    )
    db.add(log)
    db.flush()
    return log
