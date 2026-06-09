from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Segment
from app.db.repositories.search import fetch_frame_media_map
from app.services.openai_vision_rerank import (
    blend_rerank_score,
    run_openai_vision_rerank,
    select_rerank_candidates,
    should_run_openai_vision_rerank,
)
from app.services.search_service import build_visual_search_result, filter_result_payloads


def build_question_evidence_terms(question: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9]+", question.lower()):
        if len(token) <= 2 or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def score_evidence_row(row: dict[str, object], evidence_terms: list[str]) -> float:
    if not evidence_terms:
        return 0.0
    ocr_text = str(row.get("ocr_text", "")).lower()
    caption_text = str(row.get("caption_text", "")).lower()
    ocr_hits = sum(1 for term in evidence_terms if term in ocr_text)
    caption_hits = sum(1 for term in evidence_terms if term in caption_text)
    return min(1.0, ((ocr_hits * 1.5) + caption_hits) / max(len(evidence_terms), 1))


def rank_question_candidates(rows: list[dict[str, object]], question: str) -> list[dict[str, object]]:
    evidence_terms = build_question_evidence_terms(question)
    ranked = []
    for row in rows:
        ranked.append(
            {
                **row,
                "_question_local_score": score_evidence_row(row, evidence_terms),
            }
        )
    return sorted(ranked, key=lambda item: float(item["_question_local_score"]), reverse=True)


def _load_segment_rows(db: Session) -> list[dict[str, object]]:
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
        )
    ).all()
    return [
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
        }
        for row in rows
    ]


def run_question_search(
    db: Session,
    question: str,
    *,
    use_openai_rerank: bool | None = None,
) -> dict[str, object]:
    query = question.strip()
    if not query:
        return {
            "mode": "question",
            "query": question,
            "expanded_queries": [],
            "results": [],
            "parsed_query": None,
        }

    ranked_rows = rank_question_candidates(_load_segment_rows(db), query)
    ranked_rows = [row for row in ranked_rows if float(row.get("_question_local_score", 0.0)) > 0][: settings.question_search_candidate_pool]
    frame_media = fetch_frame_media_map(
        db,
        [int(row["keyframe_id"]) for row in ranked_rows if row["keyframe_id"] is not None],
    )

    results: list[dict[str, object]] = []
    for row in ranked_rows:
        keyframe_id = int(row["keyframe_id"]) if row["keyframe_id"] is not None else None
        if keyframe_id is None:
            continue
        media = frame_media.get(keyframe_id, {})
        if not media:
            continue
        results.append(
            build_visual_search_result(
                row=row,
                keyframe_id=keyframe_id,
                media=media,
                score=float(row.get("_question_local_score", 0.0)),
            )
        )

    if should_run_openai_vision_rerank(
        enabled=(settings.openai_enabled and settings.openai_vision_rerank_enabled) if use_openai_rerank is None else (settings.openai_enabled and use_openai_rerank),
        api_key=settings.openai_api_key,
        candidates=results,
    ):
        top_candidates = select_rerank_candidates(results, settings.question_search_rerank_top_k)
        vision_scores = run_openai_vision_rerank(
            query,
            top_candidates,
            instruction=(
                "You are reranking candidate video frames for question answering. "
                "Score each frame from 0.0 to 1.0 based on how useful it would be for answering the user's question. "
                "Prefer frames that visibly contain text, overlays, screens, captions, or evidence relevant to the answer. "
                "Return JSON only with the shape "
                '{"items":[{"frame_id":123,"vision_score":0.91}]}. '
                f"User question: {query}"
            ),
        )
        if vision_scores:
            reranked: list[dict[str, object]] = []
            for index, row in enumerate(results):
                frame_id = int(row.get("frame_id") or -1)
                if index < settings.question_search_rerank_top_k and frame_id in vision_scores:
                    row = {
                        **row,
                        "score": blend_rerank_score(
                            local_score=float(row.get("score", 0.0) or 0.0),
                            vision_score=float(vision_scores[frame_id]),
                            local_weight=settings.openai_vision_rerank_local_weight,
                            vision_weight=settings.openai_vision_rerank_vision_weight,
                        ),
                    }
                reranked.append(row)
            results = sorted(reranked, key=lambda item: float(item.get("score", 0.0)), reverse=True)

    results = filter_result_payloads(results, threshold=settings.text_result_score_threshold)
    results = [{key: value for key, value in item.items() if key != "_image_path"} for item in results]

    return {
        "mode": "question",
        "query": question,
        "expanded_queries": [],
        "results": results,
        "parsed_query": {"evidence_terms": build_question_evidence_terms(query)},
    }
