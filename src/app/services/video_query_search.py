from __future__ import annotations

import shutil
import tempfile
from collections import defaultdict
from mimetypes import guess_type
from pathlib import Path

import cv2
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.db.repositories.search import fetch_frame_media_map, search_segment_candidates
from app.services.openai_vision_rerank import (
    blend_rerank_score,
    run_openai_vision_rerank,
    select_rerank_candidates,
    should_run_openai_vision_rerank,
)
from app.services.search_service import _get_search_dense_encoder, build_visual_search_result, filter_result_payloads


def validate_query_clip_duration(duration_sec: float, max_duration_sec: float) -> bool:
    return duration_sec <= max_duration_sec


def aggregate_query_frame_scores(per_frame_hits: list[dict[int, float]]) -> list[tuple[int, float]]:
    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    total_queries = max(len(per_frame_hits), 1)

    for hit_map in per_frame_hits:
        for candidate_id, score in hit_map.items():
            totals[candidate_id] += score
            counts[candidate_id] += 1

    ranked: list[tuple[int, float]] = []
    for candidate_id, total in totals.items():
        consistency_bonus = counts[candidate_id] / total_queries
        ranked.append((candidate_id, total + (0.15 * consistency_bonus)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _vector_similarity_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def _normalize_public_scores(results: list[dict[str, object]]) -> list[dict[str, object]]:
    if not results:
        return results
    max_score = max(float(item.get("score", 0.0) or 0.0) for item in results)
    if max_score <= 0:
        return [{**item, "score": 0.0} for item in results]
    return [
        {
            **item,
            "score": round(float(item.get("score", 0.0) or 0.0) / max_score, 3),
        }
        for item in results
    ]


def _save_upload_to_temp(upload: UploadFile, work_dir: Path) -> Path:
    suffix = Path(upload.filename or "query.mp4").suffix or ".mp4"
    clip_path = work_dir / f"query{suffix}"
    with clip_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return clip_path


def _probe_video_duration(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Uploaded clip could not be read.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0 or frame_count <= 0:
            raise HTTPException(status_code=400, detail="Uploaded clip could not be read.")
        return frame_count / fps
    finally:
        capture.release()


def _extract_query_frames(video_path: Path, output_dir: Path, frame_count: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Uploaded clip could not be read.")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            raise HTTPException(status_code=400, detail="No query frames could be extracted from this clip.")
        sample_count = max(1, min(frame_count, total_frames))
        indices = sorted({round(index * (total_frames - 1) / max(sample_count - 1, 1)) for index in range(sample_count)})
        extracted: list[Path] = []
        for ordinal, frame_index in enumerate(indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue
            frame_path = output_dir / f"query_frame_{ordinal:02d}.png"
            cv2.imwrite(str(frame_path), frame)
            extracted.append(frame_path)
        if not extracted:
            raise HTTPException(status_code=400, detail="No query frames could be extracted from this clip.")
        return extracted
    finally:
        capture.release()


def _collect_local_candidates(db: Session, query_frame_paths: list[Path]) -> list[dict[str, object]]:
    embeddings = _get_search_dense_encoder().embed_images([str(path) for path in query_frame_paths])
    rows_by_segment: dict[int, dict[str, object]] = {}
    per_query_scores: list[dict[int, float]] = []

    for embedding in embeddings:
        rows = search_segment_candidates(
            db,
            embedding.values,
            limit=settings.video_query_local_candidate_pool,
        )
        hit_map: dict[int, float] = {}
        for row in rows:
            segment_id = int(row["segment_id"])
            hit_map[segment_id] = _vector_similarity_score(float(row.get("vector_distance", 0.0) or 0.0))
            existing = rows_by_segment.setdefault(segment_id, dict(row))
            for key, value in row.items():
                if key not in existing or existing[key] in (None, "", [], {}):
                    existing[key] = value
        per_query_scores.append(hit_map)

    ranked_segment_ids = [segment_id for segment_id, _score in aggregate_query_frame_scores(per_query_scores)]
    ranked_rows = [rows_by_segment[segment_id] for segment_id in ranked_segment_ids if segment_id in rows_by_segment]
    frame_media = fetch_frame_media_map(
        db,
        [int(row["keyframe_id"]) for row in ranked_rows if row["keyframe_id"] is not None],
    )

    results: list[dict[str, object]] = []
    for segment_id, aggregate_score in aggregate_query_frame_scores(per_query_scores):
        row = rows_by_segment.get(segment_id)
        if row is None:
            continue
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
                score=aggregate_score,
            )
        )
    return results


def _apply_video_query_openai_rerank(
    query_frame_paths: list[Path],
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
    vision_scores = run_openai_vision_rerank(
        query="video-query",
        candidates=top_candidates,
        query_image_paths=query_frame_paths,
    )
    if not vision_scores:
        return results

    updated: list[dict[str, object]] = []
    for index, row in enumerate(results):
        frame_id = int(row.get("frame_id") or -1)
        if index < settings.openai_vision_rerank_top_k and frame_id in vision_scores:
            row = {
                **row,
                "score": blend_rerank_score(
                    local_score=float(row.get("score", 0.0) or 0.0),
                    vision_score=float(vision_scores[frame_id]),
                    local_weight=settings.openai_vision_rerank_local_weight,
                    vision_weight=settings.openai_vision_rerank_vision_weight,
                ),
            }
        updated.append(row)
    return sorted(updated, key=lambda item: float(item.get("score", 0.0)), reverse=True)


async def run_video_query_search(
    db: Session,
    upload: UploadFile,
    *,
    use_openai_rerank: bool | None = None,
) -> dict[str, object]:
    guessed_type = guess_type(upload.filename or "query.mp4")[0] or ""
    content_type = str(upload.content_type or "")
    if not content_type.startswith("video/") and not guessed_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video clip.")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="video-query-", dir=str(settings.data_dir)))
    clip_path: Path | None = None
    try:
        clip_path = _save_upload_to_temp(upload, temp_root)
        duration_sec = _probe_video_duration(clip_path)
        if not validate_query_clip_duration(duration_sec, settings.video_query_max_duration_sec):
            raise HTTPException(
                status_code=400,
                detail=f"Query clip must be {settings.video_query_max_duration_sec:g} seconds or shorter.",
            )

        query_frame_paths = _extract_query_frames(
            clip_path,
            temp_root / "frames",
            frame_count=settings.video_query_frame_count,
        )
        results = _collect_local_candidates(db, query_frame_paths)
        results = _apply_video_query_openai_rerank(
            query_frame_paths,
            results,
            use_openai_rerank=use_openai_rerank,
        )
        results = filter_result_payloads(results, threshold=settings.image_result_score_threshold)
        results = _normalize_public_scores(results)
        results = [{key: value for key, value in item.items() if key != "_image_path"} for item in results]
        return {
            "mode": "video",
            "query": upload.filename or "query-video",
            "expanded_queries": [],
            "results": results,
            "parsed_query": None,
        }
    finally:
        await upload.close()
        shutil.rmtree(temp_root, ignore_errors=True)
