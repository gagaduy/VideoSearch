import asyncio
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from app.services import video_query_search
from app.services.video_query_search import (
    _extract_query_frames,
    _collect_local_candidates,
    _normalize_public_scores,
    aggregate_query_frame_scores,
    run_video_query_search,
    validate_query_clip_duration,
)


def test_validate_query_clip_duration_rejects_long_clips() -> None:
    assert validate_query_clip_duration(duration_sec=12.0, max_duration_sec=10.0) is False


def test_validate_query_clip_duration_accepts_short_clips() -> None:
    assert validate_query_clip_duration(duration_sec=8.5, max_duration_sec=10.0) is True


def test_aggregate_query_frame_scores_rewards_repeat_matches() -> None:
    per_frame_hits = [
        {11: 0.82, 18: 0.40},
        {11: 0.75, 27: 0.50},
        {11: 0.79, 18: 0.41},
    ]

    ranked = aggregate_query_frame_scores(per_frame_hits)

    assert ranked[0][0] == 11
    assert ranked[0][1] > ranked[1][1]


def test_normalize_public_scores_maps_video_results_to_zero_one_scale() -> None:
    rows = [
        {"frame_id": 6, "score": 3.545},
        {"frame_id": 531, "score": 2.257},
        {"frame_id": 653, "score": 2.241},
    ]

    normalized = _normalize_public_scores(rows)

    assert normalized[0]["score"] == 1.0
    assert 0.0 <= normalized[1]["score"] <= 1.0
    assert 0.0 <= normalized[2]["score"] <= 1.0
    assert [item["frame_id"] for item in normalized] == [6, 531, 653]


def test_extract_query_frames_returns_multiple_samples(tmp_path: Path) -> None:
    video_path = tmp_path / "query.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        (32, 32),
    )
    for index in range(8):
        frame = np.full((32, 32, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    extracted = _extract_query_frames(video_path, tmp_path / "frames", frame_count=4)

    assert len(extracted) == 4
    assert all(path.exists() for path in extracted)


def test_run_video_query_search_returns_local_results_when_rerank_falls_back(monkeypatch, tmp_path: Path) -> None:
    query_path = tmp_path / "query.mp4"
    query_path.write_bytes(b"video-bytes")

    monkeypatch.setattr(
        "app.services.video_query_search._save_upload_to_temp",
        lambda upload, work_dir: query_path,
    )
    monkeypatch.setattr("app.services.video_query_search._probe_video_duration", lambda path: 4.0)
    monkeypatch.setattr(
        "app.services.video_query_search._extract_query_frames",
        lambda video_path, output_dir, frame_count: [tmp_path / "f1.png", tmp_path / "f2.png"],
    )
    monkeypatch.setattr(
        "app.services.video_query_search._collect_local_candidates",
        lambda db, query_frame_paths: [{"frame_id": 11, "score": 0.7, "_image_path": "/tmp/frame.png"}],
    )
    monkeypatch.setattr(
        "app.services.video_query_search._apply_video_query_openai_rerank",
        lambda query_frame_paths, results, **kwargs: results,
    )

    upload = UploadFile(filename="query.mp4", file=query_path.open("rb"))
    payload = asyncio.run(run_video_query_search(db=None, upload=upload))

    assert payload["mode"] == "video"
    assert payload["results"][0]["frame_id"] == 11
    assert "_image_path" not in payload["results"][0]


def test_collect_local_candidates_reuses_shared_dense_encoder(monkeypatch, tmp_path: Path) -> None:
    frame_path = tmp_path / "query_frame.png"
    frame_path.write_bytes(b"frame")

    class _SharedEncoder:
        def embed_images(self, image_paths: list[str]):
            assert image_paths == [str(frame_path)]
            return [type("Embedding", (), {"values": [0.1, 0.2, 0.3], "model_name": "stub"})()]

    monkeypatch.setattr(video_query_search, "_get_search_dense_encoder", lambda: _SharedEncoder())
    monkeypatch.setattr(
        video_query_search,
        "search_segment_candidates",
        lambda db, query_embedding, limit: [
            {
                "segment_id": 11,
                "video_id": 2,
                "segment_index": 1,
                "start_timestamp_sec": 3.0,
                "end_timestamp_sec": 4.0,
                "keyframe_id": 5,
                "caption_text": "red car",
                "ocr_text": "",
                "labels": ["car"],
                "object_counts": {"car": 1},
                "object_positions": {"car": ["center"]},
                "semantic_entities": [],
                "semantic_counts": {},
                "vector_distance": 0.42,
            }
        ],
    )
    monkeypatch.setattr(
        video_query_search,
        "fetch_frame_media_map",
        lambda db, frame_ids: {5: {"image_path": "a.png", "thumb_path": "a.webp"}},
    )

    results = _collect_local_candidates(db=object(), query_frame_paths=[frame_path])

    assert len(results) == 1
    assert results[0]["frame_id"] == 5
