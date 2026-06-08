from pathlib import Path

from app.services import search_service


def test_compute_result_component_scores_uses_ocr_text_for_ocr_score() -> None:
    row = {
        "caption_text": "man beside car",
        "ocr_text": "speed zone sign",
        "semantic_aliases": {},
        "semantic_counts": {},
        "semantic_entities": [],
        "labels": [],
        "object_counts": {},
        "object_positions": {},
    }

    scores = search_service._compute_result_component_scores(
        row=row,
        text_terms=["speed", "zone"],
        dense_score=0.2,
        text_score=0.1,
        object_score=0.0,
        temporal_score=0.0,
    )

    assert scores["ocr_score"] == 1.0


def test_filter_results_applies_threshold_before_cap() -> None:
    rows = [
        {"score": 0.31, "segment_id": 1},
        {"score": 0.29, "segment_id": 2},
        {"score": 0.17, "segment_id": 3},
    ]

    filtered = search_service._filter_display_results(rows, threshold=0.18, limit=2)

    assert [row["segment_id"] for row in filtered] == [1, 2]


def test_filter_results_returns_empty_when_no_row_meets_threshold() -> None:
    rows = [
        {"score": 0.12, "segment_id": 1},
        {"score": 0.10, "segment_id": 2},
    ]

    assert search_service._filter_display_results(rows, threshold=0.18, limit=16) == []


def test_run_image_search_returns_mode_and_filtered_results(monkeypatch) -> None:
    image_path = Path("query.png")

    monkeypatch.setattr(search_service, "_embed_query_image", lambda path: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        search_service,
        "search_segment_candidates",
        lambda db, query_embedding, limit=80: [
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
            },
            {
                "segment_id": 12,
                "video_id": 2,
                "segment_index": 2,
                "start_timestamp_sec": 4.0,
                "end_timestamp_sec": 5.0,
                "keyframe_id": 6,
                "caption_text": "weak",
                "ocr_text": "",
                "labels": ["car"],
                "object_counts": {"car": 1},
                "object_positions": {"car": ["center"]},
                "semantic_entities": [],
                "semantic_counts": {},
                "vector_distance": 9.0,
            },
        ],
    )
    monkeypatch.setattr(
        search_service,
        "fetch_frame_media_map",
        lambda db, frame_ids: {
            5: {"image_path": "a.png", "thumb_path": "a.webp"},
            6: {"image_path": "b.png", "thumb_path": "b.webp"},
        },
    )

    payload = search_service.run_image_search(object(), image_path)

    assert payload["mode"] == "image"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["segment_id"] == 11


def test_apply_openai_vision_rerank_updates_only_top_8(monkeypatch) -> None:
    rows = [{"frame_id": index + 1, "score": 0.8 - (index * 0.01)} for index in range(10)]
    monkeypatch.setattr(search_service.settings, "openai_vision_rerank_enabled", True)
    monkeypatch.setattr(search_service.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(search_service.settings, "openai_vision_rerank_top_k", 8)
    monkeypatch.setattr(
        search_service,
        "run_openai_vision_rerank",
        lambda query, candidates: {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8},
    )

    reranked = search_service._apply_openai_vision_rerank("query", rows)

    assert len(reranked) == 10
    assert reranked[0]["frame_id"] == 8
    assert reranked[-1]["frame_id"] == 1
    assert [row["frame_id"] for row in reranked if row["frame_id"] in {9, 10}] == [9, 10]


def test_apply_openai_vision_rerank_returns_local_results_when_disabled(monkeypatch) -> None:
    rows = [{"frame_id": 1, "score": 0.6}]
    monkeypatch.setattr(search_service.settings, "openai_vision_rerank_enabled", False)

    assert search_service._apply_openai_vision_rerank("query", rows) == rows
