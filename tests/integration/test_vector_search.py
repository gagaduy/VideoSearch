from sqlalchemy.orm import Session

from app.db.models import Frame, Segment, Video
from app.db.session import get_session_factory
from app.services.query_understanding import ObjectFilter, StructuredQuery
from app.services.search_service import run_search


def test_run_search_prefers_nearest_segment_for_query(monkeypatch) -> None:
    session: Session = get_session_factory()()
    try:
        class _OpenClip:
            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"values": [0.2, 0.2, 0.1, 0.3, 0.2, 0.0, 0.0, 0.0]})()

        monkeypatch.setattr("app.services.search_service.OpenClipAdapter", _OpenClip)

        video = Video(filename="vector.mp4", source_path="./data/videos/vector.mp4", status="indexed")
        session.add(video)
        session.flush()

        near_frame = Frame(
            video_id=video.id,
            timestamp_sec=5.0,
            frame_index=1,
            image_path="data/frames/near.jpg",
            thumb_path="data/thumbs/near.webp",
            is_keyframe=True,
        )
        far_frame = Frame(
            video_id=video.id,
            timestamp_sec=15.0,
            frame_index=2,
            image_path="data/frames/far.jpg",
            thumb_path="data/thumbs/far.webp",
            is_keyframe=True,
        )
        session.add_all([near_frame, far_frame])
        session.flush()

        session.add_all(
            [
                Segment(
                    video_id=video.id,
                    segment_index=1,
                    start_timestamp_sec=5.0,
                    end_timestamp_sec=9.0,
                    keyframe_id=near_frame.id,
                    caption_text="a transparent deep sea fish",
                    ocr_text="",
                    object_labels_json=["vector-test"],
                    embedding=[0.22596000134944916, 0.23519599437713623, 0.07932399958372116, 0.33491700887680054, 0.21256199479103088, 0.0, 0.0, 0.0],
                    raw_json=None,
                ),
                Segment(
                    video_id=video.id,
                    segment_index=2,
                    start_timestamp_sec=15.0,
                    end_timestamp_sec=19.0,
                    keyframe_id=far_frame.id,
                    caption_text="an iceberg near a boat",
                    ocr_text="",
                    object_labels_json=["vector-test"],
                    embedding=[0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0],
                    raw_json=None,
                ),
            ]
        )
        session.commit()

        results = run_search(session, query="deep sea fish", object_labels=["vector-test"])
        own_results = [item for item in results["results"] if item["frame_id"] in {near_frame.id, far_frame.id}]

        assert [item["frame_id"] for item in own_results[:2]] == [near_frame.id, far_frame.id]
        assert own_results[0]["score"] > own_results[1]["score"]
    finally:
        session.close()


def test_run_search_applies_object_count_filters_from_structured_query(monkeypatch) -> None:
    session: Session = get_session_factory()()
    try:
        class _OpenClip:
            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"values": [0.2] * 8})()

        monkeypatch.setattr("app.services.search_service.OpenClipAdapter", _OpenClip)
        monkeypatch.setattr(
            "app.services.search_service.parse_structured_query",
            lambda query, api_key, model: StructuredQuery(
                original_query=query,
                semantic_query="fish together",
                semantic_queries=["fish together"],
                object_filters=[ObjectFilter(label="fish", min_count=2)],
                must_terms=[],
                soft_terms=[],
                temporal_steps=[],
            ),
        )

        video = Video(filename="count.mp4", source_path="./data/videos/count.mp4", status="indexed")
        session.add(video)
        session.flush()

        one_frame = Frame(
            video_id=video.id,
            timestamp_sec=5.0,
            frame_index=1,
            image_path="data/frames/one.jpg",
            thumb_path="data/thumbs/one.webp",
            is_keyframe=True,
        )
        two_frame = Frame(
            video_id=video.id,
            timestamp_sec=15.0,
            frame_index=2,
            image_path="data/frames/two.jpg",
            thumb_path="data/thumbs/two.webp",
            is_keyframe=True,
        )
        session.add_all([one_frame, two_frame])
        session.flush()

        session.add_all(
            [
                Segment(
                    video_id=video.id,
                    segment_index=1,
                    start_timestamp_sec=5.0,
                    end_timestamp_sec=9.0,
                    keyframe_id=one_frame.id,
                    caption_text="one fish swimming",
                    ocr_text="",
                    object_labels_json=["fish"],
                    object_counts_json={"fish": 1},
                    embedding=[0.2] * 8,
                    raw_json=None,
                ),
                Segment(
                    video_id=video.id,
                    segment_index=2,
                    start_timestamp_sec=15.0,
                    end_timestamp_sec=19.0,
                    keyframe_id=two_frame.id,
                    caption_text="two fish swimming together",
                    ocr_text="",
                    object_labels_json=["fish"],
                    object_counts_json={"fish": 2},
                    embedding=[0.21] * 8,
                    raw_json=None,
                ),
            ]
        )
        session.commit()

        results = run_search(session, query="two fish swimming together", object_labels=[])

        own_results = [item for item in results["results"] if item["frame_id"] in {one_frame.id, two_frame.id}]
        assert [item["frame_id"] for item in own_results] == [two_frame.id]
        assert own_results[0]["object_counts"] == {"fish": 2}
    finally:
        session.close()


def test_run_search_prefers_semantic_counts_over_detector_noise(monkeypatch) -> None:
    session: Session = get_session_factory()()
    try:
        class _OpenClip:
            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"values": [0.2] * 8})()

        monkeypatch.setattr("app.services.search_service.OpenClipAdapter", _OpenClip)
        monkeypatch.setattr(
            "app.services.search_service.parse_structured_query",
            lambda query, api_key, model: StructuredQuery(
                original_query=query,
                semantic_query="two fish swimming together",
                semantic_queries=["two fish swimming together"],
                object_filters=[ObjectFilter(label="fish", min_count=2)],
                must_terms=[],
                soft_terms=[],
                temporal_steps=[],
            ),
        )

        video = Video(filename="semantic-count.mp4", source_path="./data/videos/semantic-count.mp4", status="indexed")
        session.add(video)
        session.flush()

        noisy_frame = Frame(
            video_id=video.id,
            timestamp_sec=5.0,
            frame_index=1,
            image_path="data/frames/noisy.jpg",
            thumb_path="data/thumbs/noisy.webp",
            is_keyframe=True,
        )
        clean_frame = Frame(
            video_id=video.id,
            timestamp_sec=15.0,
            frame_index=2,
            image_path="data/frames/clean.jpg",
            thumb_path="data/thumbs/clean.webp",
            is_keyframe=True,
        )
        session.add_all([noisy_frame, clean_frame])
        session.flush()

        session.add_all(
            [
                Segment(
                    video_id=video.id,
                    segment_index=1,
                    start_timestamp_sec=5.0,
                    end_timestamp_sec=9.0,
                    keyframe_id=noisy_frame.id,
                    caption_text="two fish swimming together in dark water",
                    ocr_text="",
                    object_labels_json=["broccoli"],
                    object_counts_json={"broccoli": 1},
                    semantic_entities_json=[
                        {"label": "eel", "count": 2, "aliases": ["eel", "fish", "sea creature"], "regions": []}
                    ],
                    semantic_counts_json={"fish": 2, "eel": 2},
                    embedding=[0.2] * 8,
                    raw_json=None,
                ),
                Segment(
                    video_id=video.id,
                    segment_index=2,
                    start_timestamp_sec=15.0,
                    end_timestamp_sec=19.0,
                    keyframe_id=clean_frame.id,
                    caption_text="one fish swimming alone",
                    ocr_text="",
                    object_labels_json=["fish"],
                    object_counts_json={"fish": 1},
                    semantic_entities_json=[
                        {"label": "fish", "count": 1, "aliases": ["fish"], "regions": []}
                    ],
                    semantic_counts_json={"fish": 1},
                    embedding=[0.21] * 8,
                    raw_json=None,
                ),
            ]
        )
        session.commit()

        results = run_search(session, query="two fish swimming together", object_labels=[])

        own_results = [item for item in results["results"] if item["frame_id"] in {noisy_frame.id, clean_frame.id}]
        assert [item["frame_id"] for item in own_results] == [noisy_frame.id]
        assert own_results[0]["object_counts"] == {"fish": 2, "eel": 2}
    finally:
        session.close()
