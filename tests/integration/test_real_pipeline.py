from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Frame, FrameObject, Segment, Video
from app.db.session import get_session_factory
from worker import pipeline


def test_run_index_pipeline_persists_frames_from_extracted_paths(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")

        video = Video(filename="real.mp4", source_path=str(tmp_path / "real.mp4"), status="pending")
        session.add(video)
        session.commit()

        frame_one = tmp_path / "frame_000001.png"
        frame_two = tmp_path / "frame_000002.png"
        frame_one.write_bytes(b"frame-one")
        frame_two.write_bytes(b"frame-two")

        monkeypatch.setattr(pipeline, "_prepare_frame_paths", lambda video_id, source_path: [frame_one, frame_two])
        monkeypatch.setattr(pipeline, "keep_distinct_frames", lambda frames, distance_threshold: frames)
        monkeypatch.setattr(
            pipeline,
            "copy_or_create_thumbnail",
            lambda image_path, thumb_path: thumb_path.parent.mkdir(parents=True, exist_ok=True) or thumb_path.write_bytes(b"thumb") or thumb_path,
        )

        class _OpenClip:
            def embed_image(self, image_path: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.1, 0.2, 0.3]})()

            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

        class _Caption:
            def caption(self, image_path: str) -> dict[str, object]:
                return {"caption": f"caption {Path(image_path).name}", "model_name": "stub"}

        class _Ocr:
            engine_name = "stub-ocr"

            def extract_text(self, image_path: str) -> dict[str, object]:
                return {"text": f"text {Path(image_path).stem}", "tokens": ["text", Path(image_path).stem], "raw": []}

        class _Detector:
            model_name = "stub-yolo"

            def detect(self, image_path: str) -> list[dict[str, object]]:
                return [{"label": "person", "score": 0.9, "bbox": [0, 0, 1, 1]}]

        class _InternVL:
            def describe_image(self, image_path: str) -> dict[str, object]:
                return {
                    "caption": f"semantic {Path(image_path).stem}",
                    "tags": ["person", "scene"],
                    "entities": [{"label": "person", "aliases": ["human"]}],
                    "model_name": "stub-internvl",
                }

        monkeypatch.setattr(pipeline, "OpenClipAdapter", _OpenClip)
        monkeypatch.setattr(pipeline, "CaptionAdapter", _Caption)
        monkeypatch.setattr(pipeline, "PaddleOcrAdapter", _Ocr)
        monkeypatch.setattr(pipeline, "YoloDetectionAdapter", _Detector)
        monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)

        payload = pipeline.run_index_pipeline(session, int(video.id))

        frame_count = session.query(Frame).filter(Frame.video_id == video.id).count()
        object_count = session.query(FrameObject).join(Frame, Frame.id == FrameObject.frame_id).filter(Frame.video_id == video.id).count()
        segment = session.query(Segment).filter(Segment.video_id == video.id).one()
        segment_count = 1

        assert payload["frame_count"] == 2
        assert payload["segment_count"] == 1
        assert frame_count == 2
        assert object_count == 1
        assert segment_count == 1
        assert segment.object_labels_json == ["person"]
        assert segment.object_counts_json == {"person": 1}
        assert segment.embedding_branch_a is not None
        assert len(segment.embedding_branch_a) == 3
        assert segment.embedding_branch_b == [0.4, 0.5, 0.6]
        assert segment.ocr_tokens_json == ["text", "frame_000002"]
        assert segment.stage_failures_json == {}
    finally:
        session.close()
