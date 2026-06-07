import json
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.models import Frame, FrameCaption, FrameEmbedding, FrameObject, FrameOcr, IndexJob, QueryLog, Segment, Video
from app.db.session import get_session_factory
from worker import keyframe_importer


def test_import_keyframe_dataset_indexes_existing_keyframes(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    frame_dir = dataset_root / "keyframe" / "L01" / "L01_V001"
    frame_dir.mkdir(parents=True)
    (frame_dir / "001.jpg").write_bytes(b"frame-1")
    (frame_dir / "002.jpg").write_bytes(b"frame-2")

    metadata_dir = dataset_root / "media-info-b1" / "media-info"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "L01_V001.json").write_text(
        json.dumps(
            {
                "title": "Video 1",
                "author": "Author",
                "length": 12.0,
                "publish_date": "2024-01-01",
            }
        )
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
            return {"text": "dataset text", "tokens": ["dataset", "text"], "raw": []}

    class _Detector:
        model_name = "stub-yolo-world"

        def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
            return [{"label": "person", "matched_prompt": "person", "score": 0.9, "bbox": [0, 0, 1, 1]}]

    class _Semantic:
        def extract(self, image_path: str, caption: str, ocr_text: str) -> dict[str, object]:
            return {"entities": [{"label": "person", "aliases": ["human"]}], "counts": {"person": 1}}

    class _InternVL:
        def describe_image(self, image_path: str) -> dict[str, object]:
            return {
                "caption": "person in scene",
                "tags": ["person"],
                "entities": [{"label": "person", "aliases": ["human"]}],
                "model_name": "stub-internvl",
            }

    monkeypatch.setattr(keyframe_importer, "OpenClipAdapter", _OpenClip)
    monkeypatch.setattr(keyframe_importer, "CaptionAdapter", _Caption)
    monkeypatch.setattr(keyframe_importer, "PaddleOcrAdapter", _Ocr)
    monkeypatch.setattr(keyframe_importer, "YoloDetectionAdapter", _Detector)
    monkeypatch.setattr(keyframe_importer, "SemanticEntityAdapter", _Semantic)
    monkeypatch.setattr(keyframe_importer, "InternvlAdapter", _InternVL)

    session: Session = get_session_factory()()
    try:
        session.execute(delete(QueryLog))
        session.execute(delete(IndexJob))
        session.execute(delete(FrameObject))
        session.execute(delete(FrameOcr))
        session.execute(delete(FrameCaption))
        session.execute(delete(FrameEmbedding))
        session.execute(delete(Frame))
        session.execute(delete(Segment))
        session.execute(delete(Video))
        session.commit()

        result = keyframe_importer.import_keyframe_dataset(session, dataset_root)

        assert result["video_count"] == 1
        assert result["frame_count"] == 2
        assert session.query(Video).count() == 1
        assert session.query(Frame).count() == 2
        assert session.query(FrameObject).count() == 1
        segment = session.query(Segment).one()
        video = session.query(Video).one()

        assert video.filename == "L01_V001.keyframes"
        assert video.source_path == str(frame_dir)
        assert video.duration_sec == 12.0
        assert segment.object_labels_json == ["person"]
        assert segment.raw_json["dataset"] == "keyframe_import"
    finally:
        session.close()
