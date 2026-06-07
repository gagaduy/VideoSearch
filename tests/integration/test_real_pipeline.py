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
            model_name = "stub-yolo-world"

            def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
                return [{"label": "person", "matched_prompt": "person", "score": 0.9, "bbox": [0, 0, 1, 1]}]

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
        assert segment.raw_json["object_detector_family"] == "yolo_world"
        assert "" in segment.raw_json["object_prompt_set"]
    finally:
        session.close()


def test_run_index_pipeline_reuses_single_vlm_pass_for_segment_enrichment(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")

        video = Video(filename="reuse.mp4", source_path=str(tmp_path / "reuse.mp4"), status="pending")
        session.add(video)
        session.commit()

        frame_one = tmp_path / "frame_000001.png"
        frame_one.write_bytes(b"frame-one")

        monkeypatch.setattr(pipeline, "_prepare_frame_paths", lambda video_id, source_path: [frame_one])
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
                raise AssertionError("caption adapter should not run when InternVL already produced a caption")

        class _Ocr:
            engine_name = "stub-ocr"

            def extract_text(self, image_path: str) -> dict[str, object]:
                return {"text": "text frame_000001", "tokens": ["text", "frame_000001"], "raw": []}

        class _Detector:
            model_name = "stub-yolo-world"

            def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
                return [{"label": "person", "matched_prompt": "person", "score": 0.9, "bbox": [0, 0, 1, 1]}]

        class _Semantic:
            def extract(self, image_path: str, caption_text: str, ocr_text: str) -> dict[str, object]:
                raise AssertionError("semantic entity adapter should not run when InternVL already produced entities")

        calls = {"internvl": 0}

        class _InternVL:
            def describe_image(self, image_path: str) -> dict[str, object]:
                calls["internvl"] += 1
                return {
                    "caption": "person near sign",
                    "tags": ["person", "sign"],
                    "entities": [{"label": "person", "aliases": ["human"]}],
                    "model_name": "stub-internvl",
                }

        monkeypatch.setattr(pipeline, "OpenClipAdapter", _OpenClip)
        monkeypatch.setattr(pipeline, "CaptionAdapter", _Caption)
        monkeypatch.setattr(pipeline, "PaddleOcrAdapter", _Ocr)
        monkeypatch.setattr(pipeline, "YoloDetectionAdapter", _Detector)
        monkeypatch.setattr(pipeline, "SemanticEntityAdapter", _Semantic)
        monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)

        pipeline.run_index_pipeline(session, int(video.id))

        segment = session.query(Segment).filter(Segment.video_id == video.id).one()
        assert calls["internvl"] == 1
        assert segment.caption_text == "person near sign"
        assert segment.semantic_counts_json["person"] == 1
    finally:
        session.close()


def test_run_index_pipeline_uses_sparse_vlm_enrichment_in_balanced_profile(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    original_profile = pipeline.settings.indexing_profile
    original_stride = pipeline.settings.internvl_sparse_stride
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")
        monkeypatch.setattr(pipeline.settings, "indexing_profile", "balanced")
        monkeypatch.setattr(pipeline.settings, "internvl_sparse_stride", 3)

        video = Video(filename="balanced.mp4", source_path=str(tmp_path / "balanced.mp4"), status="pending")
        session.add(video)
        session.commit()

        frame_paths = []
        for index in range(1, 4):
            frame_path = tmp_path / f"frame_{index:06d}.png"
            frame_path.write_bytes(f"frame-{index}".encode("utf-8"))
            frame_paths.append(frame_path)

        monkeypatch.setattr(pipeline, "_prepare_frame_paths", lambda video_id, source_path: frame_paths)
        monkeypatch.setattr(pipeline, "keep_distinct_frames", lambda frames, distance_threshold: frames)
        monkeypatch.setattr(
            pipeline,
            "copy_or_create_thumbnail",
            lambda image_path, thumb_path: thumb_path.parent.mkdir(parents=True, exist_ok=True) or thumb_path.write_bytes(b"thumb") or thumb_path,
        )

        class _OpenClip:
            def embed_image(self, image_path: str) -> object:
                stem = Path(image_path).stem
                if stem.endswith("1"):
                    values = [1.0, 0.0, 0.0]
                elif stem.endswith("2"):
                    values = [0.0, 1.0, 0.0]
                else:
                    values = [0.0, 0.0, 1.0]
                return type("Embedding", (), {"model_name": "stub", "values": values})()

            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

        class _Caption:
            def caption(self, image_path: str) -> dict[str, object]:
                raise AssertionError("caption adapter should stay unused in balanced mode when InternVL is skipped")

        class _Ocr:
            engine_name = "stub-ocr"

            def extract_text(self, image_path: str) -> dict[str, object]:
                stem = Path(image_path).stem
                return {"text": f"text {stem}", "tokens": ["text", stem], "raw": []}

        class _Detector:
            model_name = "stub-yolo-world"

            def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
                stem = Path(image_path).stem
                return [{"label": f"object_{stem[-1]}", "matched_prompt": "object", "score": 0.9, "bbox": [0, 0, 1, 1]}]

        class _Semantic:
            def extract(self, image_path: str, caption_text: str, ocr_text: str) -> dict[str, object]:
                raise AssertionError("semantic entity adapter should stay unused in balanced mode when InternVL is skipped")

        calls = {"internvl": 0}

        class _InternVL:
            def describe_image(self, image_path: str) -> dict[str, object]:
                calls["internvl"] += 1
                stem = Path(image_path).stem
                return {
                    "caption": f"semantic {stem}",
                    "tags": [f"tag_{stem[-1]}"],
                    "entities": [{"label": f"entity_{stem[-1]}", "aliases": [f"alias_{stem[-1]}"]}],
                    "model_name": "stub-internvl",
                }

        monkeypatch.setattr(pipeline, "OpenClipAdapter", _OpenClip)
        monkeypatch.setattr(pipeline, "CaptionAdapter", _Caption)
        monkeypatch.setattr(pipeline, "PaddleOcrAdapter", _Ocr)
        monkeypatch.setattr(pipeline, "YoloDetectionAdapter", _Detector)
        monkeypatch.setattr(pipeline, "SemanticEntityAdapter", _Semantic)
        monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)

        pipeline.run_index_pipeline(session, int(video.id))

        segments = session.query(Segment).filter(Segment.video_id == video.id).order_by(Segment.segment_index.asc()).all()
        assert len(segments) == 3
        assert calls["internvl"] == 2
        assert segments[0].caption_text == "semantic frame_000001"
        assert segments[1].caption_text == "object_2 text frame_000002"
        assert segments[1].semantic_counts_json["object_2"] == 1
        assert segments[2].caption_text == "semantic frame_000003"
    finally:
        pipeline.settings.indexing_profile = original_profile
        pipeline.settings.internvl_sparse_stride = original_stride
        session.close()
