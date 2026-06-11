from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Frame, FrameObject, Segment, Video
from app.db.session import get_session_factory
from worker import pipeline


def test_run_index_pipeline_persists_frames_from_extracted_paths(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    original_detector_family = pipeline.settings.object_detector_family
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")
        monkeypatch.setattr(pipeline.settings, "object_detector_family", "yolo_world")

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

            def embed_images(self, image_paths: list[str]) -> list[object]:
                return [self.embed_image(image_path) for image_path in image_paths]

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
        monkeypatch.setattr(pipeline, "build_object_detector", lambda: _Detector())
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
        assert segment.raw_json["object_detector_model"] == "stub-yolo-world"
        assert "" in segment.raw_json["object_prompt_set"]
    finally:
        pipeline.settings.object_detector_family = original_detector_family
        session.close()


def test_run_index_pipeline_records_codetr_detector_family(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    original_detector_family = pipeline.settings.object_detector_family
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")
        monkeypatch.setattr(pipeline.settings, "object_detector_family", "codetr")

        video = Video(filename="codetr.mp4", source_path=str(tmp_path / "codetr.mp4"), status="pending")
        session.add(video)
        session.commit()

        frame_path = tmp_path / "frame_000001.png"
        frame_path.write_bytes(b"frame-one")

        monkeypatch.setattr(pipeline, "_prepare_frame_paths", lambda video_id, source_path: [frame_path])
        monkeypatch.setattr(pipeline, "keep_distinct_frames", lambda frames, distance_threshold: frames)
        monkeypatch.setattr(
            pipeline,
            "copy_or_create_thumbnail",
            lambda image_path, thumb_path: thumb_path.parent.mkdir(parents=True, exist_ok=True) or thumb_path.write_bytes(b"thumb") or thumb_path,
        )

        class _OpenClip:
            def embed_image(self, image_path: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.1, 0.2, 0.3]})()

            def embed_images(self, image_paths: list[str]) -> list[object]:
                return [self.embed_image(image_path) for image_path in image_paths]

            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

        class _Caption:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def caption(self, image_path: str) -> dict[str, object]:
                return {"caption": "", "model_name": "stub"}

        class _Ocr:
            engine_name = "stub-ocr"

            def extract_text(self, image_path: str) -> dict[str, object]:
                return {"text": "codetr text", "tokens": ["codetr", "text"], "raw": []}

        class _Detector:
            model_name = "co_detr_stub"

            def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
                return [{"label": "car", "matched_prompt": "car", "score": 0.9, "bbox": [0, 0, 1, 1]}]

        class _InternVL:
            def describe_image(self, image_path: str) -> dict[str, object]:
                return {
                    "caption": "car on track",
                    "tags": ["car"],
                    "entities": [{"label": "car", "aliases": ["vehicle"]}],
                    "model_name": "stub-internvl",
                }

        monkeypatch.setattr(pipeline, "OpenClipAdapter", _OpenClip)
        monkeypatch.setattr(pipeline, "CaptionAdapter", _Caption)
        monkeypatch.setattr(pipeline, "PaddleOcrAdapter", _Ocr)
        monkeypatch.setattr(pipeline, "build_object_detector", lambda: _Detector())
        monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)

        payload = pipeline.run_index_pipeline(session, int(video.id))

        segment = session.query(Segment).filter(Segment.video_id == video.id).one()
        assert segment.raw_json["object_detector_family"] == "codetr"
        assert segment.raw_json["object_detector_model"] == "co_detr_stub"
        assert "detector" in payload["stage_timings"]
    finally:
        pipeline.settings.object_detector_family = original_detector_family
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

            def embed_images(self, image_paths: list[str]) -> list[object]:
                return [self.embed_image(image_path) for image_path in image_paths]

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
        monkeypatch.setattr(pipeline, "build_object_detector", lambda: _Detector())
        monkeypatch.setattr(pipeline, "SemanticEntityAdapter", _Semantic)
        monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)

        pipeline.run_index_pipeline(session, int(video.id))

        segment = session.query(Segment).filter(Segment.video_id == video.id).one()
        assert calls["internvl"] == 1
        assert segment.caption_text == "person near sign"
        assert segment.semantic_counts_json["person"] == 1
    finally:
        session.close()


def test_run_index_pipeline_uses_sparse_vlm_enrichment_in_local_profile(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    original_profile = pipeline.settings.indexing_profile
    original_stride = pipeline.settings.internvl_sparse_stride
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")
        monkeypatch.setattr(pipeline.settings, "indexing_profile", "local")
        monkeypatch.setattr(pipeline.settings, "internvl_sparse_stride", 3)

        video = Video(filename="local.mp4", source_path=str(tmp_path / "local.mp4"), status="pending")
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

            def embed_images(self, image_paths: list[str]) -> list[object]:
                return [self.embed_image(image_path) for image_path in image_paths]

            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

        class _Caption:
            def caption(self, image_path: str) -> dict[str, object]:
                raise AssertionError("caption adapter should stay unused in local mode when InternVL is skipped")

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
                raise AssertionError("semantic entity adapter should stay unused in local mode when InternVL is skipped")

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
        monkeypatch.setattr(pipeline, "build_object_detector", lambda: _Detector())
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


def test_run_index_pipeline_leaves_semantic_fields_empty_when_vlm_selected_but_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session: Session = get_session_factory()()
    original_profile = pipeline.settings.indexing_profile
    original_stride = pipeline.settings.internvl_sparse_stride
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")
        monkeypatch.setattr(pipeline.settings, "indexing_profile", "local")
        monkeypatch.setattr(pipeline.settings, "internvl_sparse_stride", 6)

        video = Video(filename="vlm-fail.mp4", source_path=str(tmp_path / "vlm-fail.mp4"), status="pending")
        session.add(video)
        session.commit()

        frame_path = tmp_path / "frame_000001.png"
        frame_path.write_bytes(b"frame-1")

        monkeypatch.setattr(pipeline, "_prepare_frame_paths", lambda video_id, source_path: [frame_path])
        monkeypatch.setattr(pipeline, "keep_distinct_frames", lambda frames, distance_threshold: frames)
        monkeypatch.setattr(
            pipeline,
            "copy_or_create_thumbnail",
            lambda image_path, thumb_path: thumb_path.parent.mkdir(parents=True, exist_ok=True) or thumb_path.write_bytes(b"thumb") or thumb_path,
        )

        calls = {"caption": 0, "semantic": 0}

        class _OpenClip:
            def embed_image(self, image_path: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [1.0, 0.0, 0.0]})()

            def embed_images(self, image_paths: list[str]) -> list[object]:
                return [self.embed_image(image_path) for image_path in image_paths]

            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

        class _Caption:
            def caption(self, image_path: str) -> dict[str, object]:
                calls["caption"] += 1
                return {"caption": "fallback caption", "model_name": "stub-caption", "confidence": 0.5}

        class _Ocr:
            engine_name = "stub-ocr"

            def extract_text(self, image_path: str) -> dict[str, object]:
                return {"text": "text frame_000001", "tokens": ["text", "frame_000001"], "raw": []}

        class _Detector:
            model_name = "stub-yolo-world"

            def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
                return [{"label": "object_1", "matched_prompt": "object", "score": 0.9, "bbox": [0, 0, 1, 1]}]

        class _Semantic:
            def extract(self, image_path: str, caption_text: str, ocr_text: str) -> dict[str, object]:
                calls["semantic"] += 1
                return {"entities": [{"label": "fallback"}], "counts": {"fallback": 1}}

        class _InternVL:
            def describe_image(self, image_path: str) -> dict[str, object]:
                raise RuntimeError("stub internvl failure")

        monkeypatch.setattr(pipeline, "OpenClipAdapter", _OpenClip)
        monkeypatch.setattr(pipeline, "CaptionAdapter", _Caption)
        monkeypatch.setattr(pipeline, "PaddleOcrAdapter", _Ocr)
        monkeypatch.setattr(pipeline, "build_object_detector", lambda: _Detector())
        monkeypatch.setattr(pipeline, "SemanticEntityAdapter", _Semantic)
        monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)

        pipeline.run_index_pipeline(session, int(video.id))

        segment = session.query(Segment).filter(Segment.video_id == video.id).one()
        assert calls == {"caption": 0, "semantic": 0}
        assert segment.caption_text == ""
        assert segment.semantic_entities_json == []
        assert segment.semantic_counts_json == {}
        assert segment.ocr_text == "text frame_000001"
        assert segment.object_counts_json["object_1"] == 1
        assert "branch_b" in segment.stage_failures_json
        assert "caption" not in segment.stage_failures_json
        assert "semantic_entities" not in segment.stage_failures_json
    finally:
        pipeline.settings.indexing_profile = original_profile
        pipeline.settings.internvl_sparse_stride = original_stride
        session.close()


def test_index_prepared_frames_batches_openclip_image_embedding(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    original_batch_size = pipeline.settings.openclip_batch_size
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")
        monkeypatch.setattr(pipeline.settings, "openclip_batch_size", 2)

        video = Video(filename="batch.mp4", source_path=str(tmp_path / "batch.mp4"), status="pending")
        session.add(video)
        session.commit()

        frame_sources = []
        for index in range(1, 5):
            frame_path = tmp_path / f"frame_{index:06d}.png"
            frame_path.write_bytes(f"frame-{index}".encode("utf-8"))
            frame_sources.append({"image_path": str(frame_path), "timestamp_sec": float(index - 1), "frame_index": index})

        monkeypatch.setattr(
            pipeline,
            "copy_or_create_thumbnail",
            lambda image_path, thumb_path: thumb_path.parent.mkdir(parents=True, exist_ok=True) or thumb_path.write_bytes(b"thumb") or thumb_path,
        )

        calls = {"embed_images": 0}

        class _OpenClip:
            def embed_images(self, image_paths: list[str]) -> list[object]:
                calls["embed_images"] += 1
                return [
                    type("Embedding", (), {"model_name": "stub", "values": [float(index), 0.0, 0.0]})()
                    for index, _ in enumerate(image_paths, start=1)
                ]

            def embed_image(self, image_path: str) -> object:
                raise AssertionError("pipeline should batch OpenCLIP image embedding")

            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

        class _Caption:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def caption(self, image_path: str) -> dict[str, object]:
                return {"caption": "", "model_name": "stub"}

        class _Ocr:
            engine_name = "stub-ocr"

            def extract_text(self, image_path: str) -> dict[str, object]:
                return {"text": "", "tokens": [], "raw": []}

        class _Detector:
            model_name = "stub-yolo-world"

            def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
                return []

        class _Semantic:
            def extract(self, image_path: str, caption_text: str, ocr_text: str) -> dict[str, object]:
                return {"entities": [], "counts": {}}

        class _InternVL:
            def describe_image(self, image_path: str) -> dict[str, object]:
                return {"caption": "", "tags": [], "entities": [], "model_name": "stub-internvl"}

        payload = pipeline.index_prepared_frames(
            session,
            int(video.id),
            frame_sources,
            openclip=_OpenClip(),
            captioner=_Caption(),
            ocr_engine=_Ocr(),
            detector=_Detector(),
            entity_extractor=_Semantic(),
            branch_b_adapter=_InternVL(),
        )

        assert payload["frame_count"] == 4
        assert calls["embed_images"] == 2
    finally:
        pipeline.settings.openclip_batch_size = original_batch_size
        session.close()


def test_run_index_pipeline_returns_stage_timings(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    try:
        monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(pipeline.settings, "frames_dir", tmp_path / "frames")
        monkeypatch.setattr(pipeline.settings, "thumbs_dir", tmp_path / "thumbs")
        monkeypatch.setattr(pipeline.settings, "enable_stage_timing", True)

        video = Video(filename="timed.mp4", source_path=str(tmp_path / "timed.mp4"), status="pending")
        session.add(video)
        session.commit()

        frame_path = tmp_path / "frame_000001.png"
        frame_path.write_bytes(b"frame-one")

        monkeypatch.setattr(pipeline, "_prepare_frame_paths", lambda video_id, source_path: [frame_path])
        monkeypatch.setattr(pipeline, "keep_distinct_frames", lambda frames, distance_threshold: frames)
        monkeypatch.setattr(
            pipeline,
            "copy_or_create_thumbnail",
            lambda image_path, thumb_path: thumb_path.parent.mkdir(parents=True, exist_ok=True) or thumb_path.write_bytes(b"thumb") or thumb_path,
        )

        class _OpenClip:
            def embed_image(self, image_path: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.1, 0.2, 0.3]})()

            def embed_images(self, image_paths: list[str]) -> list[object]:
                return [self.embed_image(image_path) for image_path in image_paths]

            def embed_text(self, text: str) -> object:
                return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

        class _Caption:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def caption(self, image_path: str) -> dict[str, object]:
                return {"caption": "", "model_name": "stub"}

        class _Ocr:
            engine_name = "stub-ocr"

            def extract_text(self, image_path: str) -> dict[str, object]:
                return {"text": "timed text", "tokens": ["timed", "text"], "raw": []}

        class _Detector:
            model_name = "stub-yolo-world"

            def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
                return [{"label": "car", "matched_prompt": "car", "score": 0.9, "bbox": [0, 0, 1, 1]}]

        class _InternVL:
            def describe_image(self, image_path: str) -> dict[str, object]:
                return {
                    "caption": "car in daylight",
                    "tags": ["car"],
                    "entities": [{"label": "car", "aliases": ["vehicle"]}],
                    "model_name": "stub-internvl",
                }

        monkeypatch.setattr(pipeline, "OpenClipAdapter", _OpenClip)
        monkeypatch.setattr(pipeline, "CaptionAdapter", _Caption)
        monkeypatch.setattr(pipeline, "PaddleOcrAdapter", _Ocr)
        monkeypatch.setattr(pipeline, "build_object_detector", lambda: _Detector())
        monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)

        payload = pipeline.run_index_pipeline(session, int(video.id))

        segment = session.query(Segment).filter(Segment.video_id == video.id).one()
        assert "stage_timings" in payload
        assert "frame_embedding" in payload["stage_timings"]
        assert "ocr" in payload["stage_timings"]
        assert "detector" in payload["stage_timings"]
        assert "branch_b" in payload["stage_timings"]
        assert "stage_timings" in segment.raw_json
    finally:
        session.close()


def test_index_prepared_frames_releases_gpu_heavy_adapters_between_phases(monkeypatch, tmp_path: Path) -> None:
    session: Session = get_session_factory()()
    try:
        video = Video(filename="phased.mp4", source_path=str(tmp_path / "phased.mp4"), status="pending")
        session.add(video)
        session.commit()

        frame_path = tmp_path / "frame_000001.png"
        frame_path.write_bytes(b"frame-one")

        actions: list[str] = []

        class _OpenClip:
            def __init__(self) -> None:
                self.closed = False

            def embed_images(self, image_paths: list[str]) -> list[object]:
                actions.append("openclip.embed_images")
                return [type("Embedding", (), {"model_name": "stub", "values": [0.1, 0.2, 0.3]})() for _ in image_paths]

            def embed_text(self, text: str) -> object:
                assert internvl.closed, "branch_b text embedding should run only after InternVL is released"
                actions.append("openclip.embed_text")
                return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

            def close(self) -> None:
                self.closed = True
                actions.append("openclip.close")

        class _Caption:
            def caption(self, image_path: str) -> dict[str, object]:
                actions.append("caption.caption")
                return {"caption": "", "model_name": "stub"}

            def close(self) -> None:
                actions.append("caption.close")

        class _Ocr:
            engine_name = "stub-ocr"

            def extract_text(self, image_path: str) -> dict[str, object]:
                actions.append("ocr.extract")
                return {"text": "timed text", "tokens": ["timed", "text"], "raw": []}

        class _Detector:
            model_name = "stub-codetr"

            def __init__(self) -> None:
                self.closed = False

            def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
                assert openclip.closed, "image embedding adapter should be released before detector phase"
                actions.append("detector.detect")
                return [{"label": "car", "matched_prompt": "car", "score": 0.9, "bbox": [0, 0, 1, 1]}]

            def close(self) -> None:
                self.closed = True
                actions.append("detector.close")

        class _Semantic:
            def extract(self, image_path: str, caption_text: str, ocr_text: str) -> dict[str, object]:
                actions.append("semantic.extract")
                return {"entities": [], "counts": {}}

        class _InternVL:
            def __init__(self) -> None:
                self.closed = False

            def describe_image(self, image_path: str) -> dict[str, object]:
                assert detector.closed, "detector should be released before VLM phase"
                actions.append("internvl.describe")
                return {
                    "caption": "a black sports car",
                    "tags": ["car"],
                    "entities": [{"label": "car", "aliases": ["vehicle"]}],
                    "model_name": "stub-internvl",
                }

            def close(self) -> None:
                self.closed = True
                actions.append("internvl.close")

        openclip = _OpenClip()
        detector = _Detector()
        internvl = _InternVL()

        payload = pipeline.index_prepared_frames(
            session,
            int(video.id),
            [{"image_path": str(frame_path), "timestamp_sec": 0.0, "frame_index": 1}],
            openclip=openclip,
            captioner=_Caption(),
            ocr_engine=_Ocr(),
            detector=detector,
            entity_extractor=_Semantic(),
            branch_b_adapter=internvl,
        )

        assert payload["segment_count"] == 1
        assert actions.index("openclip.close") < actions.index("detector.detect")
        assert actions.index("detector.close") < actions.index("internvl.describe")
        assert actions.index("internvl.close") < actions.index("openclip.embed_text")
    finally:
        session.close()
