from worker import pipeline


def test_optional_mean_vector_returns_none_for_empty_vectors() -> None:
    assert pipeline._optional_mean_vector([[], []]) is None


def test_should_run_vlm_enrichment_in_full_profile() -> None:
    assert pipeline._should_run_vlm_enrichment(
        segment_index=2,
        segment_count=5,
        profile="full",
        sparse_stride=3,
        ocr_text="street sign",
        objects=[{"label": "car"}],
    )


def test_should_run_vlm_enrichment_in_local_profile_uses_sparse_schedule() -> None:
    assert not pipeline._should_run_vlm_enrichment(
        segment_index=2,
        segment_count=10,
        profile="local",
        sparse_stride=6,
        ocr_text="street sign",
        objects=[{"label": "car"}],
    )


def test_should_run_vlm_enrichment_in_local_profile_when_supporting_evidence_is_missing() -> None:
    assert pipeline._should_run_vlm_enrichment(
        segment_index=2,
        segment_count=5,
        profile="local",
        sparse_stride=3,
        ocr_text="",
        objects=[],
    )


def test_should_run_vlm_enrichment_in_local_profile_for_first_and_last_segments() -> None:
    assert pipeline._should_run_vlm_enrichment(
        segment_index=1,
        segment_count=5,
        profile="local",
        sparse_stride=3,
        ocr_text="street sign",
        objects=[{"label": "car"}],
    )
    assert pipeline._should_run_vlm_enrichment(
        segment_index=5,
        segment_count=5,
        profile="local",
        sparse_stride=3,
        ocr_text="street sign",
        objects=[{"label": "car"}],
    )


def test_should_skip_vlm_enrichment_in_local_profile_for_dense_middle_segments() -> None:
    assert not pipeline._should_run_vlm_enrichment(
        segment_index=2,
        segment_count=5,
        profile="local",
        sparse_stride=3,
        ocr_text="street sign",
        objects=[{"label": "car"}],
    )


def test_build_lightweight_caption_prefers_objects_and_ocr() -> None:
    caption = pipeline._build_lightweight_caption(
        [
            {"label": "car"},
            {"label": "person"},
            {"label": "car"},
        ],
        "repair shop sign",
    )

    assert caption == "car person repair shop sign"


def test_should_commit_progress_respects_interval_and_final_item() -> None:
    assert not pipeline._should_commit_progress(index=1, total=20, interval=12)
    assert pipeline._should_commit_progress(index=12, total=20, interval=12)
    assert pipeline._should_commit_progress(index=20, total=20, interval=12)


def test_record_stage_timing_accumulates_elapsed_time() -> None:
    timings: dict[str, dict[str, float | int]] = {}

    pipeline._record_stage_timing(timings, "ocr", 0.25)
    pipeline._record_stage_timing(timings, "ocr", 0.5)

    assert timings["ocr"]["count"] == 2
    assert timings["ocr"]["total_sec"] == 0.75


def test_prepare_enrich_inputs_returns_segment_image_payloads(tmp_path) -> None:
    class _Frame:
        def __init__(self, image_path: str) -> None:
            self.image_path = image_path

    class _Db:
        def __init__(self, frame) -> None:
            self._frame = frame

        def get(self, model, keyframe_id):
            return self._frame

    class _Segment:
        def __init__(self) -> None:
            self.keyframe_id = 7

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"sample")

    payloads = pipeline._prepare_enrich_inputs(_Db(_Frame(str(image_path))), [_Segment()])

    assert payloads[0]["image_path"] == str(image_path)


def test_prepare_enrich_inputs_preserves_segment_order(tmp_path) -> None:
    class _Frame:
        def __init__(self, image_path: str) -> None:
            self.image_path = image_path

    class _Db:
        def get(self, model, keyframe_id):
            return _Frame(str(tmp_path / f"{keyframe_id}.png"))

    class _Segment:
        def __init__(self, keyframe_id: int) -> None:
            self.keyframe_id = keyframe_id

    first = _Segment(3)
    second = _Segment(9)

    payloads = pipeline._prepare_enrich_inputs(_Db(), [first, second])

    assert [payload["segment"] for payload in payloads] == [first, second]
