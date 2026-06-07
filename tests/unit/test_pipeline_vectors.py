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


def test_should_run_vlm_enrichment_in_balanced_profile_for_sparse_segments() -> None:
    assert pipeline._should_run_vlm_enrichment(
        segment_index=2,
        segment_count=5,
        profile="balanced",
        sparse_stride=3,
        ocr_text="",
        objects=[],
    )


def test_should_skip_vlm_enrichment_in_balanced_profile_for_dense_middle_segments() -> None:
    assert not pipeline._should_run_vlm_enrichment(
        segment_index=2,
        segment_count=5,
        profile="balanced",
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
