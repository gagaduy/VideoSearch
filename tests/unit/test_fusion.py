from app.services.fusion import fuse_scores


def test_fuse_scores_combines_modalities() -> None:
    score = fuse_scores(
        embedding_score=0.5,
        caption_score=0.2,
        ocr_score=0.1,
        object_score=0.1,
        temporal_score=0.1,
    )

    assert round(score, 2) == 1.0

