from app.services.local_rerank import score_local_candidate


def test_score_local_candidate_combines_branch_signals() -> None:
    score = score_local_candidate(
        {
            "dense_score": 0.5,
            "text_score": 0.2,
            "ocr_score": 0.1,
            "object_score": 0.1,
            "entity_score": 0.05,
            "temporal_score": 0.05,
        }
    )

    assert round(score, 2) == 0.24
