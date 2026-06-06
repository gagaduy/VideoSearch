from app.services.retrieval_fusion import apply_constraint_penalty, fuse_branch_rankings


def test_reciprocal_rank_fusion_prefers_consistent_cross_branch_hits() -> None:
    rankings = {
        "dense_a": [10, 20, 30],
        "dense_b": [20, 10, 40],
        "ocr_text": [20, 50],
    }

    scores = fuse_branch_rankings(rankings)

    assert scores[20] > scores[10] > scores[30]


def test_hard_constraint_penalty_drops_non_matching_segment() -> None:
    item = {"segment_id": 9, "hard_constraints_passed": False}

    assert apply_constraint_penalty(item, 0.8) == 0.0
