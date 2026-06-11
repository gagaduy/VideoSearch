from app.services.retrieval_fusion import apply_constraint_penalty, fuse_branch_rankings


def test_reciprocal_rank_fusion_prefers_consistent_cross_branch_hits() -> None:
    rankings = {
        "dense_a": [10, 20, 30],
        "dense_b": [20, 10, 40],
        "ocr_text": [20, 50],
    }

    scores = fuse_branch_rankings(rankings)

    assert scores[20] > scores[10] > scores[30]


def test_weighted_branch_fusion_prefers_dense_hit_when_dense_weight_is_higher() -> None:
    rankings = {
        "dense_a": [10, 20],
        "ocr_text": [20, 10],
    }

    scores = fuse_branch_rankings(rankings)

    assert scores[10] > scores[20]


def test_weighted_branch_fusion_normalizes_single_branch_to_unit_scale() -> None:
    rankings = {
        "ocr_text": [50, 60, 70],
    }

    scores = fuse_branch_rankings(rankings)

    assert scores[50] == 1.0
    assert 0.0 < scores[60] < 1.0
    assert 0.0 < scores[70] < scores[60]


def test_hard_constraint_penalty_drops_non_matching_segment() -> None:
    item = {"segment_id": 9, "hard_constraints_passed": False}

    assert apply_constraint_penalty(item, 0.8) == 0.0


def test_count_refine_penalty_boosts_exact_count_and_penalizes_overcount() -> None:
    exact = {
        "segment_id": 1,
        "hard_constraints_passed": True,
        "count_refine_active": True,
        "count_refine_score": 1.0,
    }
    over = {
        "segment_id": 2,
        "hard_constraints_passed": True,
        "count_refine_active": True,
        "count_refine_score": 0.2,
    }

    assert apply_constraint_penalty(exact, 0.4) > 0.4
    assert apply_constraint_penalty(over, 0.4) < 0.4
