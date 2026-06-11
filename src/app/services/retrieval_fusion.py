from __future__ import annotations

from collections import defaultdict


BRANCH_WEIGHTS: dict[str, float] = {
    "dense_a": 0.30,
    "dense_b": 0.20,
    "ocr_text": 0.20,
    "object_entity": 0.20,
    "temporal_seed": 0.10,
}


def _normalized_rrf_scores(ranking: list[int], k: int) -> dict[int, float]:
    if not ranking:
        return {}
    raw_scores = {int(segment_id): 1.0 / (k + index) for index, segment_id in enumerate(ranking, start=1)}
    max_score = max(raw_scores.values(), default=0.0)
    if max_score <= 0.0:
        return {segment_id: 0.0 for segment_id in raw_scores}
    return {segment_id: score / max_score for segment_id, score in raw_scores.items()}


def fuse_branch_rankings(rankings: dict[str, list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    active_weight = 0.0
    for branch_name, ranking in rankings.items():
        weight = float(BRANCH_WEIGHTS.get(branch_name, 0.0))
        if weight <= 0.0 or not ranking:
            continue
        branch_scores = _normalized_rrf_scores(ranking, k)
        active_weight += weight
        for segment_id, branch_score in branch_scores.items():
            scores[segment_id] += weight * branch_score
    if active_weight <= 0.0:
        return {}
    for segment_id in list(scores):
        scores[segment_id] /= active_weight
    return dict(scores)


def apply_constraint_penalty(item: dict[str, object], score: float) -> float:
    if not item.get("hard_constraints_passed", True):
        return 0.0
    if item.get("count_refine_active", False):
        return score + (0.4 * (float(item.get("count_refine_score", 0.0)) - 0.5))
    return score
