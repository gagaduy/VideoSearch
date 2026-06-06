from __future__ import annotations

from collections import defaultdict


def fuse_branch_rankings(rankings: dict[str, list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings.values():
        for index, segment_id in enumerate(ranking, start=1):
            scores[segment_id] += 1.0 / (k + index)
    return dict(scores)


def apply_constraint_penalty(item: dict[str, object], score: float) -> float:
    return score if item.get("hard_constraints_passed", True) else 0.0
