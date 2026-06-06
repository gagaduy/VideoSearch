from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from app.db.session import get_session_factory
from app.services.search_service import run_search


def evaluate_queries(
    queries: list[dict[str, object]],
    search_fn: Callable[[str, list[str]], dict[str, object]],
) -> dict[str, float]:
    hits_at_10 = 0
    reciprocal_ranks: list[float] = []
    for item in queries:
        results = search_fn(str(item["query"]), list(item.get("object_labels", []))).get("results", [])[:10]
        expected = {int(segment_id) for segment_id in item.get("expected_segment_ids", [])}
        ranks = [
            index
            for index, result in enumerate(results, start=1)
            if int(result["segment_id"]) in expected
        ]
        if ranks:
            hits_at_10 += 1
            reciprocal_ranks.append(1.0 / ranks[0])
        else:
            reciprocal_ranks.append(0.0)

    query_count = max(len(queries), 1)
    return {
        "query_count": float(len(queries)),
        "recall_at_10": hits_at_10 / query_count,
        "mrr_at_10": sum(reciprocal_ranks) / query_count,
    }


def run_fixture(path: str | Path) -> dict[str, float]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text())

    def _search(query: str, object_labels: list[str]) -> dict[str, object]:
        session = get_session_factory()()
        try:
            return run_search(session, query, object_labels)
        finally:
            session.close()

    return evaluate_queries(payload, _search)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality from labeled query fixtures.")
    parser.add_argument("fixture", help="Path to the JSON fixture file")
    args = parser.parse_args()
    print(json.dumps(run_fixture(args.fixture), indent=2, sort_keys=True))
