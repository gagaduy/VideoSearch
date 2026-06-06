from scripts.evaluate_retrieval import evaluate_queries


def test_evaluate_retrieval_emits_metrics() -> None:
    def _fake_run_search(query: str, object_labels: list[str]) -> dict[str, object]:
        if query == "red boat":
            return {"results": [{"segment_id": 5}, {"segment_id": 9}]}
        return {"results": [{"segment_id": 2}]}

    payload = evaluate_queries(
        [
            {"query": "red boat", "object_labels": [], "expected_segment_ids": [5]},
            {"query": "fish", "object_labels": [], "expected_segment_ids": [7]},
        ],
        _fake_run_search,
    )

    assert payload["recall_at_10"] == 0.5
    assert payload["mrr_at_10"] == 0.5
    assert payload["query_count"] == 2
