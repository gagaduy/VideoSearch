from app.services.query_understanding import ObjectFilter, TemporalStep
from app.services.retrieval_branches import collect_branch_candidates


def test_collect_branch_candidates_returns_per_branch_rankings(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.retrieval_branches.search_dense_branch",
        lambda db, query_embedding, column_name, limit=80: [{"segment_id": 1, "branch": column_name}],
    )
    monkeypatch.setattr(
        "app.services.retrieval_branches.search_text_branch",
        lambda db, query_terms, limit=80: [{"segment_id": 2, "branch": "text"}],
    )
    monkeypatch.setattr(
        "app.services.retrieval_branches.search_object_branch",
        lambda db, object_filters, limit=80: [{"segment_id": 3, "branch": "object"}],
    )
    monkeypatch.setattr(
        "app.services.retrieval_branches.search_temporal_seed_branch",
        lambda db, temporal_steps, limit=80: [{"segment_id": 4, "branch": "temporal"}],
    )

    class _DenseEncoder:
        def embed_text(self, text: str):
            return type("Embedding", (), {"values": [0.1, 0.2, 0.3]})()

    result = collect_branch_candidates(
        db=None,
        semantic_query="red boat",
        expanded_queries=["red boat"],
        object_filters=[ObjectFilter(label="boat", min_count=1)],
        temporal_steps=[TemporalStep(text="boat appears")],
        dense_encoder=_DenseEncoder(),
    )

    assert "dense_a" in result
    assert "dense_b" in result
    assert "ocr_text" in result
    assert "object_entity" in result
    assert "temporal_seed" in result
