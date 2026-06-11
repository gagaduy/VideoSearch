from app.services.query_understanding import ObjectFilter, TemporalStep
from app.services.retrieval_branches import collect_branch_candidates
from app.db.repositories.branch_search import search_object_branch
from app.db.models import Segment
from app.db.session import get_session_factory


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


def test_search_object_branch_prefers_exact_count_over_overcount() -> None:
    session = get_session_factory()()
    try:
        session.query(Segment).delete()
        exact = Segment(
            video_id=1,
            segment_index=1,
            start_timestamp_sec=0.0,
            end_timestamp_sec=1.0,
            keyframe_id=1,
            caption_text="cars on track",
            ocr_text="",
            object_labels_json=["car"],
            object_counts_json={"car": 3},
            object_positions_json={"car": ["center"]},
            semantic_entities_json=[],
            semantic_aliases_json={},
            semantic_counts_json={},
            raw_json={},
        )
        over = Segment(
            video_id=1,
            segment_index=2,
            start_timestamp_sec=1.0,
            end_timestamp_sec=2.0,
            keyframe_id=2,
            caption_text="more cars on track",
            ocr_text="",
            object_labels_json=["car"],
            object_counts_json={"car": 5},
            object_positions_json={"car": ["center"]},
            semantic_entities_json=[],
            semantic_aliases_json={},
            semantic_counts_json={},
            raw_json={},
        )
        session.add_all([exact, over])
        session.commit()

        results = search_object_branch(session, [ObjectFilter(label="car", min_count=3)], limit=10)

        assert [row["segment_id"] for row in results[:2]] == [exact.id, over.id]
        assert results[0]["object_score"] > results[1]["object_score"]
    finally:
        session.query(Segment).delete()
        session.commit()
        session.close()
