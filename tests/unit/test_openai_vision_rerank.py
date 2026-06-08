from app.services.openai_vision_rerank import (
    blend_rerank_score,
    parse_vision_rerank_scores,
    run_openai_vision_rerank,
    select_rerank_candidates,
    should_run_openai_vision_rerank,
)


def test_selects_at_most_top_k_candidates() -> None:
    rows = [{"frame_id": index, "score": 1.0 - (index * 0.01)} for index in range(12)]
    selected = select_rerank_candidates(rows, top_k=8)
    assert len(selected) == 8
    assert selected[0]["frame_id"] == 0


def test_blend_rerank_score_uses_045_055_weights() -> None:
    value = blend_rerank_score(local_score=0.4, vision_score=0.9, local_weight=0.45, vision_weight=0.55)
    assert value == 0.675


def test_skip_rerank_without_api_key() -> None:
    decision = should_run_openai_vision_rerank(enabled=True, api_key="", candidates=[{"frame_id": 1}])
    assert decision is False


def test_parse_vision_rerank_scores_returns_frame_score_map() -> None:
    payload = {
        "items": [
            {"frame_id": 11, "vision_score": 0.91},
            {"frame_id": 15, "vision_score": 0.44},
        ]
    }

    assert parse_vision_rerank_scores(payload) == {11: 0.91, 15: 0.44}


def test_parse_vision_rerank_scores_ignores_invalid_rows() -> None:
    payload = {"items": [{"frame_id": "oops"}, {"vision_score": 0.3}]}
    assert parse_vision_rerank_scores(payload) == {}


def test_run_openai_vision_rerank_returns_empty_scores_on_exception(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.openai_vision_rerank._request_openai_vision_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    scores = run_openai_vision_rerank("query", [{"frame_id": 11, "image_url": "/x"}])

    assert scores == {}
