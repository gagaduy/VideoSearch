# OpenAI Vision Rerank Top 8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OpenAI-powered vision rerank stage that reorders the top 8 local search results using blended scoring while preserving safe fallback to local-only retrieval.

**Architecture:** Keep the current local retrieval pipeline as stage-one recall, then pass up to 8 top candidates through a dedicated OpenAI vision rerank service. Blend `local_score` and `vision_score` at `0.45 / 0.55`, and return local results unchanged whenever OpenAI is unavailable, times out, or fails.

**Tech Stack:** FastAPI, existing search service layer, OpenAI API integration, static web client compatibility, pytest

---

## File Map

- Modify: `src/app/config.py`
  - Add rerank feature flag, top-K limit, timeout, and score weights.
- Create: `src/app/services/openai_vision_rerank.py`
  - Isolated service for building the OpenAI request, parsing results, and applying fallback rules.
- Modify: `src/app/services/search_service.py`
  - Call the rerank stage after local ranking and blend candidate scores.
- Test: `tests/unit/test_config.py`
  - Cover rerank config defaults.
- Create: `tests/unit/test_openai_vision_rerank.py`
  - Cover request shaping, score blending helpers, and fallback behavior.
- Modify: `tests/unit/test_search_service.py`
  - Cover top-8 rerank application and skipped rerank paths.
- Modify: `tests/integration/test_search_api.py`
  - Cover search response stability when rerank is active and when it falls back.

### Task 1: Add Rerank Config Defaults

**Files:**
- Modify: `src/app/config.py`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing config test**

```python
def test_openai_vision_rerank_defaults() -> None:
    settings = Settings()

    assert settings.openai_vision_rerank_enabled is True
    assert settings.openai_vision_rerank_top_k == 8
    assert settings.openai_vision_rerank_timeout_sec == 8
    assert settings.openai_vision_rerank_local_weight == 0.45
    assert settings.openai_vision_rerank_model == "gpt-4.1-mini"
    assert settings.openai_vision_rerank_vision_weight == 0.55
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.conda/bin/python -m pytest tests/unit/test_config.py::test_openai_vision_rerank_defaults -v`
Expected: FAIL because rerank settings do not exist on `Settings`

- [ ] **Step 3: Write minimal implementation**

```python
class Settings(BaseSettings):
    # existing fields...
    openai_vision_rerank_enabled: bool = True
    openai_vision_rerank_top_k: int = 8
    openai_vision_rerank_timeout_sec: int = 8
    openai_vision_rerank_local_weight: float = 0.45
    openai_vision_rerank_vision_weight: float = 0.55
    openai_vision_rerank_model: str = "gpt-4.1-mini"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.conda/bin/python -m pytest tests/unit/test_config.py::test_openai_vision_rerank_defaults -v`
Expected: PASS

### Task 2: Add Isolated Vision Rerank Service

**Files:**
- Create: `src/app/services/openai_vision_rerank.py`
- Create: `tests/unit/test_openai_vision_rerank.py`

- [ ] **Step 1: Write the failing unit tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.conda/bin/python -m pytest tests/unit/test_openai_vision_rerank.py -v`
Expected: FAIL because the new service file and helpers do not exist

- [ ] **Step 3: Write minimal implementation**

```python
def select_rerank_candidates(results: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    return list(results[:top_k])


def blend_rerank_score(*, local_score: float, vision_score: float, local_weight: float, vision_weight: float) -> float:
    return round((local_score * local_weight) + (vision_score * vision_weight), 3)


def should_run_openai_vision_rerank(*, enabled: bool, api_key: str, candidates: list[dict[str, object]]) -> bool:
    return bool(enabled and api_key and candidates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.conda/bin/python -m pytest tests/unit/test_openai_vision_rerank.py -v`
Expected: PASS

### Task 3: Add OpenAI Response Parsing And Fallback Rules

**Files:**
- Modify: `src/app/services/openai_vision_rerank.py`
- Modify: `tests/unit/test_openai_vision_rerank.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.conda/bin/python -m pytest tests/unit/test_openai_vision_rerank.py::test_parse_vision_rerank_scores_returns_frame_score_map tests/unit/test_openai_vision_rerank.py::test_parse_vision_rerank_scores_ignores_invalid_rows -v`
Expected: FAIL because parser helper does not exist

- [ ] **Step 3: Write minimal implementation**

```python
def parse_vision_rerank_scores(payload: dict[str, object]) -> dict[int, float]:
    items = payload.get("items", [])
    scores: dict[int, float] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        frame_id = item.get("frame_id")
        vision_score = item.get("vision_score")
        if isinstance(frame_id, int) and isinstance(vision_score, (int, float)):
            scores[int(frame_id)] = float(vision_score)
    return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.conda/bin/python -m pytest tests/unit/test_openai_vision_rerank.py::test_parse_vision_rerank_scores_returns_frame_score_map tests/unit/test_openai_vision_rerank.py::test_parse_vision_rerank_scores_ignores_invalid_rows -v`
Expected: PASS

### Task 4: Wire Top-8 Rerank Into Search Service

**Files:**
- Modify: `src/app/services/search_service.py`
- Modify: `tests/unit/test_search_service.py`

- [ ] **Step 1: Write the failing search-service tests**

```python
def test_apply_openai_vision_rerank_updates_only_top_8(monkeypatch) -> None:
    rows = [{"frame_id": index + 1, "score": 0.8 - (index * 0.01)} for index in range(10)]

    monkeypatch.setattr(
        "app.services.search_service.run_openai_vision_rerank",
        lambda query, candidates: {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8},
    )

    reranked = search_service._apply_openai_vision_rerank("query", rows)

    assert len(reranked) == 10
    assert reranked[0]["frame_id"] == 8
    assert reranked[-1]["frame_id"] == 10


def test_apply_openai_vision_rerank_returns_local_results_when_disabled(monkeypatch) -> None:
    rows = [{"frame_id": 1, "score": 0.6}]
    monkeypatch.setattr(search_service.settings, "openai_vision_rerank_enabled", False)
    assert search_service._apply_openai_vision_rerank("query", rows) == rows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.conda/bin/python -m pytest tests/unit/test_search_service.py::test_apply_openai_vision_rerank_updates_only_top_8 tests/unit/test_search_service.py::test_apply_openai_vision_rerank_returns_local_results_when_disabled -v`
Expected: FAIL because `_apply_openai_vision_rerank` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
def _apply_openai_vision_rerank(query: str, results: list[dict[str, object]]) -> list[dict[str, object]]:
    if not should_run_openai_vision_rerank(
        enabled=settings.openai_vision_rerank_enabled,
        api_key=settings.openai_api_key,
        candidates=results,
    ):
        return results

    top_candidates = select_rerank_candidates(results, settings.openai_vision_rerank_top_k)
    vision_scores = run_openai_vision_rerank(query, top_candidates)
    updated: list[dict[str, object]] = []
    for index, row in enumerate(results):
        frame_id = int(row.get("frame_id") or -1)
        if index < settings.openai_vision_rerank_top_k and frame_id in vision_scores:
            local_score = float(row.get("score", 0.0) or 0.0)
            row = {
                **row,
                "score": blend_rerank_score(
                    local_score=local_score,
                    vision_score=float(vision_scores[frame_id]),
                    local_weight=settings.openai_vision_rerank_local_weight,
                    vision_weight=settings.openai_vision_rerank_vision_weight,
                ),
            }
        updated.append(row)
    return sorted(updated, key=lambda item: float(item.get("score", 0.0)), reverse=True)
```

- [ ] **Step 4: Call the helper from `run_search` after local filtering**

```python
results = _filter_display_results(
    results,
    threshold=settings.text_result_score_threshold,
    limit=settings.search_result_display_limit,
)
results = _apply_openai_vision_rerank(query, results)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.conda/bin/python -m pytest tests/unit/test_search_service.py::test_apply_openai_vision_rerank_updates_only_top_8 tests/unit/test_search_service.py::test_apply_openai_vision_rerank_returns_local_results_when_disabled -v`
Expected: PASS

### Task 5: Add Safe OpenAI Failure Fallback

**Files:**
- Modify: `src/app/services/openai_vision_rerank.py`
- Modify: `tests/unit/test_openai_vision_rerank.py`

- [ ] **Step 1: Write the failing fallback test**

```python
def test_run_openai_vision_rerank_returns_empty_scores_on_exception(monkeypatch) -> None:
    monkeypatch.setattr("app.services.openai_vision_rerank._request_openai_vision_scores", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    scores = run_openai_vision_rerank("query", [{"frame_id": 11, "image_url": "/x"}])

    assert scores == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.conda/bin/python -m pytest tests/unit/test_openai_vision_rerank.py::test_run_openai_vision_rerank_returns_empty_scores_on_exception -v`
Expected: FAIL because `run_openai_vision_rerank` or its fallback path does not exist yet

- [ ] **Step 3: Write minimal implementation**

```python
def run_openai_vision_rerank(query: str, candidates: list[dict[str, object]]) -> dict[int, float]:
    try:
        payload = _request_openai_vision_scores(query, candidates)
    except Exception:
        return {}
    return parse_vision_rerank_scores(payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.conda/bin/python -m pytest tests/unit/test_openai_vision_rerank.py::test_run_openai_vision_rerank_returns_empty_scores_on_exception -v`
Expected: PASS

### Task 6: Keep API Response Stable

**Files:**
- Modify: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write the failing integration test**

```python
def test_search_endpoint_returns_results_when_openai_rerank_falls_back(monkeypatch, tmp_path: Path) -> None:
    _stub_indexing_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr("app.services.search_service.run_openai_vision_rerank", lambda query, candidates: {})

    client = TestClient(app)
    created = client.post("/videos", json={"filename": "search-rerank.mp4", "source_path": "./data/videos/search-rerank.mp4"})
    job_id = created.json()["job"]["id"]
    client.post(f"/jobs/{job_id}/run")

    response = client.post("/search", json={"query": "frame placeholder", "object_labels": []})

    assert response.status_code == 200
    assert response.json()["results"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.conda/bin/python -m pytest tests/integration/test_search_api.py::test_search_endpoint_returns_results_when_openai_rerank_falls_back -v`
Expected: FAIL until rerank wiring is active and import path exists

- [ ] **Step 3: Run test again after wiring is complete**

Run: `./.conda/bin/python -m pytest tests/integration/test_search_api.py::test_search_endpoint_returns_results_when_openai_rerank_falls_back -v`
Expected: PASS

### Task 7: Focused Verification

**Files:**
- Modify: none
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_openai_vision_rerank.py`
- Test: `tests/unit/test_search_service.py`
- Test: `tests/integration/test_search_api.py`

- [ ] **Step 1: Run the focused suite**

Run: `./.conda/bin/python -m pytest tests/unit/test_config.py tests/unit/test_openai_vision_rerank.py tests/unit/test_search_service.py tests/integration/test_search_api.py -v`
Expected: PASS

- [ ] **Step 2: Run one adjacent regression**

Run: `./.conda/bin/python -m pytest tests/integration/test_video_api.py tests/unit/test_web_assets.py -v`
Expected: PASS, confirming rerank did not break web/API baseline behavior

- [ ] **Step 3: Manual smoke check**

Procedure:

- ensure `API` is running
- run a text query that previously had near-miss ordering
- compare result ordering with rerank enabled and disabled
- confirm search still returns quickly enough and still works when `OPENAI_API_KEY` is removed

Expected:

- rerank changes only the top of the list
- fallback returns valid local results
- no DB migration or reindexing is needed
