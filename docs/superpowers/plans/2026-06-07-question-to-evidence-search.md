# Question To Evidence Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated `Search By Question` mode that ranks frames by how useful they are for answering a natural-language question, with OCR/caption-biased local retrieval and OpenAI rerank on top candidates.

**Architecture:** The backend adds a Q&A-specific retrieval service instead of overloading normal text search. It interprets the full question as an evidence request, gathers OCR/caption/semantic-heavy candidates, reranks the top candidates with OpenAI for answer-bearing usefulness, and returns the existing frame-card payload to a new sidebar panel in the web UI.

**Tech Stack:** FastAPI, SQLAlchemy, existing retrieval services, OpenAI Responses API, plain web frontend, pytest.

---

## File Map

- Create: `src/app/services/question_search.py`
  Implements question parsing, evidence-term extraction, local evidence-biased retrieval, and OpenAI rerank orchestration.
- Modify: `src/app/config.py`
  Add Q&A search tuning values such as candidate pool size and rerank top-k.
- Modify: `src/app/services/openai_vision_rerank.py`
  Add a question-answerability rerank path or prompt variant for answer-bearing evidence.
- Modify: `src/app/services/search_service.py`
  Reuse result payload helpers and threshold logic so Q&A search returns the same card shape as other modes.
- Modify: `src/app/api/routes/search.py`
  Add a dedicated endpoint for question search.
- Modify: `src/app/schemas/search.py`
  Add request schema for Q&A search if separate payload typing is useful.
- Create: `tests/unit/test_question_search.py`
  Covers evidence-term extraction, OCR/caption-biased local ranking, and rerank fallback.
- Modify: `tests/integration/test_search_api.py`
  Add endpoint coverage for question search.
- Modify: `web/index.html`
  Add `Search By Question` panel without replacing text or video clip search.
- Modify: `web/app.js`
  Add form submission, in-flight guarding, and result rendering for question search.
- Modify: `web/styles.css`
  Add any minimal styles needed for the new panel and question textarea/input.
- Modify: `tests/unit/test_web_assets.py`
  Assert that the new question-search panel exists.

### Task 1: Add Failing Tests for Question Search Core Behavior

**Files:**
- Create: `tests/unit/test_question_search.py`
- Modify: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write failing unit tests for evidence extraction and ranking helpers**

```python
from app.services.question_search import (
    build_question_evidence_terms,
    score_evidence_row,
)


def test_build_question_evidence_terms_extracts_textual_clues():
    question = "Ten cua loai virus nay la gi? Doan video ve viec che tao vaccine phong mot loai virus."

    terms = build_question_evidence_terms(question)

    assert "virus" in terms
    assert "vaccine" in terms


def test_score_evidence_row_prefers_ocr_overlap():
    row = {
        "caption_text": "scientist explaining a vaccine",
        "ocr_text": "virus vaccine trial information",
        "labels": [],
        "object_counts": {},
        "object_positions": {},
        "semantic_entities": [],
        "semantic_counts": {},
    }

    score = score_evidence_row(row, ["virus", "vaccine"])

    assert score > 0.5
```

- [ ] **Step 2: Run the unit tests to verify they fail**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_question_search.py -v`

Expected: FAIL with missing module or missing symbol errors.

- [ ] **Step 3: Write failing integration tests for the new API route**

```python
def test_question_search_route_is_registered() -> None:
    client = TestClient(app)
    response = client.options(
        "/search/question",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code != 404


def test_question_search_returns_results(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "app.api.routes.search.run_question_search",
        lambda db, question: {
            "mode": "question",
            "query": question,
            "expanded_queries": [],
            "results": [{"frame_id": 11, "score": 0.81, "thumb_url": "/media/frames/11/thumb"}],
            "parsed_query": None,
        },
        raising=False,
    )

    response = client.post("/search/question", json={"question": "What is the virus name?"})

    assert response.status_code == 200
    assert response.json()["mode"] == "question"
    assert response.json()["results"][0]["frame_id"] == 11
```

- [ ] **Step 4: Run the targeted integration tests to verify they fail**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_search_api.py -k "question_search" -v`

Expected: FAIL because the route does not exist yet.

- [ ] **Step 5: Commit the red tests**

```bash
git add tests/unit/test_question_search.py tests/integration/test_search_api.py
git commit -m "test: add failing coverage for question search"
```

### Task 2: Add Configuration and Implement Question Search Service

**Files:**
- Modify: `src/app/config.py`
- Create: `src/app/services/question_search.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_question_search.py`

- [ ] **Step 1: Add failing config assertions**

```python
def test_question_search_defaults() -> None:
    settings = Settings()

    assert settings.question_search_candidate_pool == 24
    assert settings.question_search_rerank_top_k == 8
```

- [ ] **Step 2: Run the config test to verify it fails**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_config.py -k "question_search" -v`

Expected: FAIL because the settings do not exist yet.

- [ ] **Step 3: Add minimal settings**

```python
question_search_candidate_pool: int = 24
question_search_rerank_top_k: int = 8
```

- [ ] **Step 4: Implement minimal question search helpers**

```python
import re


def build_question_evidence_terms(question: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", question.lower()) if len(token) > 2]


def score_evidence_row(row: dict[str, object], evidence_terms: list[str]) -> float:
    ocr_text = str(row.get("ocr_text", "")).lower()
    caption_text = str(row.get("caption_text", "")).lower()
    if not evidence_terms:
        return 0.0
    ocr_hits = sum(1 for term in evidence_terms if term in ocr_text)
    caption_hits = sum(1 for term in evidence_terms if term in caption_text)
    return min(1.0, ((ocr_hits * 1.5) + caption_hits) / max(len(evidence_terms), 1))
```

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_config.py tests/unit/test_question_search.py -v`

Expected: PASS

- [ ] **Step 6: Commit the service foundation**

```bash
git add src/app/config.py src/app/services/question_search.py tests/unit/test_config.py tests/unit/test_question_search.py
git commit -m "feat: add question search service foundation"
```

### Task 3: Add the Q&A Search API Route and Shared Result Payload Wiring

**Files:**
- Modify: `src/app/api/routes/search.py`
- Modify: `src/app/schemas/search.py`
- Modify: `src/app/services/search_service.py`
- Modify: `tests/integration/test_search_api.py`

- [ ] **Step 1: Add a failing route assertion if needed**

```python
def test_question_search_route_uses_json_payload() -> None:
    client = TestClient(app)
    response = client.post("/search/question", json={"question": ""})
    assert response.status_code in {200, 400, 422}
```

- [ ] **Step 2: Run the route test to verify it fails**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_search_api.py -k "question_search_route" -v`

Expected: FAIL because the route does not exist yet.

- [ ] **Step 3: Add request schema and route**

```python
class QuestionSearchRequest(BaseModel):
    question: str


@router.post("/search/question", response_model=SearchResponse)
def search_by_question(payload: QuestionSearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    return SearchResponse.model_validate(run_question_search(db, payload.question))
```

- [ ] **Step 4: Reuse the existing result payload helper**

Use the same frame-card structure already returned by image/video/text search. Do not invent a second result format.

- [ ] **Step 5: Run the targeted integration tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_search_api.py -k "question_search" -v`

Expected: PASS

- [ ] **Step 6: Commit the API layer**

```bash
git add src/app/api/routes/search.py src/app/schemas/search.py src/app/services/search_service.py tests/integration/test_search_api.py
git commit -m "feat: add question search API route"
```

### Task 4: Implement OCR/Citation-Biased Local Retrieval and OpenAI Rerank

**Files:**
- Modify: `src/app/services/question_search.py`
- Modify: `src/app/services/openai_vision_rerank.py`
- Modify: `tests/unit/test_question_search.py`

- [ ] **Step 1: Add a failing unit test for local candidate ordering**

```python
def test_question_search_prefers_answer_bearing_text_rows():
    rows = [
        {
            "segment_id": 1,
            "ocr_text": "virus vaccine information appears here",
            "caption_text": "screen with text",
            "labels": [],
            "object_counts": {},
            "object_positions": {},
            "semantic_entities": [],
            "semantic_counts": {},
        },
        {
            "segment_id": 2,
            "ocr_text": "",
            "caption_text": "people talking in a room",
            "labels": ["person"],
            "object_counts": {"person": 2},
            "object_positions": {},
            "semantic_entities": [],
            "semantic_counts": {},
        },
    ]

    ranked = rank_question_candidates(rows, "What is the virus name?")

    assert ranked[0]["segment_id"] == 1
```

- [ ] **Step 2: Run the targeted unit tests to verify they fail**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_question_search.py -v`

Expected: FAIL if the ranking function does not exist or does not yet prefer the OCR-heavy row.

- [ ] **Step 3: Implement the local ranking path**

```python
def rank_question_candidates(rows: list[dict[str, object]], question: str) -> list[dict[str, object]]:
    evidence_terms = build_question_evidence_terms(question)
    ranked = []
    for row in rows:
        ranked.append(
            {
                **row,
                "_question_local_score": score_evidence_row(row, evidence_terms),
            }
        )
    return sorted(ranked, key=lambda item: float(item["_question_local_score"]), reverse=True)
```

Then wire it into `run_question_search()` so that:

- local candidates come from the existing corpus
- OCR/caption-biased scores influence ranking strongly
- top-k candidates are passed to OpenAI rerank

- [ ] **Step 4: Add or extend OpenAI rerank prompt for answer-bearing usefulness**

```python
def run_openai_question_rerank(question: str, candidates: list[dict[str, object]]) -> dict[int, float]:
    ...
```

The prompt should ask: which candidate frame is most useful for answering the question, not which frame is most visually similar.

- [ ] **Step 5: Run focused backend tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_question_search.py tests/integration/test_search_api.py -k "question_search" -v`

Expected: PASS

- [ ] **Step 6: Commit the retrieval and rerank path**

```bash
git add src/app/services/question_search.py src/app/services/openai_vision_rerank.py tests/unit/test_question_search.py tests/integration/test_search_api.py
git commit -m "feat: rank evidence frames for question search"
```

### Task 5: Add the `Search By Question` UI Panel

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Write failing web asset assertions**

```python
def test_index_contains_question_search_panel() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "Search By Question" in html
    assert 'id="question-search-form"' in html
```

- [ ] **Step 2: Run the web asset test to verify it fails**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_web_assets.py -k "question_search" -v`

Expected: FAIL because the panel does not exist yet.

- [ ] **Step 3: Add the sidebar panel markup**

```html
<section class="panel">
  <h2>Search By Question</h2>
  <form id="question-search-form" class="stack">
    <textarea id="question-query" name="question" rows="4" placeholder="What is the name of the virus shown in this video?"></textarea>
    <button type="submit">Find Evidence Frames</button>
  </form>
  <p class="muted">Find frames that are most likely to contain information needed to answer the question.</p>
</section>
```

- [ ] **Step 4: Add client-side behavior**

```javascript
let questionSearchInFlight = false;

async function submitQuestionSearch(question) {
  const response = await fetch(`${apiBase}/search/question`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return response.json();
}
```

The implementation must:

- block duplicate submits
- validate empty input
- render returned results through the existing result grid
- show a Q&A-specific empty state message

- [ ] **Step 5: Run the web asset tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_web_assets.py -v`

Expected: PASS

- [ ] **Step 6: Commit the UI panel**

```bash
git add web/index.html web/app.js web/styles.css tests/unit/test_web_assets.py
git commit -m "feat: add question search UI panel"
```

### Task 6: Full Verification and Manual Smoke Test

**Files:**
- No new files

- [ ] **Step 1: Run the focused Q&A suite**

Run:

```bash
PYTHONPATH=src ./.conda/bin/python -m pytest \
  tests/unit/test_config.py \
  tests/unit/test_question_search.py \
  tests/integration/test_search_api.py \
  tests/unit/test_web_assets.py -v
```

Expected: PASS

- [ ] **Step 2: Run adjacent regressions**

Run:

```bash
PYTHONPATH=src ./.conda/bin/python -m pytest \
  tests/unit/test_video_query_search.py \
  tests/unit/test_search_service.py \
  tests/integration/test_video_api.py \
  tests/unit/test_media_preview.py -v
```

Expected: PASS

- [ ] **Step 3: Manual smoke test**

Checklist:

- open `http://localhost:8080`
- submit a long natural-language question in `Search By Question`
- confirm the button disables while searching
- confirm results render in the grid
- confirm clicking a result still updates timeline preview
- confirm empty question is blocked
- confirm text search and video clip search still work

- [ ] **Step 4: Commit any final test-driven fixes**

```bash
git add src/app web tests
git commit -m "test: verify question search end to end"
```

## Self-Review

- Spec coverage: the plan covers a new dedicated Q&A panel, OCR/caption-biased local retrieval, OpenAI answer-bearing rerank, route addition, UI behavior, and regression testing.
- Placeholder scan: all tasks have explicit files, commands, and code snippets; no `TODO` markers remain.
- Type consistency: the plan consistently uses `run_question_search`, `build_question_evidence_terms`, `score_evidence_row`, and `rank_question_candidates` across tasks.
