# UI Layout And Image Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the web UI into a full-page two-pane layout, reduce noisy result rendering with threshold-plus-cap filtering, add image-to-frame search, and block repeated search submissions while requests are active.

**Architecture:** Extend the existing search backend with a dedicated image-search endpoint and a shared result filtering helper so text and image modes can reuse the same rendering path. On the frontend, replace the stacked page with a viewport workspace that keeps control forms in a left rail and results in a right pane, while adding request-state guards to stop duplicate in-flight submissions.

**Tech Stack:** FastAPI, SQLAlchemy/pgvector, existing OpenCLIP adapter, static HTML/CSS/JavaScript, pytest

---

## File Map

- Modify: `src/app/config.py`
  - Add calibrated result display limits and thresholds for text and image retrieval.
- Modify: `src/app/schemas/search.py`
  - Add request/response schema support for image-search metadata if needed.
- Modify: `src/app/services/search_service.py`
  - Add shared post-filter helper, image-search service entrypoint, and response shaping.
- Modify: `src/app/api/routes/search.py`
  - Add the new image-search route and wire it to the service layer.
- Modify: `src/worker/adapters/openclip_adapter.py`
  - Reuse existing image embedding capability for uploaded query images if any helper signature adjustments are needed.
- Modify: `web/index.html`
  - Replace stacked page structure with left rail and right workspace; add image-search form and active query image slot.
- Modify: `web/styles.css`
  - Implement full-viewport workspace layout, internal scroll regions, responsive behavior, and loading/empty states.
- Modify: `web/app.js`
  - Add text/image search request guards, image-search submit flow, shared result rendering, and empty-state handling.
- Test: `tests/unit/test_config.py`
  - Cover new threshold/cap config defaults.
- Test: `tests/unit/test_search_service.py`
  - Cover threshold-plus-cap filtering and image-search result shaping.
- Test: `tests/integration/test_search_api.py`
  - Cover image-search upload route and search response contract.
- Test: `tests/unit/test_web_assets.py`
  - Cover new UI form structure and search-button blocking hooks.

### Task 1: Add Config For Display Filtering

**Files:**
- Modify: `src/app/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing config test**

```python
def test_result_display_defaults() -> None:
    settings = Settings()
    assert settings.search_result_display_limit == 16
    assert settings.text_result_score_threshold == 0.18
    assert settings.image_result_score_threshold == 0.20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_result_display_defaults -v`
Expected: FAIL with missing `search_result_display_limit` and threshold settings on `Settings`

- [ ] **Step 3: Write minimal implementation**

```python
class Settings(BaseSettings):
    # existing fields...
    search_result_display_limit: int = 16
    text_result_score_threshold: float = 0.18
    image_result_score_threshold: float = 0.20
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py::test_result_display_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/config.py tests/unit/test_config.py
git commit -m "feat: add search result display config"
```

### Task 2: Add Shared Result Filtering For Text Search

**Files:**
- Modify: `src/app/services/search_service.py`
- Test: `tests/unit/test_search_service.py`

- [ ] **Step 1: Write the failing service tests**

```python
def test_filter_results_applies_threshold_before_cap() -> None:
    rows = [
        {"score": 0.31, "segment_id": 1},
        {"score": 0.29, "segment_id": 2},
        {"score": 0.17, "segment_id": 3},
    ]

    filtered = _filter_display_results(rows, threshold=0.18, limit=2)

    assert [row["segment_id"] for row in filtered] == [1, 2]


def test_filter_results_returns_empty_when_no_row_meets_threshold() -> None:
    rows = [
        {"score": 0.12, "segment_id": 1},
        {"score": 0.10, "segment_id": 2},
    ]

    assert _filter_display_results(rows, threshold=0.18, limit=16) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_search_service.py::test_filter_results_applies_threshold_before_cap tests/unit/test_search_service.py::test_filter_results_returns_empty_when_no_row_meets_threshold -v`
Expected: FAIL with `_filter_display_results` undefined

- [ ] **Step 3: Write minimal implementation**

```python
def _filter_display_results(rows: list[dict[str, object]], threshold: float, limit: int) -> list[dict[str, object]]:
    kept = [
        row for row in rows
        if float(row.get("score", 0.0) or 0.0) >= threshold
    ]
    return kept[:limit]
```

- [ ] **Step 4: Apply helper inside text search**

```python
results = _filter_display_results(
    results,
    threshold=settings.text_result_score_threshold,
    limit=settings.search_result_display_limit,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_search_service.py::test_filter_results_applies_threshold_before_cap tests/unit/test_search_service.py::test_filter_results_returns_empty_when_no_row_meets_threshold -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/app/services/search_service.py tests/unit/test_search_service.py
git commit -m "feat: filter displayed text search results"
```

### Task 3: Add Image Search Service Flow

**Files:**
- Modify: `src/app/services/search_service.py`
- Modify: `src/app/schemas/search.py`
- Test: `tests/unit/test_search_service.py`

- [ ] **Step 1: Write the failing image-search service test**

```python
def test_run_image_search_returns_mode_and_filtered_results(tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "query.png"
    image_path.write_bytes(b"fake-image")

    monkeypatch.setattr("app.services.search_service._embed_query_image", lambda path: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "app.services.search_service.search_segments_by_embedding",
        lambda db, query_embedding, limit=80: [
            {"segment_id": 11, "score": 0.42, "keyframe_id": 5, "caption_text": "red car"},
            {"segment_id": 12, "score": 0.11, "keyframe_id": 6, "caption_text": "weak"},
        ],
    )
    monkeypatch.setattr("app.services.search_service.fetch_frame_media_map", lambda db, frame_ids: {5: {"image_path": "a.png", "thumb_path": "a.webp"}})

    payload = run_image_search(db_session, image_path)

    assert payload["mode"] == "image"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["segment_id"] == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_search_service.py::test_run_image_search_returns_mode_and_filtered_results -v`
Expected: FAIL with `run_image_search` undefined

- [ ] **Step 3: Write minimal implementation**

```python
def _embed_query_image(image_path: Path) -> list[float]:
    adapter = OpenClipAdapter(model_name=settings.embedding_model)
    return adapter.embed_image(str(image_path)).values


def run_image_search(db: Session, image_path: Path) -> dict[str, object]:
    query_embedding = _embed_query_image(image_path)
    rows = search_segments_by_embedding(db, query_embedding, limit=80)
    filtered = _filter_display_results(
        rows,
        threshold=settings.image_result_score_threshold,
        limit=settings.search_result_display_limit,
    )
    return {
        "mode": "image",
        "results": filtered,
    }
```

- [ ] **Step 4: Shape image-search results to match existing web rendering**

```python
return {
    "mode": "image",
    "query": image_path.name,
    "expanded_queries": [],
    "results": results,
    "parsed_query": None,
}
```

- [ ] **Step 5: Run the service test to verify it passes**

Run: `pytest tests/unit/test_search_service.py::test_run_image_search_returns_mode_and_filtered_results -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/app/services/search_service.py src/app/schemas/search.py tests/unit/test_search_service.py
git commit -m "feat: add image search service flow"
```

### Task 4: Expose Image Search API Endpoint

**Files:**
- Modify: `src/app/api/routes/search.py`
- Modify: `src/app/schemas/search.py`
- Test: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write the failing API test**

```python
def test_image_search_returns_ranked_results(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "app.api.routes.search.run_image_search",
        lambda db, image_path: {
            "mode": "image",
            "query": "query.png",
            "expanded_queries": [],
            "parsed_query": None,
            "results": [{"segment_id": 7, "score": 0.51}],
        },
    )

    response = client.post(
        "/search/image",
        files={"file": ("query.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "image"
    assert response.json()["results"][0]["segment_id"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_search_api.py::test_image_search_returns_ranked_results -v`
Expected: FAIL with missing `/search/image` route

- [ ] **Step 3: Write minimal implementation**

```python
@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> SearchResponse:
    suffix = Path(file.filename or "query.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)
    try:
        payload = run_image_search(db, temp_path)
        return SearchResponse.model_validate(payload)
    finally:
        temp_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Add a missing-file or invalid-content test**

```python
def test_image_search_rejects_non_image_upload(client: TestClient) -> None:
    response = client.post(
        "/search/image",
        files={"file": ("query.txt", b"not-image", "text/plain")},
    )

    assert response.status_code in {400, 422}
```

- [ ] **Step 5: Run API tests to verify they pass**

Run: `pytest tests/integration/test_search_api.py::test_image_search_returns_ranked_results tests/integration/test_search_api.py::test_image_search_rejects_non_image_upload -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/app/api/routes/search.py src/app/schemas/search.py tests/integration/test_search_api.py
git commit -m "feat: expose image search API"
```

### Task 5: Rebuild The Web Layout Around A Left Rail

**Files:**
- Modify: `web/index.html`
- Modify: `web/styles.css`
- Test: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Write the failing web asset tests**

```python
def test_index_contains_image_search_form() -> None:
    html = Path("web/index.html").read_text()
    assert 'id="image-search-form"' in html
    assert 'id="query-image"' in html


def test_index_uses_workspace_shell_layout() -> None:
    html = Path("web/index.html").read_text()
    assert 'class="workspace"' in html
    assert 'class="left-rail"' in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_web_assets.py::test_index_contains_image_search_form tests/unit/test_web_assets.py::test_index_uses_workspace_shell_layout -v`
Expected: FAIL because the new form and layout classes are missing

- [ ] **Step 3: Write minimal HTML structure**

```html
<main class="workspace">
  <aside class="left-rail">
    <section class="panel">...</section>
    <section class="panel">...</section>
    <section class="panel">
      <h2>Search By Image</h2>
      <form id="image-search-form" class="stack">
        <input id="query-image" name="file" type="file" accept="image/*" required>
        <button type="submit">Find Similar Frames</button>
      </form>
      <div id="query-image-preview"></div>
    </section>
  </aside>
  <section class="workspace-main">...</section>
</main>
```

- [ ] **Step 4: Write minimal CSS for viewport layout**

```css
.workspace {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
}

.left-rail {
  height: 100vh;
  overflow: auto;
}

.workspace-main {
  height: 100vh;
  overflow: hidden;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_web_assets.py::test_index_contains_image_search_form tests/unit/test_web_assets.py::test_index_uses_workspace_shell_layout -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add web/index.html web/styles.css tests/unit/test_web_assets.py
git commit -m "feat: add full-page search workspace layout"
```

### Task 6: Add Text Search Request Guard And Empty-State Rendering

**Files:**
- Modify: `web/app.js`
- Test: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Write the failing asset tests for request guards**

```python
def test_app_blocks_duplicate_text_search_submit() -> None:
    script = Path("web/app.js").read_text()
    assert "let textSearchInFlight = false;" in script
    assert "if (textSearchInFlight) {" in script
    assert 'submitButton.textContent = "Searching...";' in script


def test_app_renders_empty_state_message() -> None:
    script = Path("web/app.js").read_text()
    assert "No strong matches found for this query." in script
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_web_assets.py::test_app_blocks_duplicate_text_search_submit tests/unit/test_web_assets.py::test_app_renders_empty_state_message -v`
Expected: FAIL because the request guard and empty-state message are missing

- [ ] **Step 3: Write minimal request-guard implementation**

```javascript
let textSearchInFlight = false;

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (textSearchInFlight) {
    return;
  }

  textSearchInFlight = true;
  submitButton.disabled = true;
  submitButton.textContent = "Searching...";
  try {
    // existing fetch logic
  } finally {
    textSearchInFlight = false;
    submitButton.disabled = false;
    submitButton.textContent = "Search";
  }
});
```

- [ ] **Step 4: Render a no-results state instead of an empty grid**

```javascript
if (!data.results.length) {
  results.innerHTML = "<p class=\"empty-state\">No strong matches found for this query.</p>";
  return;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_web_assets.py::test_app_blocks_duplicate_text_search_submit tests/unit/test_web_assets.py::test_app_renders_empty_state_message -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add web/app.js tests/unit/test_web_assets.py
git commit -m "feat: guard text search submissions"
```

### Task 7: Add Image Search UI Flow

**Files:**
- Modify: `web/app.js`
- Modify: `web/index.html`
- Modify: `web/styles.css`
- Test: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Write the failing image-search asset tests**

```python
def test_app_contains_image_search_request_guard() -> None:
    script = Path("web/app.js").read_text()
    assert "let imageSearchInFlight = false;" in script
    assert 'fetch(`${apiBase}/search/image`' in script


def test_index_contains_query_image_preview_slot() -> None:
    html = Path("web/index.html").read_text()
    assert 'id="query-image-preview"' in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_web_assets.py::test_app_contains_image_search_request_guard tests/unit/test_web_assets.py::test_index_contains_query_image_preview_slot -v`
Expected: FAIL because image search UI flow is not implemented

- [ ] **Step 3: Write minimal image-search submit flow**

```javascript
let imageSearchInFlight = false;

imageSearchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (imageSearchInFlight) {
    return;
  }

  const formData = new FormData(imageSearchForm);
  const file = formData.get("file");
  if (!(file instanceof File) || !file.size) {
    results.innerHTML = "<p class=\"empty-state\">Choose an image before searching.</p>";
    return;
  }

  imageSearchInFlight = true;
  imageSubmitButton.disabled = true;
  imageSubmitButton.textContent = "Searching...";
  try {
    const response = await fetch(`${apiBase}/search/image`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    renderSearchResults(data);
  } finally {
    imageSearchInFlight = false;
    imageSubmitButton.disabled = false;
    imageSubmitButton.textContent = "Find Similar Frames";
  }
});
```

- [ ] **Step 4: Add active query-image preview rendering**

```javascript
const queryImageUrl = URL.createObjectURL(file);
queryImagePreview.innerHTML = `<img class="query-image-preview" src="${queryImageUrl}" alt="Query image preview">`;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_web_assets.py::test_app_contains_image_search_request_guard tests/unit/test_web_assets.py::test_index_contains_query_image_preview_slot -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add web/app.js web/index.html web/styles.css tests/unit/test_web_assets.py
git commit -m "feat: add image search UI flow"
```

### Task 8: Run Focused Regression Coverage

**Files:**
- Modify: none
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_search_service.py`
- Test: `tests/integration/test_search_api.py`
- Test: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Run the focused suite**

Run: `pytest tests/unit/test_config.py tests/unit/test_search_service.py tests/integration/test_search_api.py tests/unit/test_web_assets.py -v`
Expected: PASS for new config, filtering, image-search API, and web asset coverage

- [ ] **Step 2: Run one adjacent regression suite**

Run: `pytest tests/integration/test_video_api.py tests/integration/test_real_pipeline.py -v`
Expected: PASS, proving UI/search changes did not break upload or indexing-adjacent contracts

- [ ] **Step 3: Commit if any test-only follow-up fixes were needed**

```bash
git add tests/unit/test_config.py tests/unit/test_search_service.py tests/integration/test_search_api.py tests/unit/test_web_assets.py
git commit -m "test: finalize UI and image search coverage"
```

### Task 9: Manual Smoke Check

**Files:**
- Modify: none

- [ ] **Step 1: Start the local stack without clearing DB**

Run:

```bash
docker compose up -d postgres web
bash scripts/run_api.sh
bash scripts/run_worker.sh
```

Expected: `postgres` and `web` stay up, API health reaches `{"status":"ok"}`, worker polls jobs

- [ ] **Step 2: Verify the UI shell and both search forms**

Run:

```bash
curl -sf http://localhost:8000/health
curl -I -sf http://localhost:8080
```

Expected:

- API returns `{"status":"ok"}`
- web responds with `HTTP/1.1 200 OK`

- [ ] **Step 3: Manually test duplicate-submit blocking**

Procedure:

- open `http://localhost:8080`
- submit a text query and click the same search button repeatedly while loading
- confirm only one request is sent from the active form and the button stays disabled with a loading label
- submit an image query and confirm the same blocking behavior

Expected:

- no stacked duplicate requests from the same form
- empty-state message appears when no result clears the threshold
- image search preview remains visible while its results are active

## Self-Review Checklist

- Spec coverage:
  - full-page layout: Task 5
  - threshold-plus-cap filtering: Tasks 1-3
  - image-to-frame search: Tasks 3, 4, 7
  - duplicate-submit blocking: Tasks 6, 7, 9
  - empty/error states: Tasks 4, 6, 7
- Placeholder scan:
  - no `TODO`, `TBD`, or deferred implementation notes remain
- Type consistency:
  - `run_image_search`, `_filter_display_results`, and config field names are defined before later tasks depend on them
