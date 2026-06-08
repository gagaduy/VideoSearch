# Short Video Query Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `Search By Image` with short uploaded video clip search that returns the most similar indexed frames using local multi-frame retrieval plus OpenAI vision rerank on the top 8 candidates.

**Architecture:** The backend adds a dedicated clip-query pipeline that extracts multiple frames from a temporary uploaded video, embeds them with the existing OpenCLIP adapter, merges candidate evidence across query frames, and then reranks the top 8 candidate frames with OpenAI vision. The frontend replaces the image-search sidebar panel with a video-upload panel while reusing the current result grid and preview pane.

**Tech Stack:** FastAPI, SQLAlchemy, OpenCLIP, ffmpeg/OpenCV-based frame extraction, OpenAI Responses API, plain web frontend, pytest.

---

## File Map

- Modify: `src/app/config.py`
  Add clip-query settings such as max duration, query frame count, candidate pool size, and rerank toggles.
- Create: `src/app/services/video_query_search.py`
  Hold temporary clip processing, query frame extraction, local candidate aggregation, and rerank orchestration for the new mode.
- Modify: `src/app/services/openai_vision_rerank.py`
  Extend rerank helpers to support multi-query-frame payloads instead of only single-frame text/image rerank use.
- Modify: `src/app/api/routes/search.py`
  Add a multipart upload endpoint for short clip search and wire response serialization.
- Modify: `src/app/schemas/search.py`
  Add response schema helpers if the new route needs a separate response model or mode label.
- Modify: `src/app/services/search_service.py`
  Reuse common result shaping helpers so clip search returns the same frame-card payload as text search.
- Create: `tests/unit/test_video_query_search.py`
  Cover frame extraction selection, score aggregation, and rerank candidate selection/fallback.
- Modify: `tests/integration/test_search_api.py`
  Add end-to-end API coverage for clip upload validation and successful results.
- Modify: `web/index.html`
  Replace the image-search panel markup with a video-search panel.
- Modify: `web/app.js`
  Replace image-search behavior with clip-upload search behavior and duplicate-submit blocking.
- Modify: `web/styles.css`
  Adjust sidebar styling for the new video query panel and preview box.
- Modify: `tests/unit/test_web_assets.py`
  Assert that the UI renders `Search By Video Clip` instead of image search.

### Task 1: Add Failing Backend Tests for Clip Query Pipeline

**Files:**
- Create: `tests/unit/test_video_query_search.py`
- Modify: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write failing unit tests for clip-query helper behavior**

```python
from pathlib import Path

from app.services.video_query_search import (
    aggregate_query_frame_scores,
    validate_query_clip_duration,
)


def test_validate_query_clip_duration_rejects_long_clips():
    assert validate_query_clip_duration(duration_sec=12.0, max_duration_sec=10.0) is False


def test_validate_query_clip_duration_accepts_short_clips():
    assert validate_query_clip_duration(duration_sec=8.5, max_duration_sec=10.0) is True


def test_aggregate_query_frame_scores_rewards_repeat_matches():
    per_frame_hits = [
        {11: 0.82, 18: 0.40},
        {11: 0.75, 27: 0.50},
        {11: 0.79, 18: 0.41},
    ]

    ranked = aggregate_query_frame_scores(per_frame_hits)

    assert ranked[0][0] == 11
    assert ranked[0][1] > ranked[1][1]
```

- [ ] **Step 2: Run unit tests to verify they fail**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_video_query_search.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `video_query_search`.

- [ ] **Step 3: Write failing integration tests for the new clip-search API**

```python
def test_video_query_search_rejects_too_long_clip(client, tmp_path):
    clip_path = tmp_path / "query.mp4"
    clip_path.write_bytes(b"fake-video")

    response = client.post(
        "/search/video-query",
        files={"file": ("query.mp4", clip_path.read_bytes(), "video/mp4")},
    )

    assert response.status_code in {400, 422}


def test_video_query_search_returns_results(monkeypatch, client, tmp_path):
    clip_path = tmp_path / "query.mp4"
    clip_path.write_bytes(b"fake-video")

    monkeypatch.setattr(
        "app.services.video_query_search.run_video_query_search",
        lambda db, upload: {
            "mode": "video",
            "query": "query.mp4",
            "expanded_queries": [],
            "results": [{"frame_id": 11, "score": 0.9, "thumb_url": "/media/frames/11/thumb"}],
            "parsed_query": None,
        },
    )

    response = client.post(
        "/search/video-query",
        files={"file": ("query.mp4", clip_path.read_bytes(), "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "video"
    assert response.json()["results"][0]["frame_id"] == 11
```

- [ ] **Step 4: Run the targeted API tests to verify they fail**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_search_api.py -k "video_query_search" -v`

Expected: FAIL because `/search/video-query` does not exist yet.

- [ ] **Step 5: Commit the red tests**

```bash
git add tests/unit/test_video_query_search.py tests/integration/test_search_api.py
git commit -m "test: add failing coverage for video query search"
```

### Task 2: Implement Clip Query Service and Configuration

**Files:**
- Modify: `src/app/config.py`
- Create: `src/app/services/video_query_search.py`
- Modify: `src/app/services/openai_vision_rerank.py`

- [ ] **Step 1: Add failing config assertions first**

```python
def test_video_query_search_defaults():
    values = Settings().model_dump()

    assert values["video_query_max_duration_sec"] == 10.0
    assert values["video_query_frame_count"] == 6
    assert values["video_query_local_candidate_pool"] == 24
```

- [ ] **Step 2: Run the config test to verify it fails**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_config.py -k "video_query" -v`

Expected: FAIL because the settings do not exist yet.

- [ ] **Step 3: Add the minimal config fields**

```python
video_query_max_duration_sec: float = 10.0
video_query_frame_count: int = 6
video_query_local_candidate_pool: int = 24
```

- [ ] **Step 4: Implement the clip-query service**

```python
from collections import defaultdict
from pathlib import Path


def validate_query_clip_duration(duration_sec: float, max_duration_sec: float) -> bool:
    return duration_sec <= max_duration_sec


def aggregate_query_frame_scores(per_frame_hits: list[dict[int, float]]) -> list[tuple[int, float]]:
    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)

    for hit_map in per_frame_hits:
        for frame_id, score in hit_map.items():
            totals[frame_id] += score
            counts[frame_id] += 1

    ranked = []
    for frame_id, total in totals.items():
        consistency_bonus = counts[frame_id] / max(len(per_frame_hits), 1)
        ranked.append((frame_id, total + (0.15 * consistency_bonus)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)
```

- [ ] **Step 5: Extend OpenAI vision rerank helper to accept multiple query images**

```python
def run_openai_vision_rerank(
    *,
    query_image_paths: list[Path],
    candidates: list[dict[str, object]],
    api_key: str | None,
    model: str,
    timeout_sec: int,
) -> dict[int, float]:
    ...
```

The implementation should:

- accept a list of query frame paths
- encode all query frames into the request
- keep the same JSON response contract `{frame_id: score}`
- preserve the current empty-dict fallback on error

- [ ] **Step 6: Run backend unit tests to verify they pass**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_config.py tests/unit/test_video_query_search.py tests/unit/test_openai_vision_rerank.py -v`

Expected: PASS

- [ ] **Step 7: Commit the backend service foundation**

```bash
git add src/app/config.py src/app/services/video_query_search.py src/app/services/openai_vision_rerank.py tests/unit/test_config.py tests/unit/test_video_query_search.py tests/unit/test_openai_vision_rerank.py
git commit -m "feat: add clip query retrieval service"
```

### Task 3: Add API Route and Shared Result Shaping

**Files:**
- Modify: `src/app/api/routes/search.py`
- Modify: `src/app/schemas/search.py`
- Modify: `src/app/services/search_service.py`
- Modify: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write the smallest failing route test if not already covered**

```python
def test_video_query_search_route_is_registered(client):
    response = client.options("/search/video-query")
    assert response.status_code != 404
```

- [ ] **Step 2: Run the route test to verify it fails**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_search_api.py -k "route_is_registered" -v`

Expected: FAIL with `404`.

- [ ] **Step 3: Add the FastAPI route**

```python
@router.post("/video-query", response_model=SearchResponse)
async def search_by_video_query(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> SearchResponse:
    return SearchResponse.model_validate(
        await run_video_query_search(db, file)
    )
```

- [ ] **Step 4: Reuse existing response shape helpers instead of duplicating search card assembly**

```python
def build_frame_result_payload(...):
    return {
        "frame_id": keyframe_id,
        "score": score,
        "thumb_url": thumb_url,
        "image_url": image_url,
        "preview_url": preview_url,
        ...
    }
```

Move shared shaping logic into a helper in `search_service.py` if the new clip-search flow needs the same structure.

- [ ] **Step 5: Run the integration search API tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_search_api.py -k "video_query_search" -v`

Expected: PASS

- [ ] **Step 6: Commit the API layer**

```bash
git add src/app/api/routes/search.py src/app/schemas/search.py src/app/services/search_service.py tests/integration/test_search_api.py
git commit -m "feat: add short video query search endpoint"
```

### Task 4: Implement Real Local Retrieval and Top-8 Rerank Flow

**Files:**
- Modify: `src/app/services/video_query_search.py`
- Modify: `src/app/services/openai_vision_rerank.py`
- Modify: `tests/unit/test_video_query_search.py`

- [ ] **Step 1: Add a failing unit test for local top-k reduction before rerank**

```python
def test_selects_only_top_8_candidates_for_openai_rerank():
    candidates = [
        {"frame_id": index, "score": 1.0 - (index * 0.01), "_image_path": f"/tmp/{index}.jpg"}
        for index in range(12)
    ]

    selected = select_rerank_candidates(candidates, top_k=8)

    assert len(selected) == 8
    assert selected[-1]["frame_id"] == 7
```

- [ ] **Step 2: Run the targeted unit tests to verify any remaining failures**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_video_query_search.py tests/unit/test_openai_vision_rerank.py -v`

Expected: FAIL if the real orchestration path is not wired yet.

- [ ] **Step 3: Implement the end-to-end service flow**

```python
async def run_video_query_search(db: Session, upload: UploadFile) -> dict[str, object]:
    temp_video_path = await save_upload_to_temp(upload)
    duration_sec = probe_video_duration(temp_video_path)
    if not validate_query_clip_duration(duration_sec, settings.video_query_max_duration_sec):
        raise HTTPException(status_code=400, detail="Query clip must be 10 seconds or shorter.")

    query_frame_paths = extract_query_frames(
        temp_video_path,
        frame_count=settings.video_query_frame_count,
    )
    local_results = retrieve_candidates_for_query_frames(
        db,
        query_frame_paths=query_frame_paths,
        candidate_pool=settings.video_query_local_candidate_pool,
    )
    reranked = apply_video_query_openai_rerank(query_frame_paths, local_results)
    return {
        "mode": "video",
        "query": upload.filename or "query-video",
        "expanded_queries": [],
        "results": reranked,
        "parsed_query": None,
    }
```

The implementation must also:

- clean up temporary clip and query frames in `finally`
- fall back to local results if rerank returns an empty score map
- strip any private `_image_path` field before returning

- [ ] **Step 4: Run the focused backend tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_video_query_search.py tests/unit/test_openai_vision_rerank.py tests/integration/test_search_api.py -k "video_query_search" -v`

Expected: PASS

- [ ] **Step 5: Commit the retrieval flow**

```bash
git add src/app/services/video_query_search.py src/app/services/openai_vision_rerank.py tests/unit/test_video_query_search.py tests/unit/test_openai_vision_rerank.py tests/integration/test_search_api.py
git commit -m "feat: wire local and OpenAI rerank for video query search"
```

### Task 5: Replace Image Search UI With Video Clip Search UI

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `tests/unit/test_web_assets.py`

- [ ] **Step 1: Write the failing frontend asset assertions**

```python
def test_sidebar_contains_video_query_panel():
    html = (WEB_DIR / "index.html").read_text()
    assert "Search By Video Clip" in html
    assert "Search By Image" not in html
```

- [ ] **Step 2: Run the web asset tests to verify they fail**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_web_assets.py -k "video_query" -v`

Expected: FAIL because the old image-search UI is still present.

- [ ] **Step 3: Replace the sidebar markup**

```html
<section class="panel">
  <h3>Search By Video Clip</h3>
  <input id="video-query-input" type="file" accept="video/*" />
  <button id="video-query-submit">Find Similar Frames</button>
  <video id="video-query-preview" controls muted playsinline hidden></video>
  <p class="muted">Use a short clip under 10 seconds.</p>
</section>
```

- [ ] **Step 4: Replace the client-side behavior**

```javascript
async function submitVideoQuerySearch(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/search/video-query", {
    method: "POST",
    body: formData,
  });

  return response.json();
}
```

The implementation must:

- disable the submit button while the request is in flight
- show the selected clip in the sidebar preview
- render returned results through the same result-grid renderer
- show user-facing validation errors cleanly

- [ ] **Step 5: Run the frontend asset tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_web_assets.py -v`

Expected: PASS

- [ ] **Step 6: Commit the UI replacement**

```bash
git add web/index.html web/app.js web/styles.css tests/unit/test_web_assets.py
git commit -m "feat: replace image search UI with video clip search"
```

### Task 6: Full Regression and Manual Smoke Verification

**Files:**
- No new files

- [ ] **Step 1: Run the focused backend and frontend regression suite**

Run:

```bash
PYTHONPATH=src ./.conda/bin/python -m pytest \
  tests/unit/test_config.py \
  tests/unit/test_video_query_search.py \
  tests/unit/test_openai_vision_rerank.py \
  tests/integration/test_search_api.py \
  tests/unit/test_web_assets.py -v
```

Expected: PASS

- [ ] **Step 2: Run adjacent regressions to ensure existing search and preview flows still work**

Run:

```bash
PYTHONPATH=src ./.conda/bin/python -m pytest \
  tests/integration/test_video_api.py \
  tests/unit/test_media_preview.py \
  tests/unit/test_search_service.py -v
```

Expected: PASS

- [ ] **Step 3: Manual smoke test the running app**

Checklist:

- upload a short query clip from the sidebar
- confirm the button locks during search
- confirm results render in the existing grid
- click a result and verify timeline preview still updates
- confirm invalid long clips show a clear error
- confirm text search still works after the UI change

- [ ] **Step 4: Commit final cleanup if any test-driven touchups were needed**

```bash
git add src/app config.py src/app/services src/app/api/routes/search.py web tests
git commit -m "test: verify short video query search regression coverage"
```

## Self-Review

- Spec coverage: the plan covers UI replacement, dedicated upload endpoint, short-clip validation, multi-frame local retrieval, top-8 OpenAI rerank, fallback behavior, and regression testing.
- Placeholder scan: no `TODO`/`TBD` markers remain; each task has concrete file targets and concrete commands.
- Type consistency: the plan consistently uses `run_video_query_search`, `aggregate_query_frame_scores`, and `query_image_paths` naming across tasks.
