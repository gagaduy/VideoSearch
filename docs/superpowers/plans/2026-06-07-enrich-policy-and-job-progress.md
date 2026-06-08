# Enrich Policy And Job Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `balanced` from enrich behavior, make VLM failure semantics honest, fix the OCR score bug and misleading score presentation, and upgrade the ingest progress panel so indexing state is visible from the UI.

**Architecture:** Keep the current worker, API, and web polling boundaries intact. Backend changes stay focused in the enrich gating and search score assembly, while the frontend parses the existing `job.stage` strings into clearer progress metadata and rendering.

**Tech Stack:** Python, FastAPI, SQLAlchemy, vanilla JavaScript, HTML, CSS, pytest

---

## File Map

- Modify: `src/worker/pipeline.py`
  - Owns enrich gating, fallback behavior, and job stage updates.
- Modify: `src/app/config.py`
  - Owns default profile documentation and accepted profile semantics.
- Modify: `src/app/services/search_service.py`
  - Owns rerank input assembly and final result payload.
- Modify: `web/app.js`
  - Owns job polling, progress parsing, score display text, and status rendering.
- Modify: `web/index.html`
  - Owns markup for the richer job progress panel.
- Modify: `web/styles.css`
  - Owns layout and styling for the new progress details.
- Modify: `tests/unit/test_pipeline_vectors.py`
  - Covers enrich gating and fallback helpers.
- Modify: `tests/integration/test_real_pipeline.py`
  - Covers VLM scheduling and failure behavior.
- Modify: `tests/integration/test_video_api.py`
  - Covers presence of the richer progress panel markup.
- Create or modify: `tests/unit/test_search_service.py`
  - Covers OCR score sourcing and score presentation helper behavior.

### Task 1: Remove `balanced` From Enrich Gating

**Files:**
- Modify: `tests/unit/test_pipeline_vectors.py`
- Modify: `src/worker/pipeline.py`
- Modify: `src/app/config.py`

- [ ] **Step 1: Write the failing gating tests**

```python
def test_should_run_vlm_enrichment_in_local_profile_uses_sparse_schedule() -> None:
    assert not pipeline._should_run_vlm_enrichment(
        segment_index=2,
        segment_count=10,
        profile="local",
        sparse_stride=6,
        ocr_text="street sign",
        objects=[{"label": "car"}],
    )


def test_should_run_vlm_enrichment_in_local_profile_when_supporting_evidence_is_missing() -> None:
    assert pipeline._should_run_vlm_enrichment(
        segment_index=2,
        segment_count=5,
        profile="local",
        sparse_stride=3,
        ocr_text="",
        objects=[],
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "vlm_enrichment" -v`
Expected: FAIL because old `balanced` tests still exist and current semantics are mixed.

- [ ] **Step 3: Write minimal implementation**

```python
def _should_run_vlm_enrichment(
    *,
    segment_index: int,
    segment_count: int,
    profile: str,
    sparse_stride: int,
    ocr_text: str,
    objects: list[dict[str, object]],
) -> bool:
    normalized_profile = profile.strip().lower()
    if normalized_profile == "full":
        return True
    if not ocr_text.strip() or not objects:
        return True
    if segment_index == 1 or segment_index == segment_count:
        return True
    stride = max(sparse_stride, 1)
    return ((segment_index - 1) % stride) == 0
```

- [ ] **Step 4: Remove `balanced` references from config and tests**

```python
class Settings(BaseSettings):
    indexing_profile: str = "local"
```

Replace any `balanced`-named test with `local` or `full` semantics and remove assertions that depend on `"balanced-heuristic"` labels.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "vlm_enrichment" -v`
Expected: PASS with no `balanced` coverage left.

- [ ] **Step 6: Commit**

```bash
git add src/worker/pipeline.py src/app/config.py tests/unit/test_pipeline_vectors.py tests/integration/test_real_pipeline.py
git commit -m "refactor: remove balanced enrich profile"
```

### Task 2: Make VLM Failure Honest

**Files:**
- Modify: `tests/integration/test_real_pipeline.py`
- Modify: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing VLM failure test**

```python
def test_run_index_pipeline_leaves_semantic_fields_empty_when_vlm_selected_but_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Arrange a video where local policy selects VLM for the first segment.
    # Stub InternVL to raise, and make caption/entity adapters explode if called.
    # Assert caption_text == "" and semantic_counts_json == {} while OCR/object data remain populated.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_real_pipeline.py -k "vlm_selected_but_fails" -v`
Expected: FAIL because the current code falls back to caption and semantic extraction.

- [ ] **Step 3: Write minimal implementation**

```python
if use_vlm:
    try:
        branch_b = branch_b_adapter.describe_image(image_path)
    except Exception as exc:
        branch_b = {"caption": "", "tags": [], "entities": [], "model_name": "error"}
        stage_failures["branch_b"] = str(exc)

vlm_failed = use_vlm and "branch_b" in stage_failures

if branch_b.get("caption"):
    caption = {...}
elif use_vlm and vlm_failed:
    caption = {"caption": "", "model_name": "error", "confidence": 0.0}
elif use_vlm:
    caption = captioner.caption(image_path)
else:
    caption = {"caption": _build_lightweight_caption(objects, str(ocr["text"])), "model_name": "local-heuristic", "confidence": 0.35}
```

Apply the same rule to semantic extraction: when `vlm_failed` is true, return empty semantic data instead of calling the fallback semantic adapter.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_real_pipeline.py -k "vlm_selected_but_fails" -v`
Expected: PASS with empty VLM-driven fields and preserved OCR/object evidence.

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/integration/test_real_pipeline.py
git commit -m "fix: avoid low-quality fallback after vlm failure"
```

### Task 3: Fix OCR Score Sourcing And Clarify Result Score Presentation

**Files:**
- Create: `tests/unit/test_search_service.py`
- Modify: `src/app/services/search_service.py`
- Modify: `web/app.js`

- [ ] **Step 1: Write the failing score assembly test**

```python
def test_build_result_uses_ocr_text_for_ocr_score() -> None:
    item = {
        "caption": "man beside car",
        "ocr_text": "speed zone sign",
    }
    score = search_service._lexical_score(["speed", "zone"], item["ocr_text"])
    assert score == 1.0
```

Add a focused test around the result assembly helper or extract one if needed so the assertion fails against the current `caption`-based OCR scoring path.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_search_service.py -v`
Expected: FAIL because the current result assembly uses `caption` for `ocr_score`.

- [ ] **Step 3: Write minimal implementation**

```python
"ocr_score": _lexical_score(
    text_terms,
    str(rows_by_segment[segment_id].get("ocr_text", "")),
) if rows_by_segment[segment_id].get("ocr_text") else 0.0,
```

In the frontend, format result score as a plain retrieval score instead of an implied percentage:

```javascript
function formatRetrievalScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  return numeric.toFixed(3);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_search_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/services/search_service.py web/app.js tests/unit/test_search_service.py
git commit -m "fix: score ocr from ocr text"
```

### Task 4: Upgrade Job Progress Panel Markup And Parsing

**Files:**
- Modify: `tests/integration/test_video_api.py`
- Modify: `web/index.html`
- Modify: `web/app.js`

- [ ] **Step 1: Write the failing web markup test**

```python
def test_web_index_contains_detailed_job_progress_fields() -> None:
    contents = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="job-progress-percent"' in contents
    assert 'id="job-progress-count"' in contents
    assert 'id="job-status-note"' in contents
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_video_api.py -k "job_progress" -v`
Expected: FAIL because the new progress fields do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```html
<div class="job-status-row">
  <strong id="job-status-label">Waiting</strong>
  <span id="job-stage-label">queued</span>
</div>
<div class="job-progress-meta">
  <span id="job-progress-count">0/0</span>
  <span id="job-progress-percent">0%</span>
</div>
<div class="progress-track" aria-hidden="true">
  <div id="job-progress-bar" class="progress-bar"></div>
</div>
<p id="job-status-note" class="job-status-note">Waiting for a job.</p>
```

Add a parser in `web/app.js`:

```javascript
function parseJobProgress(job) {
  const stage = String(job?.stage || "queued");
  const match = stage.match(/^(embedding_frames|enriching_segments|indexing):(\\d+)\\/(\\d+)$/);
  if (!match) {
    return { stage, current: null, total: null, percent: progressForJob(job) };
  }
  return {
    stage: match[1],
    current: Number.parseInt(match[2], 10),
    total: Number.parseInt(match[3], 10),
    percent: progressForJob(job),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_video_api.py -k "job_progress" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/index.html web/app.js tests/integration/test_video_api.py
git commit -m "feat: add detailed job progress fields"
```

### Task 5: Improve Progress Rendering And Messaging

**Files:**
- Modify: `web/styles.css`
- Modify: `web/app.js`

- [ ] **Step 1: Write the failing frontend behavior test or targeted assertion**

If no browser test harness exists, add a narrow unit-style helper test for progress note selection in `tests/unit/test_search_service.py` replacement or create `tests/unit/test_web_progress.py` if the project already supports JS-independent helper testing through extracted pure functions.

```python
def test_progress_note_for_enriching_segments() -> None:
    note = describe_job_note({"status": "running", "stage": "enriching_segments:8/24"})
    assert "InternVL" in note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_web_progress.py -v`
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```javascript
function describeJobNote(job) {
  const stage = String(job?.stage || "queued");
  if (job?.status === "failed" || stage === "error") {
    return "Indexing failed. Check the latest job state and worker logs.";
  }
  if (stage.startsWith("embedding_frames:")) {
    return "Worker is embedding extracted frames for segment building.";
  }
  if (stage === "building_segments") {
    return "Worker is grouping nearby frames into retrieval segments.";
  }
  if (stage.startsWith("enriching_segments:")) {
    return "Worker is enriching segment keyframes with InternVL, OCR, and object detection.";
  }
  return "Worker is preparing the indexing job.";
}
```

Apply matching style rules in `web/styles.css` for `.job-progress-meta` and `.job-status-note`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_web_progress.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/styles.css web/app.js tests/unit/test_web_progress.py
git commit -m "feat: improve job progress messaging"
```

### Task 6: Final Verification

**Files:**
- Verify only

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
pytest \
  tests/unit/test_pipeline_vectors.py \
  tests/integration/test_real_pipeline.py \
  tests/integration/test_video_api.py \
  tests/unit/test_search_service.py \
  tests/unit/test_web_progress.py -v
```

Expected: PASS

- [ ] **Step 2: Run broader related search and pipeline checks**

Run:

```bash
pytest \
  tests/integration/test_search_api.py \
  tests/unit/test_local_rerank.py \
  tests/unit/test_retrieval_fusion.py -v
```

Expected: PASS

- [ ] **Step 3: Inspect git diff**

Run: `git status --short`
Expected: only intended files changed.

- [ ] **Step 4: Commit final polish if needed**

```bash
git add src/app/services/search_service.py web/index.html web/app.js web/styles.css tests
git commit -m "feat: improve indexing progress visibility"
```

## Self-Review

- Spec coverage:
  - `balanced` removal is covered by Task 1.
  - honest VLM failure behavior is covered by Task 2.
  - OCR scoring fix and score presentation are covered by Task 3.
  - progress UI visibility is covered by Tasks 4 and 5.
- Placeholder scan:
  - all tasks name exact files, commands, and target behaviors.
- Type consistency:
  - `parseJobProgress`, `describeJobNote`, and the existing Python helper names are used consistently across tasks.
