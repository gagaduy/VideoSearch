# Enrich Throughput Without Quality Tradeoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce enrich-stage wall-clock time without intentionally lowering quality by adding enrich sub-stage timing, precomputing enrich inputs, and introducing bounded overlap around heavy model work.

**Architecture:** Keep one worker and one heavy `InternVL` lane, but move non-essential lookup and preparation work out of the hottest loop. Add timing around OCR, detector, `InternVL`, branch-B embedding, and DB persistence so throughput work is guided by measurements rather than guesswork.

**Tech Stack:** Python, SQLAlchemy, PyTorch, InternVL, OpenCLIP, pytest

---

## File Map

- Modify: `src/app/config.py`
  - Add enrich throughput settings if missing or incomplete.
- Modify: `src/worker/pipeline.py`
  - Add enrich timing, input precomputation, and bounded overlap helpers.
- Modify: `tests/unit/test_config.py`
  - Cover enrich throughput settings defaults.
- Modify: `tests/unit/test_pipeline_vectors.py`
  - Cover enrich timing helpers and input-preparation helpers.
- Modify: `tests/integration/test_real_pipeline.py`
  - Cover enrich metadata and verify behavior remains semantically intact.

### Task 1: Add Enrich Throughput Settings

**Files:**
- Modify: `tests/unit/test_config.py`
- Modify: `src/app/config.py`

- [ ] **Step 1: Write the failing config test**

```python
def test_settings_expose_enrich_throughput_defaults(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path / "data",
        videos_dir=tmp_path / "data" / "videos",
        frames_dir=tmp_path / "data" / "frames",
        thumbs_dir=tmp_path / "data" / "thumbs",
    )

    assert settings.enrich_prefetch_workers == 2
    assert settings.enable_stage_timing is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL if the settings are not fully present in config or tests.

- [ ] **Step 3: Write minimal implementation**

```python
class Settings(BaseSettings):
    enrich_prefetch_workers: int = 2
    enable_stage_timing: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/config.py tests/unit/test_config.py
git commit -m "feat: add enrich throughput settings"
```

### Task 2: Add Enrich Timing Helpers

**Files:**
- Modify: `tests/unit/test_pipeline_vectors.py`
- Modify: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing timing tests**

```python
def test_record_stage_timing_accumulates_elapsed_time() -> None:
    timings = {}
    pipeline._record_stage_timing(timings, "internvl", 0.5)
    pipeline._record_stage_timing(timings, "internvl", 0.75)

    assert timings["internvl"]["count"] == 2
    assert timings["internvl"]["total_sec"] == 1.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "stage_timing" -v`
Expected: FAIL because the helper does not exist or does not cover enrich usage.

- [ ] **Step 3: Write minimal implementation**

```python
def _record_stage_timing(timings: dict[str, dict[str, float | int]], stage: str, elapsed_sec: float) -> None:
    bucket = timings.setdefault(stage, {"count": 0, "total_sec": 0.0})
    bucket["count"] = int(bucket["count"]) + 1
    bucket["total_sec"] = round(float(bucket["total_sec"]) + elapsed_sec, 6)
```

Add timing calls around:

- OCR
- detector
- `InternVL`
- branch-B text embedding
- enrich persistence/commit

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "stage_timing" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/unit/test_pipeline_vectors.py
git commit -m "feat: instrument enrich stage timings"
```

### Task 3: Precompute Enrich Input Payloads

**Files:**
- Modify: `tests/unit/test_pipeline_vectors.py`
- Modify: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing payload-preparation test**

```python
def test_prepare_enrich_inputs_returns_segment_image_payloads(tmp_path) -> None:
    payloads = pipeline._prepare_enrich_inputs(db_stub, [segment_stub])
    assert payloads[0]["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "prepare_enrich_inputs" -v`
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def _prepare_enrich_inputs(db: Session, segments: list[Segment]) -> list[dict[str, object]]:
    payloads = []
    for segment in segments:
        frame = db.get(Frame, int(segment.keyframe_id))
        if frame is None:
            continue
        payloads.append(
            {
                "segment": segment,
                "frame": frame,
                "image_path": str(frame.image_path),
            }
        )
    return payloads
```

Switch `_enrich_segment_keyframes(...)` to iterate over prepared payloads instead of resolving `Frame` rows in the hot loop.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "prepare_enrich_inputs" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/unit/test_pipeline_vectors.py
git commit -m "refactor: precompute enrich input payloads"
```

### Task 4: Persist Enrich Timing Metadata Without Changing Semantics

**Files:**
- Modify: `tests/integration/test_real_pipeline.py`
- Modify: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing integration test**

```python
def test_run_index_pipeline_returns_enrich_stage_timings(monkeypatch, tmp_path: Path) -> None:
    payload = pipeline.run_index_pipeline(session, int(video.id))
    assert "stage_timings" in payload
    assert "ocr" in payload["stage_timings"] or "internvl" in payload["stage_timings"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_real_pipeline.py -k "stage_timings" -v`
Expected: FAIL because timing payloads are not exposed yet.

- [ ] **Step 3: Write minimal implementation**

```python
return {
    "frame_count": len(frame_sources),
    "segment_count": len(segments),
    ...
    "stage_timings": stage_timings,
}
```

When `settings.enable_stage_timing` is true, also attach timing metadata to segment `raw_json` or other debug-friendly payloads without altering search semantics.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_real_pipeline.py -k "stage_timings" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/integration/test_real_pipeline.py
git commit -m "feat: expose enrich stage timing metadata"
```

### Task 5: Add Bounded Overlap For Safe Enrich Preparation

**Files:**
- Modify: `tests/unit/test_pipeline_vectors.py`
- Modify: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing helper test**

```python
def test_prepare_enrich_inputs_preserves_segment_order() -> None:
    payloads = pipeline._prepare_enrich_inputs(db_stub, [segment_a, segment_b])
    assert [payload["segment"] for payload in payloads] == [segment_a, segment_b]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "preserves_segment_order" -v`
Expected: FAIL if helper behavior is not yet covered or stable for overlap usage.

- [ ] **Step 3: Write minimal implementation**

```python
def _prepare_enrich_inputs(db: Session, segments: list[Segment]) -> list[dict[str, object]]:
    # Keep order stable so downstream semantics do not change.
    ...
```

If adding a tiny thread pool for image-path preparation or existence checks, keep the final payload order identical to segment order and gate it behind `settings.enrich_prefetch_workers`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "preserves_segment_order" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/unit/test_pipeline_vectors.py
git commit -m "feat: add bounded enrich input overlap"
```

### Task 6: Final Verification

**Files:**
- Verify only

- [ ] **Step 1: Run focused enrich throughput tests**

Run:

```bash
pytest \
  tests/unit/test_config.py \
  tests/unit/test_pipeline_vectors.py \
  tests/integration/test_real_pipeline.py -v
```

Expected: PASS

- [ ] **Step 2: Run adjacent regression tests**

Run:

```bash
pytest \
  tests/unit/test_openclip_adapter.py \
  tests/integration/test_search_api.py \
  tests/unit/test_worker_pipeline_io.py -v
```

Expected: PASS

- [ ] **Step 3: Inspect git diff**

Run: `git status --short`
Expected: only intended files changed.

## Self-Review

- Spec coverage:
  - enrich timing is covered by Tasks 2 and 4
  - enrich input prefetch is covered by Tasks 3 and 5
  - no model downgrade or enrich-stage removal is introduced
- Placeholder scan:
  - every task contains concrete files, commands, and target behavior
- Type consistency:
  - `_record_stage_timing` and `_prepare_enrich_inputs` are referenced consistently across the plan
