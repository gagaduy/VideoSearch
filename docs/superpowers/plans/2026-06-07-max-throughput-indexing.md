# Max Throughput Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase indexing throughput on a single GPU by adding stage timing, batching `OpenCLIP` frame embedding, and overlapping CPU preparation without intentionally degrading retrieval quality.

**Architecture:** Keep one worker process and the current model stack, but change how work is fed into the models. Instrument the pipeline first so bottlenecks are measurable, then batch `OpenCLIP` image embedding and add bounded CPU prefetch/overlap for enrich-related image preparation.

**Tech Stack:** Python, FastAPI worker runtime, SQLAlchemy, PyTorch, OpenCLIP, pytest

---

## File Map

- Modify: `src/app/config.py`
  - Add throughput-oriented settings with safe defaults.
- Modify: `src/worker/adapters/openclip_adapter.py`
  - Add batched image embedding support while preserving existing single-image API.
- Modify: `src/worker/pipeline.py`
  - Add stage timing, switch frame embedding to batching, and add bounded prefetch helpers.
- Modify: `tests/unit/test_config.py`
  - Cover new throughput settings defaults.
- Modify: `tests/unit/test_openclip_adapter.py`
  - Cover batch embedding output shape and model metadata behavior.
- Modify: `tests/unit/test_pipeline_vectors.py`
  - Cover helper logic for batching, timing payloads, and prefetch boundaries.
- Modify: `tests/integration/test_real_pipeline.py`
  - Cover that frame embedding path uses batched `OpenCLIP` calls.

### Task 1: Add Throughput Settings

**Files:**
- Modify: `tests/unit/test_config.py`
- Modify: `src/app/config.py`

- [ ] **Step 1: Write the failing config test**

```python
def test_settings_expose_throughput_defaults(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path / "data",
        videos_dir=tmp_path / "data" / "videos",
        frames_dir=tmp_path / "data" / "frames",
        thumbs_dir=tmp_path / "data" / "thumbs",
    )

    assert settings.openclip_batch_size == 16
    assert settings.enrich_prefetch_workers == 2
    assert settings.enable_stage_timing is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL because the new settings do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class Settings(BaseSettings):
    openclip_batch_size: int = 16
    enrich_prefetch_workers: int = 2
    enable_stage_timing: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/config.py tests/unit/test_config.py
git commit -m "feat: add throughput tuning settings"
```

### Task 2: Add Batched OpenCLIP Image Embedding

**Files:**
- Modify: `tests/unit/test_openclip_adapter.py`
- Modify: `src/worker/adapters/openclip_adapter.py`

- [ ] **Step 1: Write the failing adapter test**

```python
def test_openclip_adapter_can_embed_images_in_batch(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    adapter = OpenClipAdapter(model_name="ViT-B-32")
    results = adapter.embed_images([str(image_a), str(image_b)])

    assert len(results) == 2
    assert all(result.model_name == "ViT-B-32" for result in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_openclip_adapter.py -v`
Expected: FAIL because `embed_images` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def embed_images(self, image_paths: list[str]) -> list[EmbeddingResult]:
    if not image_paths:
        return []
    if self._lazy_load():
        import torch
        from PIL import Image

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensors.append(self._preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(self._device)
        with torch.inference_mode():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return [
            EmbeddingResult(model_name=self.model_name, values=row.detach().cpu().tolist())
            for row in features
        ]
    return [
        EmbeddingResult(model_name=self.model_name, values=self._fallback_values(Path(image_path).read_bytes()))
        for image_path in image_paths
    ]
```

Make `embed_image(...)` call `embed_images([image_path])[0]` to keep one code path.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_openclip_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/adapters/openclip_adapter.py tests/unit/test_openclip_adapter.py
git commit -m "feat: add batched openclip image embedding"
```

### Task 3: Add Stage Timing Helpers

**Files:**
- Modify: `tests/unit/test_pipeline_vectors.py`
- Modify: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing timing helper test**

```python
def test_merge_stage_timings_accumulates_elapsed_time() -> None:
    timings = {}
    pipeline._record_stage_timing(timings, "ocr", 0.25)
    pipeline._record_stage_timing(timings, "ocr", 0.5)

    assert timings["ocr"]["count"] == 2
    assert timings["ocr"]["total_sec"] == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "stage_timing" -v`
Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def _record_stage_timing(timings: dict[str, dict[str, float | int]], stage: str, elapsed_sec: float) -> None:
    bucket = timings.setdefault(stage, {"count": 0, "total_sec": 0.0})
    bucket["count"] = int(bucket["count"]) + 1
    bucket["total_sec"] = round(float(bucket["total_sec"]) + elapsed_sec, 6)
```

Store the timing payload in the indexing return data and optionally in `raw_json` when `settings.enable_stage_timing` is true.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "stage_timing" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/unit/test_pipeline_vectors.py
git commit -m "feat: add stage timing instrumentation"
```

### Task 4: Batch Frame Embedding In The Pipeline

**Files:**
- Modify: `tests/integration/test_real_pipeline.py`
- Modify: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing integration test**

```python
def test_index_prepared_frames_batches_openclip_image_embedding(monkeypatch, tmp_path: Path) -> None:
    # Stub OpenClipAdapter.embed_images to count calls over a multi-frame input.
    # Assert the pipeline uses fewer batch calls than frame count.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_real_pipeline.py -k "batches_openclip" -v`
Expected: FAIL because frame embedding still loops through `embed_image(...)` one frame at a time.

- [ ] **Step 3: Write minimal implementation**

```python
frame_batch_size = max(settings.openclip_batch_size, 1)
for start_index in range(0, len(frame_sources), frame_batch_size):
    batch_sources = frame_sources[start_index:start_index + frame_batch_size]
    embeddings = openclip.embed_images([str(item["image_path"]) for item in batch_sources])
    for frame_source, embedding in zip(batch_sources, embeddings, strict=True):
        ...
```

Keep DB row creation logic the same, but move from one-image calls to batch slices.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_real_pipeline.py -k "batches_openclip" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/integration/test_real_pipeline.py
git commit -m "feat: batch frame embeddings in pipeline"
```

### Task 5: Add Bounded CPU Prefetch For Enrich Inputs

**Files:**
- Modify: `tests/unit/test_pipeline_vectors.py`
- Modify: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing prefetch test**

```python
def test_prepare_enrich_inputs_returns_segment_image_payloads() -> None:
    payloads = pipeline._prepare_enrich_inputs(
        segments=[segment_stub],
        db=db_stub,
        prefetch_workers=2,
    )
    assert payloads[0]["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "prepare_enrich_inputs" -v`
Expected: FAIL because the helper does not exist.

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

Use this helper at the start of `_enrich_segment_keyframes(...)` so image lookup is done before the heavy model calls. Keep the first pass simple and bounded rather than adding aggressive parallel futures immediately.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline_vectors.py -k "prepare_enrich_inputs" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/unit/test_pipeline_vectors.py
git commit -m "refactor: precompute enrich input payloads"
```

### Task 6: Final Verification

**Files:**
- Verify only

- [ ] **Step 1: Run focused throughput tests**

Run:

```bash
pytest \
  tests/unit/test_config.py \
  tests/unit/test_openclip_adapter.py \
  tests/unit/test_pipeline_vectors.py \
  tests/integration/test_real_pipeline.py -v
```

Expected: PASS

- [ ] **Step 2: Run adjacent regression tests**

Run:

```bash
pytest \
  tests/unit/test_worker_tasks.py \
  tests/unit/test_worker_pipeline_io.py \
  tests/unit/test_openclip_adapter.py \
  tests/integration/test_search_api.py -v
```

Expected: PASS

- [ ] **Step 3: Sanity-check runtime OpenCLIP dimensions**

Run:

```bash
PYTHONPATH=src ./.conda/bin/python - <<'PY'
from worker.adapters.openclip_adapter import OpenClipAdapter
adapter = OpenClipAdapter()
result = adapter.embed_text("sanity check")
print(len(result.values))
PY
```

Expected: `1024`

- [ ] **Step 4: Inspect git diff**

Run: `git status --short`
Expected: only intended files changed.

## Self-Review

- Spec coverage:
  - stage timing is covered by Task 3
  - `OpenCLIP` batching is covered by Tasks 2 and 4
  - bounded overlap starts with enrich input precomputation in Task 5
  - no multi-worker GPU fan-out is introduced
- Placeholder scan:
  - all steps include files, commands, and concrete code direction
- Type consistency:
  - `embed_images`, `_record_stage_timing`, and `_prepare_enrich_inputs` are referenced consistently across tasks
