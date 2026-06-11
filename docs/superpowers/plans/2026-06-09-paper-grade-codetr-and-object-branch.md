# Paper-Grade Co-DETR And Object Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Co-DETR for real in the offline indexing pipeline and tune the retrieval object branch so Co-DETR labels, counts, and positions improve object-centric search quality without adding online detector latency.

**Architecture:** Keep Co-DETR strictly offline inside the enrich/indexing worker, persist richer object evidence into the existing segment metadata, and then refine the existing local object branch to score label/count/region matches more like the paper’s object filtering path. Search continues to read indexed metadata only; no new online detector pass is introduced.

**Tech Stack:** Python 3.11, FastAPI worker pipeline, SQLAlchemy, PostgreSQL, PyTorch, MMDetection/MMEngine/MMCV, Co-DETR runtime, pytest.

---

### Task 1: Add MMDetection/Co-DETR Runtime Dependencies And Smoke Probe

**Files:**
- Modify: `requirements.txt` or the active dependency manifest used by `.conda`
- Create: `tests/unit/test_codetr_runtime_probe.py`
- Reference: `src/worker/adapters/codetr_adapter.py`

- [ ] **Step 1: Write the failing runtime probe test**

```python
from worker.adapters import codetr_adapter


def test_codetr_runtime_probe_reports_missing_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(codetr_adapter, "init_detector", None)
    adapter = codetr_adapter.CoDetrDetectionAdapter(
        config_path="cfg.py",
        checkpoint_path="model.pth",
    )

    try:
        adapter._lazy_load()
    except RuntimeError as exc:
        assert "Co-DETR dependencies are unavailable" in str(exc)
    else:
        raise AssertionError("expected runtime dependency error")
```

- [ ] **Step 2: Run the probe test**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_codetr_runtime_probe.py -v`

Expected: PASS now, confirming the adapter fails cleanly when the runtime is absent.

- [ ] **Step 3: Install the runtime dependencies**

Run the exact install commands for the local env being used by the project:

```bash
./.conda/bin/pip install -U openmim
./.conda/bin/mim install mmengine
./.conda/bin/mim install "mmcv>=2.0.0"
./.conda/bin/mim install mmdet
```

If Co-DETR requires a specific upstream package or config checkout, add it under the repo’s expected model path (for example `models/codetr/`) instead of scattering files around the workspace.

- [ ] **Step 4: Verify imports in the real env**

Run:

```bash
PYTHONPATH=src ./.conda/bin/python - <<'PY'
for name in ["mmdet", "mmcv", "mmengine"]:
    __import__(name)
    print(name, "OK")
PY
```

Expected: all three print `OK`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/unit/test_codetr_runtime_probe.py
git commit -m "build: add Co-DETR runtime dependencies"
```

### Task 2: Add Real Co-DETR Config And Checkpoint Wiring

**Files:**
- Modify: `src/app/config.py`
- Create: `models/codetr/README.md`
- Optionally create/update: `.env.example`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing config-path test**

```python
from app.config import Settings


def test_codetr_settings_support_real_model_paths() -> None:
    settings = Settings(
        object_detector_family="codetr",
        codetr_config_path="models/codetr/co_dino_5scale_r50_1x_coco.py",
        codetr_checkpoint_path="models/codetr/co_dino_5scale_r50_1x_coco.pth",
        codetr_device="cuda:0",
    )
    assert settings.codetr_config_path.endswith(".py")
    assert settings.codetr_checkpoint_path.endswith(".pth")
```

- [ ] **Step 2: Run the config test**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_config.py::test_codetr_settings_support_real_model_paths -v`

Expected: FAIL until the config paths and docs are aligned with a real runtime location.

- [ ] **Step 3: Align config defaults and model directory docs**

Use concrete defaults in `src/app/config.py`:

```python
codetr_config_path: str = "models/codetr/co_dino_5scale_r50_1x_coco.py"
codetr_checkpoint_path: str = "models/codetr/co_dino_5scale_r50_1x_coco.pth"
```

Document in `models/codetr/README.md`:
- which config/checkpoint pair is expected
- where they come from
- how to switch variants later without changing retrieval code

- [ ] **Step 4: Run config tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/app/config.py models/codetr/README.md tests/unit/test_config.py .env.example
git commit -m "docs: define real Co-DETR model path defaults"
```

### Task 3: Validate Co-DETR Offline Enrich On A Real Frame Slice

**Files:**
- Modify: `tests/integration/test_real_pipeline.py`
- Reference: `src/worker/pipeline.py`

- [ ] **Step 1: Write the failing integration test for a live Co-DETR slice**

```python
def test_codetr_detector_path_runs_when_runtime_is_available(monkeypatch, tmp_path: Path) -> None:
    from worker.adapters import codetr_adapter

    if codetr_adapter.init_detector is None:
        pytest.skip("Co-DETR runtime is not installed")

    monkeypatch.setattr(pipeline.settings, "object_detector_family", "codetr")
    ...
    payload = pipeline.run_index_pipeline(session, int(video.id))
    segment = session.query(Segment).filter(Segment.video_id == video.id).one()
    assert segment.raw_json["object_detector_family"] == "codetr"
    assert "detector" in payload["stage_timings"]
```

- [ ] **Step 2: Run the integration test**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_real_pipeline.py::test_codetr_detector_path_runs_when_runtime_is_available -v`

Expected: either SKIP before install, or FAIL if runtime is present but the live path is misconfigured.

- [ ] **Step 3: Fix any live-path issues**

Typical fixes if this fails:
- wrong config/checkpoint path
- device mismatch (`cuda:0` vs `cpu`)
- unexpected MMDetection result format in `CoDetrDetectionAdapter.detect`

Make only the minimum code changes needed to make the live path work. Keep the normalized output contract unchanged.

- [ ] **Step 4: Re-run the live integration slice**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_real_pipeline.py::test_codetr_detector_path_runs_when_runtime_is_available -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worker/adapters/codetr_adapter.py tests/integration/test_real_pipeline.py
git commit -m "test: validate live Co-DETR worker path"
```

### Task 4: Enrich Offline Object Metadata Beyond Raw Labels

**Files:**
- Modify: `src/worker/pipeline.py`
- Modify: `src/worker/retrieval_ontology.py`
- Test: `tests/unit/test_pipeline_vectors.py`
- Test: `tests/integration/test_real_pipeline.py`

- [ ] **Step 1: Write the failing metadata tests**

```python
def test_object_positions_and_counts_persist_for_codetr_objects() -> None:
    objects = [
        {"label": "car", "bbox": [0, 0, 40, 40], "score": 0.9},
        {"label": "person", "bbox": [60, 10, 90, 90], "score": 0.8},
    ]
    positions = pipeline._object_positions(objects, "tests/fixtures/frame.png")
    counts = pipeline._object_counts(objects)
    assert counts == {"car": 1, "person": 1}
    assert "left" in positions["car"] or "center" in positions["car"]
```

```python
def test_canonicalize_object_label_maps_codetr_closed_set_labels() -> None:
    assert canonicalize_object_label("automobile") == "car"
    assert canonicalize_object_label("people") == "person"
```

- [ ] **Step 2: Run the metadata tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_pipeline_vectors.py -k object tests/integration/test_real_pipeline.py -k detector_family -v`

Expected: FAIL if Co-DETR outputs expose edge cases the current metadata shaping does not normalize cleanly.

- [ ] **Step 3: Tighten object metadata shaping**

Keep the existing schema, but ensure offline metadata is richer and deterministic:

```python
segment.object_labels_json = sorted({label for label in counts})
segment.object_counts_json = counts
segment.object_positions_json = object_positions
segment.raw_json["object_detector_family"] = ...
segment.raw_json["object_detector_model"] = ...
```

In `src/worker/retrieval_ontology.py`, extend canonicalization only where Co-DETR closed-set labels require it; do not expand the ontology gratuitously.

- [ ] **Step 4: Run metadata tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_pipeline_vectors.py tests/integration/test_real_pipeline.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py src/worker/retrieval_ontology.py tests/unit/test_pipeline_vectors.py tests/integration/test_real_pipeline.py
git commit -m "feat: enrich Co-DETR object metadata for retrieval"
```

### Task 5: Tune The Retrieval Object Branch For Count And Region Constraints

**Files:**
- Modify: `src/app/db/repositories/branch_search.py`
- Modify: `src/app/services/query_understanding.py`
- Modify: `src/app/services/search_service.py`
- Test: `tests/integration/test_vector_search.py`
- Test: `tests/unit/test_search_service.py`

- [ ] **Step 1: Write the failing object-branch retrieval tests**

```python
def test_search_object_branch_rewards_matching_min_count() -> None:
    rows = [
        _segment(counts={"person": 3}, positions={"person": ["left"]}),
        _segment(counts={"person": 1}, positions={"person": ["left"]}),
    ]
    filters = [ObjectFilter(label="person", min_count=3)]
    ranked = search_object_branch(db_stub(rows), filters)
    assert ranked[0]["object_score"] > ranked[1]["object_score"]
```

```python
def test_search_object_branch_penalizes_region_mismatch() -> None:
    rows = [
        _segment(counts={"car": 1}, positions={"car": ["left"]}),
        _segment(counts={"car": 1}, positions={"car": ["right"]}),
    ]
    filters = [ObjectFilter(label="car", min_count=1, regions=["left"])]
    ranked = search_object_branch(db_stub(rows), filters)
    assert ranked[0]["object_positions"]["car"] == ["left"]
```

- [ ] **Step 2: Run the failing retrieval tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_vector_search.py -k object tests/unit/test_search_service.py -v`

Expected: FAIL because the current object branch mostly treats passing filters as binary plus a shallow count ratio.

- [ ] **Step 3: Implement minimal scoring improvements**

Keep it local-only and cheap:

```python
count_ratio = min(count / max(item.min_count, 1), 1.5)
region_bonus = 0.25 if not item.regions or any(region in positions for region in item.regions) else -0.5
score += max(count_ratio + region_bonus, 0.0)
```

Also ensure `query_understanding._extract_object_filters` keeps count and region hints for object-centric phrases such as `three people on the left`.

- [ ] **Step 4: Run retrieval tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_vector_search.py tests/unit/test_search_service.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/app/db/repositories/branch_search.py src/app/services/query_understanding.py src/app/services/search_service.py tests/integration/test_vector_search.py tests/unit/test_search_service.py
git commit -m "feat: tune object branch for Co-DETR evidence"
```

### Task 6: Validate That Search Latency Stays Stable While Object Queries Improve

**Files:**
- No required code files unless regressions are found.
- Reference: `src/app/services/search_service.py`, current local dataset

- [ ] **Step 1: Run the focused automated regression suite**

Run:

```bash
PYTHONPATH=src ./.conda/bin/python -m pytest \
  tests/unit/test_config.py \
  tests/unit/test_codetr_adapter.py \
  tests/unit/test_detector_factory.py \
  tests/unit/test_pipeline_vectors.py \
  tests/unit/test_search_service.py \
  tests/integration/test_real_pipeline.py \
  tests/integration/test_search_api.py \
  tests/integration/test_vector_search.py -v
```

Expected: PASS.

- [ ] **Step 2: Restart API in local-only mode**

Run:

```bash
OPENAI_ENABLED=false bash scripts/run_api.sh
```

Expected: API health endpoint returns `{"status":"ok"}`.

- [ ] **Step 3: Run manual object-centric query checks**

Use at least these queries against an indexed clip:
- `three people standing near a car`
- `a black sports car on a racetrack at night`
- `person on the left beside a car`

Record:
- top 5 results
- whether counts/regions improved
- wall-clock search time

- [ ] **Step 4: Compare against the user’s acceptance criteria**

Confirm:
- no online detector latency was added
- search remains in the user’s acceptable range
- object-centric ranking is visibly better than before

- [ ] **Step 5: Commit any final benchmark-note updates**

```bash
git add docs/superpowers/specs/2026-06-09-paper-grade-codetr-and-object-branch-design.md
git commit -m "docs: record Co-DETR object branch validation notes"
```

Skip the commit if no doc updates are needed, but still capture the findings in the handoff.

---

## Self-Review

- **Spec coverage:** The plan covers real Co-DETR runtime, offline enrich integration, richer object evidence, object branch tuning, and latency-safe validation.
- **Placeholder scan:** No TBD/TODO placeholders remain; each task has explicit files, tests, commands, and expected results.
- **Type consistency:** The plan consistently uses `CoDetrDetectionAdapter`, `object_detector_family`, `object_detector_model`, and the existing normalized detection payload shape.
