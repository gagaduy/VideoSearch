# Co-DETR Offline Object Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current offline YOLO-World object detector path with a Co-DETR-backed adapter for indexing/enrich, while preserving search latency and keeping the retrieval-layer object contract stable.

**Architecture:** Keep object detection behind the worker adapter layer and preserve the normalized detection payload consumed by `src/worker/pipeline.py`. Add a new Co-DETR adapter, switch detector selection through config, keep detector family metadata in segment raw payloads, and validate both correctness and indexing-time overhead with targeted worker tests and smoke benchmarks.

**Tech Stack:** Python 3.11, FastAPI worker pipeline, SQLAlchemy, PyTorch, MMDetection/Co-DETR runtime, pytest, local Postgres-backed indexing flow.

---

### Task 1: Add Config Surface For Detector Family Selection

**Files:**
- Modify: `src/app/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing config test**

```python
def test_settings_support_codetr_detector_family() -> None:
    settings = _load_settings(
        {
            "OBJECT_DETECTOR_FAMILY": "codetr",
            "CODETR_CONFIG_PATH": "configs/codetr.py",
            "CODETR_CHECKPOINT_PATH": "weights/codetr.pth",
            "CODETR_DEVICE": "cuda:0",
        }
    )
    assert settings.object_detector_family == "codetr"
    assert settings.codetr_config_path == "configs/codetr.py"
    assert settings.codetr_checkpoint_path == "weights/codetr.pth"
    assert settings.codetr_device == "cuda:0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_config.py::test_settings_support_codetr_detector_family -v`

Expected: FAIL because `Settings` does not expose Co-DETR configuration fields yet.

- [ ] **Step 3: Write minimal config implementation**

```python
class Settings(BaseSettings):
    object_detector_family: str = "yolo_world"
    codetr_config_path: str = "models/codetr/config.py"
    codetr_checkpoint_path: str = "models/codetr/model.pth"
    codetr_device: str = "cuda:0"
```

Also keep existing YOLO fields intact so fallback remains possible.

- [ ] **Step 4: Run config tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_config.py -v`

Expected: PASS, including the new Co-DETR settings test and existing detector defaults.

- [ ] **Step 5: Commit**

```bash
git add src/app/config.py tests/unit/test_config.py
git commit -m "feat: add Co-DETR detector configuration"
```

### Task 2: Introduce A Co-DETR Adapter With The Existing Detection Contract

**Files:**
- Create: `src/worker/adapters/codetr_adapter.py`
- Test: `tests/unit/test_codetr_adapter.py`
- Reference: `src/worker/adapters/yolo_adapter.py`

- [ ] **Step 1: Write the failing adapter tests**

```python
from worker.adapters.codetr_adapter import CoDetrDetectionAdapter


def test_codetr_adapter_lazy_loads_model(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _init_detector(config: str, checkpoint: str, device: str):
        calls["config"] = config
        calls["checkpoint"] = checkpoint
        calls["device"] = device
        return object()

    monkeypatch.setattr("worker.adapters.codetr_adapter.init_detector", _init_detector)
    adapter = CoDetrDetectionAdapter(
        config_path="cfg.py",
        checkpoint_path="model.pth",
        device="cuda:0",
    )
    adapter._lazy_load()
    assert calls == {"config": "cfg.py", "checkpoint": "model.pth", "device": "cuda:0"}


def test_codetr_adapter_normalizes_detections(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"frame")

    monkeypatch.setattr("worker.adapters.codetr_adapter.init_detector", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "worker.adapters.codetr_adapter.inference_detector",
        lambda model, image: {
            "pred_instances": {
                "labels": [2, 2],
                "scores": [0.95, 0.35],
                "bboxes": [[10, 20, 110, 220], [0, 0, 5, 5]],
            }
        },
    )
    monkeypatch.setattr(
        "worker.adapters.codetr_adapter.resolve_codetr_label",
        lambda class_id: {2: "car"}[class_id],
    )

    adapter = CoDetrDetectionAdapter(
        confidence_threshold=0.5,
        max_detections=8,
        config_path="cfg.py",
        checkpoint_path="model.pth",
    )
    detections = adapter.detect(str(image_path))
    assert detections == [
        {
            "label": "car",
            "matched_prompt": "car",
            "score": 0.95,
            "bbox": [10.0, 20.0, 110.0, 220.0],
            "image_path": str(image_path),
        }
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_codetr_adapter.py -v`

Expected: FAIL because the Co-DETR adapter file and helpers do not exist yet.

- [ ] **Step 3: Write the minimal adapter**

```python
from __future__ import annotations

from app.config import settings

try:
    from mmdet.apis import inference_detector, init_detector
except Exception:  # pragma: no cover
    inference_detector = None
    init_detector = None


def resolve_codetr_label(class_id: int) -> str:
    return str(class_id)


class CoDetrDetectionAdapter:
    def __init__(
        self,
        *,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> None:
        self.config_path = config_path or settings.codetr_config_path
        self.checkpoint_path = checkpoint_path or settings.codetr_checkpoint_path
        self.device = device or settings.codetr_device
        self.confidence_threshold = settings.yolo_confidence_threshold if confidence_threshold is None else confidence_threshold
        self.max_detections = settings.yolo_max_detections if max_detections is None else max_detections
        self._model = None

    def _lazy_load(self) -> bool:
        if self._model is not None:
            return True
        if init_detector is None:
            raise RuntimeError("Co-DETR dependencies are unavailable")
        self._model = init_detector(self.config_path, self.checkpoint_path, device=self.device)
        return True

    def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
        self._lazy_load()
        result = inference_detector(self._model, image_path)
        ...
```

Normalize output to the same fields used by `YoloDetectionAdapter`: `label`, `matched_prompt`, `score`, `bbox`, `image_path`. Ignore `classes` for now because this phase is offline indexing only.

- [ ] **Step 4: Run adapter tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_codetr_adapter.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worker/adapters/codetr_adapter.py tests/unit/test_codetr_adapter.py
git commit -m "feat: add Co-DETR detection adapter"
```

### Task 3: Add A Detector Factory So Pipeline Chooses YOLO Or Co-DETR From Config

**Files:**
- Create: `src/worker/adapters/detector_factory.py`
- Modify: `src/worker/pipeline.py`
- Test: `tests/unit/test_detector_factory.py`

- [ ] **Step 1: Write the failing factory tests**

```python
from worker.adapters.detector_factory import build_object_detector


def test_detector_factory_builds_codetr(monkeypatch) -> None:
    class _CoDetr:
        pass

    monkeypatch.setattr("worker.adapters.detector_factory.CoDetrDetectionAdapter", _CoDetr)
    monkeypatch.setattr("worker.adapters.detector_factory.settings.object_detector_family", "codetr")
    detector = build_object_detector()
    assert isinstance(detector, _CoDetr)


def test_detector_factory_builds_yolo_world(monkeypatch) -> None:
    class _Yolo:
        pass

    monkeypatch.setattr("worker.adapters.detector_factory.YoloDetectionAdapter", _Yolo)
    monkeypatch.setattr("worker.adapters.detector_factory.settings.object_detector_family", "yolo_world")
    detector = build_object_detector()
    assert isinstance(detector, _Yolo)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_detector_factory.py -v`

Expected: FAIL because the factory module does not exist yet.

- [ ] **Step 3: Implement the factory and wire pipeline through it**

```python
from app.config import settings
from worker.adapters.codetr_adapter import CoDetrDetectionAdapter
from worker.adapters.yolo_adapter import YoloDetectionAdapter


def build_object_detector():
    family = settings.object_detector_family.strip().lower()
    if family == "codetr":
        return CoDetrDetectionAdapter()
    if family == "yolo_world":
        return YoloDetectionAdapter()
    raise ValueError(f"unsupported object detector family: {settings.object_detector_family}")
```

In `src/worker/pipeline.py`, replace:

```python
detector = YoloDetectionAdapter()
```

with:

```python
from worker.adapters.detector_factory import build_object_detector
...
detector = build_object_detector()
```

- [ ] **Step 4: Run factory and pipeline tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_detector_factory.py tests/unit/test_pipeline_vectors.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worker/adapters/detector_factory.py src/worker/pipeline.py tests/unit/test_detector_factory.py
git commit -m "refactor: select offline detector via factory"
```

### Task 4: Preserve Detector Family Metadata In Segment Outputs

**Files:**
- Modify: `src/worker/pipeline.py`
- Test: `tests/integration/test_real_pipeline.py`

- [ ] **Step 1: Write the failing integration test**

```python
def test_pipeline_records_codetr_detector_family(monkeypatch, tmp_path: Path) -> None:
    ...
    monkeypatch.setattr("worker.adapters.detector_factory.build_object_detector", lambda: _Detector())
    monkeypatch.setattr(pipeline.settings, "object_detector_family", "codetr")
    payload = pipeline.run_index_pipeline(...)
    session = SessionLocal()
    segment = session.query(Segment).filter_by(video_id=video.id).first()
    assert segment.raw_json["object_detector_family"] == "codetr"
```

The stub detector should expose `model_name = "co_detr_stub"` so the test can also assert `segment.raw_json["object_detector_model"] == "co_detr_stub"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_real_pipeline.py::test_pipeline_records_codetr_detector_family -v`

Expected: FAIL because the pipeline still hardcodes `yolo_world` metadata.

- [ ] **Step 3: Update the segment metadata**

Change the enrich object block so it records the actual family/model from the active detector:

```python
detector_family = settings.object_detector_family.strip().lower()
segment.raw_json["object_detector_family"] = detector_family
segment.raw_json["object_detector_model"] = getattr(detector, "model_name", detector.__class__.__name__)
```

Keep `object_labels_json`, `object_counts_json`, and `object_positions_json` unchanged.

- [ ] **Step 4: Run the integration slice**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/integration/test_real_pipeline.py -k detector_family -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/integration/test_real_pipeline.py
git commit -m "feat: record active detector family in segment metadata"
```

### Task 5: Keep Retrieval-Layer Object Consumption Compatible

**Files:**
- Modify: `src/app/services/search_service.py`
- Modify: `src/app/services/question_search.py`
- Test: `tests/unit/test_search_service.py`
- Test: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write the failing compatibility test**

```python
def test_search_display_preserves_object_counts_from_codetr_indexed_rows() -> None:
    row = {
        "object_counts": {"car": 2, "person": 1},
        "object_labels": ["car", "person"],
        "raw_json": {"object_detector_family": "codetr", "object_detector_model": "co_detr_r50"},
    }
    result = _serialize_result_row(...)
    assert result["object_counts"] == {"car": 2, "person": 1}
    assert result["diagnostics"]["object_detector_family"] == "codetr"
```

If no serializer exists, write the test against the final `results[0]["diagnostics"]` payload in `tests/integration/test_search_api.py`.

- [ ] **Step 2: Run the failing test**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_search_service.py tests/integration/test_search_api.py -k detector_family -v`

Expected: FAIL because detector family/model are not surfaced consistently in search results yet.

- [ ] **Step 3: Implement the minimal compatibility layer**

Expose detector provenance in diagnostics only; do not change ranking:

```python
"diagnostics": {
    ...,
    "object_detector_family": str(row.get("raw_json", {}).get("object_detector_family", "")),
    "object_detector_model": str(row.get("raw_json", {}).get("object_detector_model", "")),
}
```

Do the same in question-search result shaping if it returns object evidence cards.

- [ ] **Step 4: Run compatibility tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_search_service.py tests/integration/test_search_api.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/app/services/search_service.py src/app/services/question_search.py tests/unit/test_search_service.py tests/integration/test_search_api.py
git commit -m "feat: surface detector provenance in search diagnostics"
```

### Task 6: Add Offline Smoke Benchmark Guard Rails

**Files:**
- Modify: `src/worker/pipeline.py`
- Test: `tests/unit/test_pipeline_vectors.py`
- Optional doc note: `docs/superpowers/specs/2026-06-08-worker-throughput-and-smoothness-without-quality-tradeoff.md`

- [ ] **Step 1: Write the failing timing test**

```python
def test_pipeline_stage_timings_include_detector_stage_for_codetr(monkeypatch, tmp_path: Path) -> None:
    timings: dict[str, float] = {}
    ...
    payload = pipeline.run_index_pipeline(...)
    assert "detector" in payload["stage_timings"]
    assert payload["stage_timings"]["detector"] >= 0.0
```

- [ ] **Step 2: Run the failing test**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_pipeline_vectors.py -k detector_stage -v`

Expected: FAIL if the Co-DETR path does not preserve detector timing yet.

- [ ] **Step 3: Preserve timing parity**

Ensure the Co-DETR-backed path still goes through the existing timed detector block in `_enrich_segment_keyframes`:

```python
detector_started = perf_counter()
objects = detector.detect(image_path, classes=object_prompts)
...
_record_stage_timing(stage_timings, "detector", detector_started)
```

Do not introduce a new untimed object stage that would hide the performance cost.

- [ ] **Step 4: Run timing and worker regression tests**

Run: `PYTHONPATH=src ./.conda/bin/python -m pytest tests/unit/test_pipeline_vectors.py tests/integration/test_real_pipeline.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py tests/unit/test_pipeline_vectors.py
git commit -m "test: preserve detector timing coverage for Co-DETR"
```

### Task 7: Run End-To-End Offline Verification And Capture Benchmark Notes

**Files:**
- No code changes required unless issues are found.
- Reference: `src/worker/pipeline.py`, `data/`, local runtime env

- [ ] **Step 1: Run the focused automated suite**

Run:

```bash
PYTHONPATH=src ./.conda/bin/python -m pytest \
  tests/unit/test_config.py \
  tests/unit/test_codetr_adapter.py \
  tests/unit/test_detector_factory.py \
  tests/unit/test_pipeline_vectors.py \
  tests/unit/test_search_service.py \
  tests/integration/test_real_pipeline.py \
  tests/integration/test_search_api.py -v
```

Expected: PASS.

- [ ] **Step 2: Run one offline indexing smoke job with Co-DETR enabled**

Run:

```bash
export OBJECT_DETECTOR_FAMILY=codetr
PYTHONPATH=src ./.conda/bin/python -m worker.main
```

Then enqueue one short video through the existing API/UI and capture:
- total job duration
- `stage_timings["detector"]`
- number of `segments`
- sample `object_counts_json` rows

- [ ] **Step 3: Compare against YOLO baseline**

Run the same short clip once with:

```bash
export OBJECT_DETECTOR_FAMILY=yolo_world
PYTHONPATH=src ./.conda/bin/python -m worker.main
```

Record:
- total job duration
- detector stage timing
- a small sample of object labels/counts for the same kind of scene

- [ ] **Step 4: Verify acceptance criteria**

Write down whether the results meet the design constraints:
- search latency unchanged (no online detector path changed)
- offline object metadata visibly improved or at least richer
- 10-minute clip projection remains inside the user’s acceptable 30–45 minute range

- [ ] **Step 5: Commit any benchmark-note updates if needed**

```bash
git add docs/superpowers/specs/2026-06-08-worker-throughput-and-smoothness-without-quality-tradeoff.md
git commit -m "docs: record Co-DETR offline benchmark notes"
```

If no doc update is needed, skip the commit and just record the benchmark findings in the implementation handoff.

---

## Self-Review

- **Spec coverage:** The plan covers offline-only detector replacement, adapter compatibility, config rollback, detector metadata, timing visibility, and benchmark verification without adding online search latency.
- **Placeholder scan:** No TBD/TODO placeholders remain; each task has concrete files, test code, commands, and expected outcomes.
- **Type consistency:** The plan consistently uses `CoDetrDetectionAdapter`, `build_object_detector()`, `object_detector_family`, and the existing normalized detection payload keys.
