from pathlib import Path

from worker.adapters.codetr_adapter import CoDetrDetectionAdapter, resolve_codetr_label


def test_resolve_codetr_label_maps_coco_ids() -> None:
    assert resolve_codetr_label(0) == "person"
    assert resolve_codetr_label(2) == "car"
    assert resolve_codetr_label(3) == "motorcycle"
    assert resolve_codetr_label(4) == "airplane"


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
        runtime_backend="direct",
    )
    adapter._lazy_load()

    assert calls == {"config": "cfg.py", "checkpoint": "model.pth", "device": "cuda:0"}


def test_codetr_adapter_normalizes_detections(monkeypatch, tmp_path: Path) -> None:
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
        runtime_backend="direct",
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


def test_codetr_adapter_raises_when_dependencies_missing(monkeypatch) -> None:
    monkeypatch.setattr("worker.adapters.codetr_adapter.init_detector", None)
    adapter = CoDetrDetectionAdapter(config_path="cfg.py", checkpoint_path="model.pth", runtime_backend="direct")

    try:
        adapter._lazy_load()
    except RuntimeError as exc:
        assert "Co-DETR dependencies are unavailable" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_codetr_adapter_subprocess_normalizes_detections(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"frame")

    class _FakeStdout:
        def __init__(self) -> None:
            self._lines = iter(
                [
                    '{"status":"ready"}\n',
                    '{"status":"ok","detections":[{"class_id":2,"score":0.91,"bbox":[1,2,3,4]}]}\n',
                ]
            )

        def fileno(self) -> int:
            return 0

        def readline(self) -> str:
            return next(self._lines, "")

    class _FakeStdin:
        def __init__(self) -> None:
            self.writes: list[str] = []

        def write(self, text: str) -> None:
            self.writes.append(text)

        def flush(self) -> None:
            return None

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = _FakeStdout()
            self.stdin = _FakeStdin()
            self.stderr = None

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            return None

    fake_process = _FakeProcess()

    monkeypatch.setattr("worker.adapters.codetr_adapter.subprocess.Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("worker.adapters.codetr_adapter.select.select", lambda *args, **kwargs: ([fake_process.stdout], [], []))

    adapter = CoDetrDetectionAdapter(
        config_path="cfg.py",
        checkpoint_path="model.pth",
        runtime_backend="subprocess",
        python_path="python",
        repo_path="third_party/Co-DETR",
    )

    detections = adapter.detect(str(image_path))

    assert detections == [
        {
            "label": "car",
            "matched_prompt": "car",
            "score": 0.91,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "image_path": str(image_path),
        }
    ]
