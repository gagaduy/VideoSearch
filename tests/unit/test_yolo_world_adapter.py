from worker.adapters.yolo_adapter import YoloDetectionAdapter


def test_yolo_world_adapter_sets_prompt_classes(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _World:
        def __init__(self, model_name: str) -> None:
            calls["model_name"] = model_name

        def set_classes(self, classes: list[str]) -> None:
            calls["classes"] = classes

        def predict(self, source: str, verbose: bool = False) -> list[object]:
            calls["source"] = source
            calls["verbose"] = verbose
            return []

    monkeypatch.setattr("worker.adapters.yolo_adapter.YOLOWorld", _World)

    adapter = YoloDetectionAdapter(model_name="yolov8s-worldv2.pt")
    adapter.detect("frame.png", classes=["boat", "person"])

    assert calls["model_name"] == "yolov8s-worldv2.pt"
    assert calls["classes"] == ["boat", "person"]
    assert calls["source"] == "frame.png"
    assert calls["verbose"] is False


def test_yolo_world_adapter_raises_when_predict_fails(monkeypatch) -> None:
    class _World:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def set_classes(self, classes: list[str]) -> None:
            return None

        def predict(self, source: str, verbose: bool = False) -> list[object]:
            raise RuntimeError("clip dependency missing")

    monkeypatch.setattr("worker.adapters.yolo_adapter.YOLOWorld", _World)

    adapter = YoloDetectionAdapter(model_name="yolov8s-worldv2.pt")

    try:
        adapter.detect("frame.png", classes=["boat"])
    except RuntimeError as exc:
        assert "yolov8s-worldv2.pt" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
