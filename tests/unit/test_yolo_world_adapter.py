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


def test_yolo_world_adapter_reuses_same_prompt_set_without_resetting_classes(monkeypatch) -> None:
    calls = {"set_classes": 0}

    class _World:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def set_classes(self, classes: list[str]) -> None:
            calls["set_classes"] += 1

        def predict(self, source: str, verbose: bool = False) -> list[object]:
            return []

    monkeypatch.setattr("worker.adapters.yolo_adapter.YOLOWorld", _World)

    adapter = YoloDetectionAdapter(model_name="yolov8s-worldv2.pt")
    adapter.detect("frame1.png", classes=["boat", "person"])
    adapter.detect("frame2.png", classes=["boat", "person"])

    assert calls["set_classes"] == 1


def test_yolo_world_adapter_filters_by_threshold_and_top_k(monkeypatch) -> None:
    class _Scalar:
        def __init__(self, value: float) -> None:
            self._value = value

        def item(self) -> float:
            return self._value

    class _TensorList:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    class _Box:
        def __init__(self, cls_id: int, score: float) -> None:
            self.cls = [_Scalar(float(cls_id))]
            self.conf = [_Scalar(score)]
            self.xyxy = [_TensorList([0.0, 0.0, 1.0, 1.0])]

    class _Result:
        names = {0: "car", 1: "person", 2: "chair"}
        boxes = [_Box(0, 0.95), _Box(1, 0.4), _Box(2, 0.1)]

    class _World:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def set_classes(self, classes: list[str]) -> None:
            return None

        def predict(self, source: str, verbose: bool = False) -> list[object]:
            return [_Result()]

    monkeypatch.setattr("worker.adapters.yolo_adapter.YOLOWorld", _World)

    adapter = YoloDetectionAdapter(
        model_name="yolov8s-worldv2.pt",
        confidence_threshold=0.2,
        max_detections=2,
    )
    detections = adapter.detect("frame.png", classes=["car", "person", "chair"])

    assert [item["label"] for item in detections] == ["car", "person"]


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
