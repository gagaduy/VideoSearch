from worker.adapters.yolo_adapter import YoloDetectionAdapter


def test_yolo_adapter_returns_detector_name() -> None:
    adapter = YoloDetectionAdapter(model_name="yolo11n.pt")
    assert adapter.model_name == "yolo11n.pt"

