from worker.adapters.yolo_adapter import YoloDetectionAdapter


def test_yolo_adapter_returns_detector_name() -> None:
    adapter = YoloDetectionAdapter(model_name="yolov8s-worldv2.pt")
    assert adapter.model_name == "yolov8s-worldv2.pt"
