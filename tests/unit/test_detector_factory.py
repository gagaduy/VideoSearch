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
