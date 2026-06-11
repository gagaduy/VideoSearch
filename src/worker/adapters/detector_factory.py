from __future__ import annotations

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
