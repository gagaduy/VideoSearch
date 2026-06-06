from app.config import settings


class YoloDetectionAdapter:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.yolo_model
        self._model = None

    def _lazy_load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            return True
        except Exception:
            self._model = None
            return False

    def detect(self, image_path: str) -> list[dict[str, object]]:
        if self._lazy_load():
            try:
                results = self._model.predict(source=image_path, verbose=False)
                detections: list[dict[str, object]] = []
                for result in results:
                    names = result.names
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        detections.append(
                            {
                                "label": str(names[class_id]),
                                "score": float(box.conf[0].item()),
                                "bbox": [float(value) for value in box.xyxy[0].tolist()],
                                "image_path": image_path,
                            }
                        )
                if detections:
                    return detections
            except Exception:
                pass
        return []
