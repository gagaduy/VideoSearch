from __future__ import annotations

from app.config import settings
from worker.retrieval_ontology import build_indexing_prompts, canonicalize_object_label

try:
    from ultralytics import YOLOWorld
except Exception:  # pragma: no cover - import availability is exercised via runtime smoke tests
    YOLOWorld = None


class YoloDetectionAdapter:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        confidence_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.yolo_model
        self.confidence_threshold = settings.yolo_confidence_threshold if confidence_threshold is None else confidence_threshold
        self.max_detections = settings.yolo_max_detections if max_detections is None else max_detections
        self._model = None
        self._configured_classes: tuple[str, ...] | None = None

    def _lazy_load(self) -> bool:
        if self._model is not None:
            return True
        if YOLOWorld is None:
            raise RuntimeError("YOLOWorld import failed; worker image is missing the required YOLO-World dependencies")
        try:
            self._model = YOLOWorld(self.model_name)
            return True
        except Exception as exc:
            self._model = None
            raise RuntimeError(f"failed to initialize YOLO-World model '{self.model_name}': {exc}") from exc

    def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
        self._lazy_load()
        try:
            prompts = list(classes or build_indexing_prompts(settings.yolo_prompt_profile))
            prompt_key = tuple(prompts)
            if self._configured_classes != prompt_key:
                self._model.set_classes(prompts)
                self._configured_classes = prompt_key
            results = self._model.predict(source=image_path, verbose=False)
            detections: list[dict[str, object]] = []
            for result in results:
                names = result.names
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    matched_prompt = str(names[class_id]).strip().lower()
                    detections.append(
                        {
                            "label": canonicalize_object_label(matched_prompt),
                            "matched_prompt": matched_prompt,
                            "score": float(box.conf[0].item()),
                            "bbox": [float(value) for value in box.xyxy[0].tolist()],
                            "image_path": image_path,
                        }
                    )
            filtered = [item for item in detections if float(item["score"]) >= self.confidence_threshold]
            filtered.sort(key=lambda item: float(item["score"]), reverse=True)
            if self.max_detections > 0:
                return filtered[: self.max_detections]
            return filtered
        except Exception as exc:
            raise RuntimeError(f"YOLO-World detection failed for model '{self.model_name}': {exc}") from exc
