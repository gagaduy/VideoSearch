from __future__ import annotations

from worker.adapters.yolo_adapter import YoloDetectionAdapter
from worker.retrieval_ontology import normalize_query_object_terms


def refine_object_matches(
    image_entries: list[dict[str, object]],
    query_terms: list[str],
    detector: YoloDetectionAdapter | None = None,
) -> dict[int, float]:
    prompts = normalize_query_object_terms(query_terms)
    if not prompts:
        return {}

    detector = detector or YoloDetectionAdapter()
    scores: dict[int, float] = {}
    for entry in image_entries:
        frame_id = int(entry.get("frame_id", 0) or 0)
        image_path = str(entry.get("image_path", "")).strip()
        if frame_id <= 0 or not image_path:
            continue
        detections = detector.detect(image_path, classes=prompts)
        best = max(
            (
                float(item.get("score", 0.0))
                for item in detections
                if str(item.get("label", "")).strip().lower() in prompts
            ),
            default=0.0,
        )
        if best > 0:
            scores[frame_id] = best
    return scores
