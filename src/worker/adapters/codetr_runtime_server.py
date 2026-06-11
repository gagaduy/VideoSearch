from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _to_python_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _serialize_result(result: Any) -> list[dict[str, object]]:
    if isinstance(result, tuple):
        result = result[0]

    detections: list[dict[str, object]] = []

    pred_instances = None
    if isinstance(result, dict):
        pred_instances = result.get("pred_instances")
    else:
        pred_instances = getattr(result, "pred_instances", None)

    if pred_instances is not None:
        labels = pred_instances["labels"] if isinstance(pred_instances, dict) else getattr(pred_instances, "labels", [])
        scores = pred_instances["scores"] if isinstance(pred_instances, dict) else getattr(pred_instances, "scores", [])
        bboxes = pred_instances["bboxes"] if isinstance(pred_instances, dict) else getattr(pred_instances, "bboxes", [])
        for class_id, score, bbox in zip(_to_python_list(labels), _to_python_list(scores), _to_python_list(bboxes)):
            detections.append(
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "bbox": [float(value) for value in _to_python_list(bbox)],
                }
            )
        return detections

    for class_id, class_result in enumerate(result):
        rows = class_result.tolist() if hasattr(class_result, "tolist") else list(class_result)
        for row in rows:
            if len(row) < 5:
                continue
            detections.append(
                {
                    "class_id": class_id,
                    "score": float(row[4]),
                    "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                }
            )
    return detections


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        print(json.dumps({"status": "error", "error": f"missing Co-DETR repo at {repo_path}"}), flush=True)
        return 1

    sys.path.insert(0, str(repo_path))

    try:
        import matplotlib

        matplotlib.use("Agg")
        from mmcv import Config
        from mmdet.apis import inference_detector, init_detector
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"failed to import Co-DETR runtime: {exc}"}), flush=True)
        return 1

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        print(json.dumps({"status": "error", "error": f"missing Co-DETR checkpoint at {checkpoint_path}"}), flush=True)
        return 1

    try:
        config = Config.fromfile(str(Path(args.config).resolve()))
        config.log_level = "ERROR"
        model = init_detector(config, str(checkpoint_path), device=args.device)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"failed to initialize Co-DETR model: {exc}"}), flush=True)
        return 1

    print(
        json.dumps(
            {
                "status": "ready",
                "config": str(Path(args.config).resolve()),
                "checkpoint": str(checkpoint_path),
                "device": args.device,
            }
        ),
        flush=True,
    )

    for raw_line in sys.stdin:
        message = raw_line.strip()
        if not message:
            continue
        try:
            payload = json.loads(message)
        except json.JSONDecodeError as exc:
            print(json.dumps({"status": "error", "error": f"invalid request JSON: {exc}"}), flush=True)
            continue

        action = str(payload.get("action", "")).strip().lower()
        if action == "shutdown":
            print(json.dumps({"status": "bye"}), flush=True)
            return 0
        if action != "detect":
            print(json.dumps({"status": "error", "error": f"unsupported action: {action}"}), flush=True)
            continue

        image_path = str(payload.get("image_path", "")).strip()
        if not image_path:
            print(json.dumps({"status": "error", "error": "missing image_path"}), flush=True)
            continue

        try:
            detections = _serialize_result(inference_detector(model, image_path))
            print(json.dumps({"status": "ok", "detections": detections}), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "error", "error": f"Co-DETR inference failed: {exc}"}), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
