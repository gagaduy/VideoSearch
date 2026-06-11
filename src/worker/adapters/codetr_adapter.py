from __future__ import annotations

import json
import os
import select
import subprocess
from pathlib import Path
from typing import Any

from app.config import settings
from worker.retrieval_ontology import canonicalize_object_label

try:
    from mmdet.apis import inference_detector, init_detector
except Exception:  # pragma: no cover - runtime availability depends on optional MMDetection install
    inference_detector = None
    init_detector = None


COCO_INSTANCE_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def resolve_codetr_label(class_id: int) -> str:
    if 0 <= class_id < len(COCO_INSTANCE_CLASSES):
        return COCO_INSTANCE_CLASSES[class_id]
    return str(class_id)


def _to_sequence(value: object) -> list[object]:
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class CoDetrDetectionAdapter:
    def __init__(
        self,
        *,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
        max_detections: int | None = None,
        runtime_backend: str | None = None,
        repo_path: str | None = None,
        python_path: str | None = None,
        startup_timeout_sec: float | None = None,
        request_timeout_sec: float | None = None,
    ) -> None:
        self.model_name = "co_detr"
        self.config_path = config_path or settings.codetr_config_path
        self.checkpoint_path = checkpoint_path or settings.codetr_checkpoint_path
        self.device = device or settings.codetr_device
        self.confidence_threshold = settings.yolo_confidence_threshold if confidence_threshold is None else confidence_threshold
        self.max_detections = settings.yolo_max_detections if max_detections is None else max_detections
        self.runtime_backend = (runtime_backend or settings.codetr_runtime_backend).strip().lower()
        self.repo_path = repo_path or settings.codetr_repo_path
        self.python_path = python_path or settings.codetr_python_path
        self.startup_timeout_sec = (
            settings.codetr_startup_timeout_sec if startup_timeout_sec is None else startup_timeout_sec
        )
        self.request_timeout_sec = (
            settings.codetr_request_timeout_sec if request_timeout_sec is None else request_timeout_sec
        )
        self._model = None
        self._process: subprocess.Popen[str] | None = None

    def _lazy_load(self) -> bool:
        if self.runtime_backend == "subprocess":
            return self._lazy_load_subprocess()
        return self._lazy_load_direct()

    def _lazy_load_direct(self) -> bool:
        if self._model is not None:
            return True
        if init_detector is None:
            raise RuntimeError("Co-DETR dependencies are unavailable")
        try:
            self._model = init_detector(self.config_path, self.checkpoint_path, device=self.device)
            return True
        except Exception as exc:
            self._model = None
            raise RuntimeError(f"failed to initialize Co-DETR model: {exc}") from exc

    def _lazy_load_subprocess(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True

        script_path = Path(__file__).with_name("codetr_runtime_server.py").resolve()
        repo_root = Path(__file__).resolve().parents[3]
        cmd = [
            str(self.python_path),
            "-u",
            str(script_path),
            "--repo-path",
            str(Path(self.repo_path).resolve()),
            "--config",
            str(Path(self.config_path).resolve()),
            "--checkpoint",
            str(Path(self.checkpoint_path).resolve()),
            "--device",
            self.device,
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONWARNINGS", "ignore")
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._process = process
        payload = self._read_subprocess_payload(timeout_sec=self.startup_timeout_sec)
        if payload.get("status") != "ready":
            self.close()
            error = str(payload.get("error", "unknown startup failure"))
            raise RuntimeError(f"failed to initialize Co-DETR subprocess runtime: {error}")
        return True

    def _read_subprocess_payload(self, *, timeout_sec: float) -> dict[str, object]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Co-DETR subprocess runtime is not running")

        ready, _, _ = select.select([self._process.stdout], [], [], timeout_sec)
        if not ready:
            error = self._collect_process_stderr()
            self.close()
            raise RuntimeError(
                f"Co-DETR subprocess timed out after {timeout_sec:.1f}s"
                + (f": {error}" if error else "")
            )

        line = self._process.stdout.readline()
        if not line:
            error = self._collect_process_stderr()
            self.close()
            raise RuntimeError("Co-DETR subprocess exited unexpectedly" + (f": {error}" if error else ""))
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            error = self._collect_process_stderr()
            self.close()
            raise RuntimeError(
                f"Co-DETR subprocess returned invalid JSON: {line.strip()}"
                + (f" | stderr: {error}" if error else "")
            ) from exc

    def _collect_process_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            if self._process.poll() is None:
                return ""
            return self._process.stderr.read().strip()
        except Exception:
            return ""

    def _direct_detections(self, image_path: str) -> list[dict[str, object]]:
        try:
            result = inference_detector(self._model, image_path)
            pred_instances = result.get("pred_instances", result) if isinstance(result, dict) else getattr(result, "pred_instances", result)
            labels = _to_sequence(pred_instances.get("labels")) if isinstance(pred_instances, dict) else _to_sequence(getattr(pred_instances, "labels", []))
            scores = _to_sequence(pred_instances.get("scores")) if isinstance(pred_instances, dict) else _to_sequence(getattr(pred_instances, "scores", []))
            bboxes = _to_sequence(pred_instances.get("bboxes")) if isinstance(pred_instances, dict) else _to_sequence(getattr(pred_instances, "bboxes", []))
            serialized = [
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "bbox": [float(value) for value in _to_sequence(bbox)],
                }
                for class_id, score, bbox in zip(labels, scores, bboxes)
            ]
            return self._normalize_detections(image_path, serialized)
        except Exception as exc:
            raise RuntimeError(f"Co-DETR detection failed: {exc}") from exc

    def _subprocess_detections(self, image_path: str) -> list[dict[str, object]]:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Co-DETR subprocess runtime is not running")
        request = {
            "action": "detect",
            "image_path": image_path,
            "confidence_threshold": self.confidence_threshold,
            "max_detections": self.max_detections,
        }
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        payload = self._read_subprocess_payload(timeout_sec=self.request_timeout_sec)
        if payload.get("status") != "ok":
            error = str(payload.get("error", "unknown detection failure"))
            raise RuntimeError(f"Co-DETR subprocess detection failed: {error}")
        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            raise RuntimeError("Co-DETR subprocess returned malformed detections payload")
        return self._normalize_detections(image_path, detections)

    def _normalize_detections(self, image_path: str, detections: list[dict[str, object]]) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for item in detections:
            try:
                score_value = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score_value < self.confidence_threshold:
                continue
            try:
                class_id = int(item.get("class_id", -1))
            except (TypeError, ValueError):
                class_id = -1
            raw_label = resolve_codetr_label(class_id)
            matched_prompt = str(raw_label).strip().lower()
            bbox = item.get("bbox", [])
            if not isinstance(bbox, list):
                bbox = _to_sequence(bbox)
            normalized.append(
                {
                    "label": canonicalize_object_label(matched_prompt),
                    "matched_prompt": matched_prompt,
                    "score": score_value,
                    "bbox": [float(value) for value in _to_sequence(bbox)],
                    "image_path": image_path,
                }
            )
        normalized.sort(key=lambda item: float(item["score"]), reverse=True)
        if self.max_detections > 0:
            return normalized[: self.max_detections]
        return normalized

    def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
        del classes  # Co-DETR offline indexing uses its closed-set label space in this phase.
        self._lazy_load()
        if self.runtime_backend == "subprocess":
            return self._subprocess_detections(image_path)
        return self._direct_detections(image_path)

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                process.stdin.flush()
        except Exception:
            pass
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass
