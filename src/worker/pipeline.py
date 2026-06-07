from math import sqrt
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Frame, FrameCaption, FrameEmbedding, FrameObject, FrameOcr, IndexJob, Segment, Video
from app.services.storage import ensure_data_dirs
from worker.adapters.caption_adapter import CaptionAdapter
from worker.adapters.internvl_adapter import InternvlAdapter
from worker.adapters.openclip_adapter import OpenClipAdapter
from worker.adapters.paddleocr_adapter import PaddleOcrAdapter
from worker.adapters.semantic_entity_adapter import SemanticEntityAdapter
from worker.adapters.yolo_adapter import YoloDetectionAdapter
from worker.io import copy_or_create_thumbnail, extract_frames_ffmpeg, write_placeholder_image
from worker.retrieval_ontology import build_indexing_prompts
from worker.sampling import keep_distinct_frames

SEGMENT_DISTANCE_THRESHOLD = 0.28
MAX_SEGMENT_DURATION_SEC = 12.0


def _update_job_stage(db: Session, job_id: int | None, *, status: str | None = None, stage: str | None = None) -> None:
    if job_id is None:
        return
    job = db.get(IndexJob, job_id)
    if job is None:
        return
    if status is not None:
        job.status = status
    if stage is not None:
        job.stage = stage
    db.commit()


def _prepare_frame_paths(video_id: int, source_path: Path) -> list[Path]:
    frame_dir = Path(settings.frames_dir) / f"video_{video_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        extracted = extract_frames_ffmpeg(source_path, frame_dir, fps=1.0)
        if extracted:
            return extracted
    fallback_frame = frame_dir / "frame_000001.png"
    return [write_placeholder_image(fallback_frame)]


def _dot(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    return sum(left[index] * right[index] for index in range(size))


def _norm(values: list[float]) -> float:
    return sqrt(sum(value * value for value in values))


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 1.0
    denominator = _norm(left) * _norm(right)
    if denominator <= 1e-8:
        return 1.0
    cosine_similarity = max(min(_dot(left, right) / denominator, 1.0), -1.0)
    return 1.0 - cosine_similarity


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    size = min(len(vector) for vector in vectors)
    averaged = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(size)]
    norm = _norm(averaged)
    if norm <= 1e-8:
        return averaged
    return [value / norm for value in averaged]


def _optional_mean_vector(vectors: list[list[float]]) -> list[float] | None:
    non_empty = [vector for vector in vectors if vector]
    if not non_empty:
        return None
    return _mean_vector(non_empty)


def _unique_text(items: list[str]) -> str:
    seen: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return " ".join(seen)


def _build_lightweight_caption(objects: list[dict[str, object]], ocr_text: str) -> str:
    labels = [str(item.get("label", "")).strip().lower() for item in objects if str(item.get("label", "")).strip()]
    return _unique_text([*labels, ocr_text.strip().lower()])


def _should_run_vlm_enrichment(
    *,
    segment_index: int,
    segment_count: int,
    profile: str,
    sparse_stride: int,
    ocr_text: str,
    objects: list[dict[str, object]],
) -> bool:
    normalized_profile = profile.strip().lower()
    if normalized_profile == "full":
        return True
    if not ocr_text.strip() or not objects:
        return True
    if segment_index == 1 or segment_index == segment_count:
        return True
    stride = max(sparse_stride, 1)
    return ((segment_index - 1) % stride) == 0


def _object_counts(objects: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in objects:
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def _object_positions(objects: list[dict[str, object]], image_path: str) -> dict[str, list[str]]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        width, height = 0, 0

    positions: dict[str, set[str]] = {}
    for item in objects:
        label = str(item.get("label", "")).strip()
        bbox = item.get("bbox", [])
        if not label or not isinstance(bbox, list) or len(bbox) != 4 or width <= 0 or height <= 0:
            continue

        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0

        horizontal = "left" if center_x < width / 3 else "right" if center_x > (2 * width) / 3 else "center"
        vertical = "top" if center_y < height / 3 else "bottom" if center_y > (2 * height) / 3 else "middle"
        positions.setdefault(label, set()).update({horizontal, vertical})

    return {label: sorted(regions) for label, regions in positions.items()}


def _build_segments(frame_items: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    if not frame_items:
        return []
    segments: list[list[dict[str, object]]] = [[frame_items[0]]]
    for item in frame_items[1:]:
        current_segment = segments[-1]
        previous_item = current_segment[-1]
        distance = _cosine_distance(
            list(previous_item["embedding_branch_a"]),
            list(item["embedding_branch_a"]),
        )
        segment_duration = float(item["timestamp_sec"]) - float(current_segment[0]["timestamp_sec"])
        if distance > SEGMENT_DISTANCE_THRESHOLD or segment_duration >= MAX_SEGMENT_DURATION_SEC:
            segments.append([item])
            continue
        current_segment.append(item)
    return segments


def _clear_video_index(db: Session, video_id: int) -> None:
    frame_ids = list(
        db.execute(
            select(Frame.id).where(Frame.video_id == video_id)
        ).scalars()
    )
    if frame_ids:
        db.execute(delete(FrameObject).where(FrameObject.frame_id.in_(frame_ids)))
        db.execute(delete(FrameOcr).where(FrameOcr.frame_id.in_(frame_ids)))
        db.execute(delete(FrameCaption).where(FrameCaption.frame_id.in_(frame_ids)))
        db.execute(delete(FrameEmbedding).where(FrameEmbedding.frame_id.in_(frame_ids)))
        db.execute(delete(Frame).where(Frame.id.in_(frame_ids)))
    db.execute(delete(Segment).where(Segment.video_id == video_id))
    db.commit()


def index_prepared_frames(
    db: Session,
    video_id: int,
    frame_sources: list[dict[str, object]],
    *,
    openclip: OpenClipAdapter,
    captioner: CaptionAdapter,
    ocr_engine: PaddleOcrAdapter,
    detector: YoloDetectionAdapter,
    entity_extractor: SemanticEntityAdapter,
    branch_b_adapter: InternvlAdapter,
    job_id: int | None = None,
    source_tag: str = "prepared_frames",
) -> dict[str, object]:
    _clear_video_index(db, video_id)
    _update_job_stage(db, job_id, status="running", stage=f"embedding_frames:0/{len(frame_sources)}")

    frame_items: list[dict[str, object]] = []
    last_embedding: list[float] = []

    for index, frame_source in enumerate(frame_sources, start=1):
        frame_path = Path(str(frame_source["image_path"]))
        timestamp_sec = float(frame_source.get("timestamp_sec", index - 1))
        frame_index = int(frame_source.get("frame_index", index))
        thumb_path = Path(settings.thumbs_dir) / f"video_{video_id}" / f"{frame_path.stem}.webp"
        copy_or_create_thumbnail(frame_path, thumb_path)

        embedding = openclip.embed_image(str(frame_path))

        frame = Frame(
            video_id=video_id,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            image_path=str(frame_path),
            thumb_path=str(thumb_path),
            is_keyframe=True,
        )
        db.add(frame)
        db.flush()
        db.add(FrameEmbedding(frame_id=frame.id, model_name=embedding.model_name, embedding=embedding.values))

        frame_items.append(
            {
                "frame_id": int(frame.id),
                "timestamp_sec": float(frame.timestamp_sec),
                "embedding_branch_a": list(embedding.values),
                "embedding_branch_b": [],
            }
        )
        last_embedding = embedding.values
        if job_id is not None:
            job = db.get(IndexJob, job_id)
            if job is not None:
                job.status = "running"
                job.stage = f"embedding_frames:{index}/{len(frame_sources)}"
        db.commit()

    _update_job_stage(db, job_id, status="running", stage="building_segments")
    segments = _persist_segments(db, video_id, frame_items)
    _update_job_stage(db, job_id, status="running", stage=f"enriching_segments:0/{len(segments)}")
    last_caption, last_ocr, last_objects = _enrich_segment_keyframes(
        db,
        segments,
        captioner=captioner,
        ocr_engine=ocr_engine,
        detector=detector,
        entity_extractor=entity_extractor,
        branch_b_adapter=branch_b_adapter,
        dense_encoder=openclip,
        job_id=job_id,
    )
    for segment in segments:
        segment.raw_json = {
            **dict(segment.raw_json or {}),
            "dataset": source_tag,
        }
    db.commit()

    return {
        "frame_count": len(frame_sources),
        "segment_count": len(segments),
        "embedding": last_embedding,
        "caption": last_caption,
        "ocr": last_ocr,
        "objects": last_objects,
    }


def _persist_segments(db: Session, video_id: int, frame_items: list[dict[str, object]]) -> list[Segment]:
    persisted_segments: list[Segment] = []
    for segment_index, items in enumerate(_build_segments(frame_items), start=1):
        keyframe_item = items[len(items) // 2]
        segment = Segment(
            video_id=video_id,
            segment_index=segment_index,
            start_timestamp_sec=float(items[0]["timestamp_sec"]),
            end_timestamp_sec=float(items[-1]["timestamp_sec"]),
            keyframe_id=int(keyframe_item["frame_id"]),
            caption_text="",
            ocr_text="",
            object_labels_json=[],
            ocr_tokens_json=[],
            object_counts_json={},
            object_positions_json={},
            semantic_entities_json=[],
            semantic_aliases_json={},
            semantic_counts_json={},
            embedding=_optional_mean_vector([list(item["embedding_branch_a"]) for item in items]),
            embedding_branch_a=_optional_mean_vector([list(item["embedding_branch_a"]) for item in items]),
            embedding_branch_b=_optional_mean_vector([list(item["embedding_branch_b"]) for item in items]),
            stage_failures_json={},
            raw_json={
                "frame_ids": [int(item["frame_id"]) for item in items],
                "frame_count": len(items),
            },
        )
        db.add(segment)
        db.flush()
        for item in items:
            frame = db.get(Frame, int(item["frame_id"]))
            if frame is not None:
                frame.segment_id = int(segment.id)
        persisted_segments.append(segment)
    db.commit()
    return persisted_segments


def _enrich_segment_keyframes(
    db: Session,
    segments: list[Segment],
    *,
    captioner: CaptionAdapter,
    ocr_engine: PaddleOcrAdapter,
    detector: YoloDetectionAdapter,
    entity_extractor: SemanticEntityAdapter,
    branch_b_adapter: InternvlAdapter,
    dense_encoder: OpenClipAdapter,
    job_id: int | None = None,
) -> tuple[str, str, list[dict[str, object]]]:
    last_caption = ""
    last_ocr = ""
    last_objects: list[dict[str, object]] = []
    object_prompts = build_indexing_prompts()

    for index, segment in enumerate(segments, start=1):
        frame = db.get(Frame, int(segment.keyframe_id))
        if frame is None:
            continue

        image_path = str(frame.image_path)
        stage_failures: dict[str, str] = {}

        try:
            ocr = ocr_engine.extract_text(image_path)
        except Exception as exc:
            ocr = {"text": "", "tokens": [], "raw": []}
            stage_failures["ocr"] = str(exc)

        try:
            objects = detector.detect(image_path, classes=object_prompts)
        except Exception as exc:
            objects = []
            stage_failures["detector"] = str(exc)
        object_positions = _object_positions(objects, image_path)

        use_vlm = _should_run_vlm_enrichment(
            segment_index=index,
            segment_count=len(segments),
            profile=settings.indexing_profile,
            sparse_stride=settings.internvl_sparse_stride,
            ocr_text=str(ocr["text"]),
            objects=objects,
        )

        branch_b: dict[str, object] = {"caption": "", "tags": [], "entities": [], "model_name": ""}
        if use_vlm:
            try:
                branch_b = branch_b_adapter.describe_image(image_path)
            except Exception as exc:
                branch_b = {"caption": "", "tags": [], "entities": [], "model_name": "error"}
                stage_failures["branch_b"] = str(exc)

        try:
            if branch_b.get("caption"):
                caption = {
                    "caption": str(branch_b["caption"]),
                    "model_name": str(branch_b.get("model_name", "internvl")),
                    "confidence": 0.9,
                }
            elif use_vlm:
                caption = captioner.caption(image_path)
            else:
                caption = {
                    "caption": _build_lightweight_caption(objects, str(ocr["text"])),
                    "model_name": "balanced-heuristic",
                    "confidence": 0.35,
                }
        except Exception as exc:
            caption = {"caption": "", "model_name": "error", "confidence": 0.0}
            stage_failures["caption"] = str(exc)

        try:
            if branch_b.get("entities"):
                normalized_entities: list[dict[str, object]] = []
                semantic_counts: dict[str, int] = {}
                for item in branch_b.get("entities", []):
                    label = str(item.get("label", "")).strip().lower()
                    if not label:
                        continue
                    aliases = [str(alias).strip().lower() for alias in item.get("aliases", []) if str(alias).strip()]
                    normalized_entities.append(
                        {
                            "label": label,
                            "count": 1,
                            "aliases": aliases,
                            "regions": [],
                            "attributes": [str(attr).strip().lower() for attr in item.get("attributes", []) if str(attr).strip()],
                        }
                    )
                    semantic_counts[label] = max(semantic_counts.get(label, 0), 1)
                    for alias in aliases:
                        semantic_counts[alias] = max(semantic_counts.get(alias, 0), 1)
                semantic = {"entities": normalized_entities, "counts": semantic_counts}
            elif use_vlm:
                semantic = entity_extractor.extract(image_path, str(caption["caption"]), str(ocr["text"]))
            else:
                semantic_counts = {label: count for label, count in _object_counts(objects).items()}
                semantic = {
                    "entities": [
                        {
                            "label": label,
                            "count": count,
                            "aliases": [],
                            "regions": list(object_positions.get(label, [])),
                            "attributes": [],
                        }
                        for label, count in semantic_counts.items()
                    ],
                    "counts": semantic_counts,
                }
        except Exception as exc:
            semantic = {"entities": [], "counts": {}}
            stage_failures["semantic_entities"] = str(exc)

        db.add(
            FrameCaption(
                frame_id=frame.id,
                model_name=str(caption["model_name"]),
                caption=str(caption["caption"]),
                confidence=float(caption.get("confidence", 1.0)) if caption.get("caption") else 0.0,
            )
        )
        db.add(
            FrameOcr(
                frame_id=frame.id,
                engine_name=ocr_engine.engine_name,
                text=str(ocr["text"]),
                raw_json={"raw": ocr["raw"]},
            )
        )
        for item in objects:
            db.add(
                FrameObject(
                    frame_id=frame.id,
                    detector_name=detector.model_name,
                    label=str(item["label"]),
                    score=float(item["score"]),
                    bbox=[float(value) for value in item["bbox"]],
                    raw_json=item,
                )
            )

        segment.caption_text = str(caption["caption"])
        segment.ocr_text = str(ocr["text"])
        segment.ocr_tokens_json = list(ocr.get("tokens", []))
        counts = _object_counts(objects)
        segment.object_labels_json = sorted(counts)
        segment.object_counts_json = counts
        segment.object_positions_json = object_positions
        segment.semantic_entities_json = list(semantic.get("entities", []))
        segment.semantic_aliases_json = {
            str(item.get("label", "")).strip().lower(): [
                str(alias).strip().lower()
                for alias in item.get("aliases", [])
                if str(alias).strip()
            ]
            for item in segment.semantic_entities_json
            if str(item.get("label", "")).strip()
        }
        segment.semantic_counts_json = dict(semantic.get("counts", {}))
        branch_b_text = _unique_text(
            [
                str(branch_b.get("caption", "")),
                *[str(tag) for tag in branch_b.get("tags", [])],
                *[str(item.get("label", "")) for item in branch_b.get("entities", [])],
            ]
        )
        if branch_b_text:
            segment.embedding_branch_b = dense_encoder.embed_text(branch_b_text).values
        segment.stage_failures_json = stage_failures
        segment.raw_json = {
            **dict(segment.raw_json or {}),
            "object_prompt_set": object_prompts,
            "object_detector_family": "yolo_world",
        }

        last_caption = str(caption["caption"])
        last_ocr = str(ocr["text"])
        last_objects = objects
        if job_id is not None:
            job = db.get(IndexJob, job_id)
            if job is not None:
                job.status = "running"
                job.stage = f"enriching_segments:{index}/{len(segments)}"
        db.commit()

    return last_caption, last_ocr, last_objects


def run_index_pipeline(db: Session, video_id: int, job_id: int | None = None) -> dict[str, object]:
    video = db.get(Video, video_id)
    if video is None:
        raise ValueError(f"video {video_id} not found")

    ensure_data_dirs()
    source_path = Path(video.source_path)
    openclip = OpenClipAdapter()
    captioner = CaptionAdapter()
    ocr_engine = PaddleOcrAdapter()
    detector = YoloDetectionAdapter()
    entity_extractor = SemanticEntityAdapter()
    branch_b_adapter = InternvlAdapter()

    _update_job_stage(db, job_id, status="running", stage="extracting_frames")
    frame_paths = keep_distinct_frames(_prepare_frame_paths(video_id, source_path), distance_threshold=8)
    if not frame_paths:
        raise RuntimeError(f"no frames extracted for video {video_id}")
    payload = index_prepared_frames(
        db,
        video_id,
        [
            {"image_path": str(frame_path), "timestamp_sec": float(index - 1), "frame_index": index}
            for index, frame_path in enumerate(frame_paths, start=1)
        ],
        openclip=openclip,
        captioner=captioner,
        ocr_engine=ocr_engine,
        detector=detector,
        entity_extractor=entity_extractor,
        branch_b_adapter=branch_b_adapter,
        job_id=job_id,
        source_tag="video_extract",
    )
    video.status = "indexed"
    if job_id is not None:
        job = db.get(IndexJob, job_id)
        if job is not None:
            job.status = "completed"
            job.stage = "done"
    db.commit()
    return {"video_id": video_id, **payload}
