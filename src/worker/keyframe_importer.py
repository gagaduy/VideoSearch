from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Video
from app.services.storage import ensure_data_dirs
from worker.adapters.caption_adapter import CaptionAdapter
from worker.adapters.internvl_adapter import InternvlAdapter
from worker.adapters.openclip_adapter import OpenClipAdapter
from worker.adapters.paddleocr_adapter import PaddleOcrAdapter
from worker.adapters.semantic_entity_adapter import SemanticEntityAdapter
from worker.adapters.yolo_adapter import YoloDetectionAdapter
from worker.pipeline import _build_caption_adapter, index_prepared_frames


@dataclass(slots=True)
class KeyframeVideoSource:
    video_code: str
    group_code: str
    frame_dir: Path
    frame_paths: list[Path]
    metadata_path: Path | None
    metadata: dict[str, object]
    duration_sec: float | None


def _discover_metadata_dirs(dataset_root: Path) -> list[Path]:
    return [path for path in dataset_root.glob("media-info*/media-info") if path.is_dir()]


def _metadata_map(dataset_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for directory in _discover_metadata_dirs(dataset_root):
        for path in sorted(directory.glob("*.json")):
            mapping[path.stem] = path
    return mapping


def discover_keyframe_videos(dataset_root: str | Path) -> list[KeyframeVideoSource]:
    root = Path(dataset_root)
    metadata_paths = _metadata_map(root)
    discovered: list[KeyframeVideoSource] = []
    keyframe_root = root / "keyframe"
    for group_dir in sorted(path for path in keyframe_root.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            frame_paths = sorted(
                [path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            )
            if not frame_paths:
                continue
            metadata_path = metadata_paths.get(video_dir.name)
            metadata = json.loads(metadata_path.read_text()) if metadata_path is not None else {}
            length = metadata.get("length")
            discovered.append(
                KeyframeVideoSource(
                    video_code=video_dir.name,
                    group_code=group_dir.name,
                    frame_dir=video_dir,
                    frame_paths=frame_paths,
                    metadata_path=metadata_path,
                    metadata=metadata,
                    duration_sec=float(length) if isinstance(length, (int, float)) else None,
                )
            )
    return discovered


def _upsert_video(db: Session, source: KeyframeVideoSource) -> Video:
    video = db.execute(
        select(Video).where(Video.source_path == str(source.frame_dir))
    ).scalar_one_or_none()
    if video is None:
        video = Video(
            filename=f"{source.video_code}.keyframes",
            source_path=str(source.frame_dir),
            duration_sec=source.duration_sec,
            fps=(len(source.frame_paths) / source.duration_sec) if source.duration_sec and source.duration_sec > 0 else None,
            status="pending",
        )
        db.add(video)
        db.flush()
        return video

    video.filename = f"{source.video_code}.keyframes"
    video.duration_sec = source.duration_sec
    video.fps = (len(source.frame_paths) / source.duration_sec) if source.duration_sec and source.duration_sec > 0 else None
    video.status = "pending"
    db.flush()
    return video


def import_keyframe_dataset(db: Session, dataset_root: str | Path) -> dict[str, object]:
    ensure_data_dirs()
    sources = discover_keyframe_videos(dataset_root)
    openclip = OpenClipAdapter()
    ocr_engine = PaddleOcrAdapter()
    detector = YoloDetectionAdapter()
    entity_extractor = SemanticEntityAdapter()
    branch_b_adapter = InternvlAdapter()
    captioner = _build_caption_adapter(branch_b_adapter)

    imported_videos = 0
    imported_frames = 0
    imported_segments = 0

    for source in sources:
        video = _upsert_video(db, source)
        duration = source.duration_sec or float(max(len(source.frame_paths), 1))
        step = (duration / max(len(source.frame_paths), 1)) if len(source.frame_paths) > 1 else 0.0
        payload = index_prepared_frames(
            db,
            int(video.id),
            [
                {
                    "image_path": str(frame_path),
                    "timestamp_sec": float(index * step),
                    "frame_index": index + 1,
                }
                for index, frame_path in enumerate(source.frame_paths)
            ],
            openclip=openclip,
            captioner=captioner,
            ocr_engine=ocr_engine,
            detector=detector,
            entity_extractor=entity_extractor,
            branch_b_adapter=branch_b_adapter,
            source_tag="keyframe_import",
        )
        video.status = "indexed"
        imported_videos += 1
        imported_frames += int(payload["frame_count"])
        imported_segments += int(payload["segment_count"])
        db.commit()

    return {
        "video_count": imported_videos,
        "frame_count": imported_frames,
        "segment_count": imported_segments,
        "dataset_root": str(Path(dataset_root)),
    }
