from pathlib import Path
from subprocess import run
import hashlib

from fastapi import HTTPException

from app.config import settings
from app.db.models import Frame, Segment, Video
from app.services.storage import ensure_data_dirs

PREVIEW_PADDING_SEC = 1.5


def _preview_bounds(frame: Frame, segment: Segment | None) -> tuple[float, float]:
    if segment is None:
        start_sec = max(frame.timestamp_sec - PREVIEW_PADDING_SEC, 0.0)
        end_sec = frame.timestamp_sec + PREVIEW_PADDING_SEC
        return start_sec, end_sec

    start_sec = max(float(segment.start_timestamp_sec) - PREVIEW_PADDING_SEC, 0.0)
    end_sec = max(float(segment.end_timestamp_sec) + PREVIEW_PADDING_SEC, start_sec + 1.0)
    return start_sec, end_sec


def _preview_source_fingerprint(video: Video) -> str:
    source_path = Path(video.source_path).resolve()
    try:
        stats = source_path.stat()
        payload = f"{source_path}:{stats.st_size}:{stats.st_mtime_ns}"
    except FileNotFoundError:
        payload = str(source_path)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def preview_cache_path(video: Video, frame: Frame) -> Path:
    ensure_data_dirs()
    preview_dir = Path(settings.previews_dir) / f"video_{int(video.id)}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprint = _preview_source_fingerprint(video)
    stem = f"segment_{frame.segment_id}" if frame.segment_id is not None else f"frame_{int(frame.id)}"
    stem = f"{stem}_{source_fingerprint}"
    return preview_dir / f"{stem}.mp4"


def build_preview_clip(video: Video, frame: Frame, segment: Segment | None) -> Path:
    source_path = Path(video.source_path).resolve()
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="source video not found")

    target = preview_cache_path(video, frame)
    if target.exists():
        return target

    start_sec, end_sec = _preview_bounds(frame, segment)
    duration_sec = max(end_sec - start_sec, 1.0)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_sec:.3f}",
        "-vf",
        "scale='min(1280,iw)':-2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        str(target),
    ]
    completed = run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not target.exists():
        raise HTTPException(status_code=500, detail="preview clip generation failed")
    return target
