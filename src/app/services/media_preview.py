from pathlib import Path
from subprocess import run

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


def preview_cache_path(video_id: int, segment_id: int | None, frame_id: int) -> Path:
    ensure_data_dirs()
    preview_dir = Path(settings.previews_dir) / f"video_{video_id}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = f"segment_{segment_id}" if segment_id is not None else f"frame_{frame_id}"
    return preview_dir / f"{stem}.mp4"


def build_preview_clip(video: Video, frame: Frame, segment: Segment | None) -> Path:
    source_path = Path(video.source_path).resolve()
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="source video not found")

    target = preview_cache_path(int(video.id), frame.segment_id, int(frame.id))
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
