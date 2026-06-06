from pathlib import Path

from app.config import settings


def ensure_data_dirs() -> None:
    for path in (settings.data_dir, settings.videos_dir, settings.frames_dir, settings.thumbs_dir, settings.previews_dir):
        Path(path).mkdir(parents=True, exist_ok=True)


def reserve_video_path(filename: str) -> Path:
    ensure_data_dirs()
    target = Path(settings.videos_dir) / Path(filename).name
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = target.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
