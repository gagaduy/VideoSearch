import base64
import shutil
import subprocess
from pathlib import Path


PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2pZ4kAAAAASUVORK5CYII="
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_placeholder_image(path: Path) -> Path:
    ensure_parent(path)
    path.write_bytes(PLACEHOLDER_PNG)
    return path


def extract_frames_ffmpeg(video_path: Path, output_dir: Path, fps: float = 1.0) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%06d.png"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps}",
                str(output_pattern),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        return [write_placeholder_image(output_dir / "frame_000001.png")]

    frames = sorted(output_dir.glob("frame_*.png"))
    if frames:
        return frames
    return [write_placeholder_image(output_dir / "frame_000001.png")]


def copy_or_create_thumbnail(image_path: Path, thumb_path: Path) -> Path:
    ensure_parent(thumb_path)
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.thumbnail((320, 320))
            image.save(thumb_path, format="WEBP")
        return thumb_path
    except Exception:
        shutil.copyfile(image_path, thumb_path)
        return thumb_path
