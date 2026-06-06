from pathlib import Path

from worker.io import copy_or_create_thumbnail, extract_frames_ffmpeg, write_placeholder_image
from worker.sampling import keep_distinct_frames


def test_write_placeholder_image_creates_file(tmp_path: Path) -> None:
    image_path = write_placeholder_image(tmp_path / "frame.png")

    assert image_path.exists()
    assert image_path.read_bytes()


def test_keep_distinct_frames_removes_identical_adjacent_frames(tmp_path: Path) -> None:
    first = write_placeholder_image(tmp_path / "frame_0001.png")
    second = tmp_path / "frame_0002.png"
    second.write_bytes(first.read_bytes())
    third = tmp_path / "frame_0003.png"
    third.write_bytes(first.read_bytes() + b"different")

    kept = keep_distinct_frames([first, second, third], distance_threshold=0)

    assert kept == [first, third]


def test_copy_or_create_thumbnail_writes_output(tmp_path: Path) -> None:
    source = write_placeholder_image(tmp_path / "source.png")
    thumb = copy_or_create_thumbnail(source, tmp_path / "thumb.webp")

    assert thumb.exists()


def test_extract_frames_ffmpeg_falls_back_to_placeholder_when_video_invalid(tmp_path: Path) -> None:
    bad_video = tmp_path / "broken.mp4"
    bad_video.write_bytes(b"not a real video")

    frames = extract_frames_ffmpeg(bad_video, tmp_path / "frames")

    assert len(frames) == 1
    assert frames[0].exists()
