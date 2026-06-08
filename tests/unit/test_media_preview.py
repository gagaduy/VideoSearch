from pathlib import Path

from app.db.models import Frame, Video
from app.services import media_preview


def test_preview_cache_path_changes_when_source_video_changes(tmp_path: Path) -> None:
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_bytes(b"first-video")
    second_video.write_bytes(b"second-video")

    video_one = Video(id=1, filename="first.mp4", source_path=str(first_video))
    video_two = Video(id=1, filename="second.mp4", source_path=str(second_video))
    frame = Frame(id=11, video_id=1, segment_id=4, timestamp_sec=10.0, frame_index=11, image_path="a.png", thumb_path="a.webp")

    first_path = media_preview.preview_cache_path(video_one, frame)
    second_path = media_preview.preview_cache_path(video_two, frame)

    assert first_path != second_path
    assert "segment_4" in first_path.name
    assert "segment_4" in second_path.name
