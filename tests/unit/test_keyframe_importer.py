import json
from pathlib import Path

from worker.keyframe_importer import discover_keyframe_videos


def test_discover_keyframe_videos_reads_dataset_layout(tmp_path: Path) -> None:
    frame_dir = tmp_path / "keyframe" / "L01" / "L01_V001"
    frame_dir.mkdir(parents=True)
    (frame_dir / "001.jpg").write_bytes(b"frame-1")
    (frame_dir / "002.jpg").write_bytes(b"frame-2")

    metadata_dir = tmp_path / "media-info-b1" / "media-info"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "L01_V001.json").write_text(
        json.dumps(
            {
                "title": "Video 1",
                "author": "Author",
                "length": 12.0,
                "publish_date": "2024-01-01",
            }
        )
    )

    videos = discover_keyframe_videos(tmp_path)

    assert len(videos) == 1
    assert videos[0].video_code == "L01_V001"
    assert videos[0].group_code == "L01"
    assert [path.name for path in videos[0].frame_paths] == ["001.jpg", "002.jpg"]
    assert videos[0].metadata["title"] == "Video 1"
    assert videos[0].duration_sec == 12.0
