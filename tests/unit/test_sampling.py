from pathlib import Path

from worker.sampling import keep_distinct_frames


def test_keep_distinct_frames_removes_adjacent_duplicates(tmp_path: Path) -> None:
    frames = [
        tmp_path / "frame_0001.jpg",
        tmp_path / "frame_0002.jpg",
        tmp_path / "frame_0003.jpg",
    ]
    frames[0].write_bytes(b"same")
    frames[1].write_bytes(b"same")
    frames[2].write_bytes(b"different")

    kept = keep_distinct_frames(frames, distance_threshold=0)

    assert kept == [frames[0], frames[2]]
