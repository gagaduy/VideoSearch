from pathlib import Path

from app.config import Settings


def test_settings_defaults_use_local_paths(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path / "data",
        videos_dir=tmp_path / "data" / "videos",
        frames_dir=tmp_path / "data" / "frames",
        thumbs_dir=tmp_path / "data" / "thumbs",
    )

    assert settings.app_env == "test"
    assert settings.data_dir == tmp_path / "data"
    assert settings.videos_dir == tmp_path / "data" / "videos"
    assert settings.frames_dir == tmp_path / "data" / "frames"
    assert settings.thumbs_dir == tmp_path / "data" / "thumbs"
    assert settings.internvl_model == "OpenGVLab/InternVL2_5-1B"
    assert settings.internvl_use_8bit is True
    assert settings.yolo_model == "yolo11l.pt"
