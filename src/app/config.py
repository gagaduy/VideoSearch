from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    database_url: str = "postgresql+psycopg://video:video@localhost:5432/video_retrieval"
    data_dir: Path = Field(default=Path("./data"))
    videos_dir: Path = Field(default=Path("./data/videos"))
    frames_dir: Path = Field(default=Path("./data/frames"))
    thumbs_dir: Path = Field(default=Path("./data/thumbs"))
    previews_dir: Path = Field(default=Path("./data/previews"))
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    indexing_profile: str = "balanced"
    internvl_model: str = "OpenGVLab/InternVL2_5-1B"
    internvl_use_8bit: bool = True
    internvl_max_new_tokens: int = 192
    internvl_sparse_stride: int = 3
    yolo_model: str = "yolov8s-worldv2.pt"
    worker_poll_interval_sec: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
