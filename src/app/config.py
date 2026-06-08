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
    indexing_profile: str = "full"
    internvl_model: str = "OpenGVLab/InternVL2_5-1B"
    internvl_use_8bit: bool = True
    internvl_max_new_tokens: int = 96
    internvl_sparse_stride: int = 6
    embedding_commit_interval: int = 12
    enrich_commit_interval: int = 2
    openclip_batch_size: int = 16
    enrich_prefetch_workers: int = 2
    enable_stage_timing: bool = True
    yolo_prompt_profile: str = "compact"
    yolo_confidence_threshold: float = 0.2
    yolo_max_detections: int = 12
    yolo_model: str = "yolov8s-worldv2.pt"
    worker_poll_interval_sec: int = 2
    search_result_display_limit: int = 16
    text_result_score_threshold: float = 0.12
    image_result_score_threshold: float = 0.2
    video_query_max_duration_sec: float = 10.0
    video_query_frame_count: int = 6
    video_query_local_candidate_pool: int = 24
    question_search_candidate_pool: int = 24
    question_search_rerank_top_k: int = 8
    openai_vision_rerank_enabled: bool = True
    openai_vision_rerank_top_k: int = 8
    openai_vision_rerank_timeout_sec: int = 8
    openai_vision_rerank_local_weight: float = 0.45
    openai_vision_rerank_vision_weight: float = 0.55
    openai_vision_rerank_model: str = "gpt-4.1-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
