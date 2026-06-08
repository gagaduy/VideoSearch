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
    assert settings.indexing_profile == "full"
    assert settings.internvl_model == "OpenGVLab/InternVL2_5-1B"
    assert settings.internvl_use_8bit is True
    assert settings.internvl_max_new_tokens == 96
    assert settings.internvl_sparse_stride == 6
    assert settings.embedding_commit_interval == 12
    assert settings.enrich_commit_interval == 2
    assert settings.openclip_batch_size == 16
    assert settings.enrich_prefetch_workers == 2
    assert settings.enable_stage_timing is True
    assert settings.yolo_prompt_profile == "compact"
    assert settings.yolo_confidence_threshold == 0.2
    assert settings.yolo_max_detections == 12
    assert settings.yolo_model == "yolov8s-worldv2.pt"


def test_result_display_defaults() -> None:
    settings = Settings()

    assert settings.search_result_display_limit == 16
    assert settings.text_result_score_threshold == 0.12
    assert settings.image_result_score_threshold == 0.2


def test_openai_vision_rerank_defaults() -> None:
    settings = Settings()

    assert settings.openai_vision_rerank_enabled is True
    assert settings.openai_vision_rerank_top_k == 8
    assert settings.openai_vision_rerank_timeout_sec == 8
    assert settings.openai_vision_rerank_local_weight == 0.45
    assert settings.openai_vision_rerank_model == "gpt-4.1-mini"
    assert settings.openai_vision_rerank_vision_weight == 0.55


def test_video_query_search_defaults() -> None:
    settings = Settings()

    assert settings.video_query_max_duration_sec == 10.0
    assert settings.video_query_frame_count == 6
    assert settings.video_query_local_candidate_pool == 24


def test_question_search_defaults() -> None:
    settings = Settings()

    assert settings.question_search_candidate_pool == 24
    assert settings.question_search_rerank_top_k == 8
