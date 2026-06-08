from pathlib import Path

from app.db.models import Video


def test_video_model_has_expected_tablename() -> None:
    assert Video.__tablename__ == "videos"


def test_web_index_contains_search_form() -> None:
    contents = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="search-form"' in contents


def test_web_index_contains_file_upload_input() -> None:
    contents = Path("web/index.html").read_text(encoding="utf-8")
    assert 'type="file"' in contents


def test_web_index_contains_job_progress_panel() -> None:
    contents = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="job-status"' in contents
    assert 'id="job-progress-bar"' in contents


def test_web_index_contains_detailed_job_progress_fields() -> None:
    contents = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="job-progress-percent"' in contents
    assert 'id="job-progress-count"' in contents
    assert 'id="job-status-note"' in contents
