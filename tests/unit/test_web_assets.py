from pathlib import Path


def test_web_app_contains_progress_helpers() -> None:
    contents = Path("web/app.js").read_text(encoding="utf-8")
    assert "function parseJobProgress(job)" in contents
    assert "function describeJobNote(job)" in contents
    assert "function formatRetrievalScore(value)" in contents


def test_web_app_uses_relative_match_label() -> None:
    contents = Path("web/app.js").read_text(encoding="utf-8")
    assert "relative match=" in contents


def test_web_app_contains_idle_progress_guard() -> None:
    contents = Path("web/app.js").read_text(encoding="utf-8")
    assert "function syncIdleJobStatus()" in contents


def test_index_contains_video_query_form() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="video-query-form"' in html
    assert 'id="query-video"' in html
    assert "Search By Video Clip" in html
    assert "Search By Image" not in html


def test_index_uses_workspace_shell_layout() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'class="workspace"' in html
    assert 'class="left-rail' in html


def test_index_uses_compact_sidebar_markup() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'class="left-rail compact-sidebar"' in html
    assert 'class="hero panel hero-compact"' in html


def test_index_uses_split_workspace_for_results_and_preview() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'class="workspace-main split-workspace"' in html
    assert 'class="workspace-results panel"' in html
    assert 'class="workspace-preview panel"' in html


def test_app_blocks_duplicate_text_search_submit() -> None:
    script = Path("web/app.js").read_text(encoding="utf-8")
    assert "let textSearchInFlight = false;" in script
    assert "if (textSearchInFlight) {" in script
    assert 'submitButton.textContent = "Searching...";' in script


def test_app_renders_empty_state_message() -> None:
    script = Path("web/app.js").read_text(encoding="utf-8")
    assert "No strong matches found for this query." in script


def test_app_contains_video_query_request_guard() -> None:
    script = Path("web/app.js").read_text(encoding="utf-8")
    assert "let videoQueryInFlight = false;" in script
    assert 'fetch(`${apiBase}/search/video-query`' in script


def test_index_contains_query_video_preview_slot() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="query-video-preview"' in html
    assert 'class="query-media-preview-slot query-media-preview-compact"' in html


def test_index_contains_question_search_panel() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "Search By Question" in html
    assert 'id="question-search-form"' in html
    assert 'id="question-query"' in html


def test_app_contains_question_search_request_guard() -> None:
    script = Path("web/app.js").read_text(encoding="utf-8")
    assert "let questionSearchInFlight = false;" in script
    assert 'fetch(`${apiBase}/search/question`' in script
