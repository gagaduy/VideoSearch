from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.exc import OperationalError

from app.db.models import Frame, FrameCaption, FrameEmbedding, FrameObject, FrameOcr, IndexJob, QueryLog, Segment, Video
from app.db.session import get_engine, get_session_factory, init_db
from app.main import app
from app.services import object_refinement
from app.services import question_search
from app.services import search_service
from worker import pipeline


def _reset_search_api_state() -> None:
    search_service._SEARCH_DENSE_ENCODER = None
    session_factory = get_session_factory()
    session = session_factory()
    try:
        session.execute(delete(QueryLog))
        session.execute(delete(IndexJob))
        session.execute(delete(FrameObject))
        session.execute(delete(FrameOcr))
        session.execute(delete(FrameCaption))
        session.execute(delete(FrameEmbedding))
        session.execute(delete(Frame))
        session.execute(delete(Segment))
        session.execute(delete(Video))
        session.commit()
    except OperationalError:
        session.close()
        get_session_factory.cache_clear()
        get_engine.cache_clear()
        init_db()
        session = get_session_factory()()
        session.execute(delete(QueryLog))
        session.execute(delete(IndexJob))
        session.execute(delete(FrameObject))
        session.execute(delete(FrameOcr))
        session.execute(delete(FrameCaption))
        session.execute(delete(FrameEmbedding))
        session.execute(delete(Frame))
        session.execute(delete(Segment))
        session.execute(delete(Video))
        session.commit()
    finally:
        session.close()


def _stub_indexing_runtime(monkeypatch, tmp_path: Path) -> None:
    frame_one = tmp_path / "frame_000001.png"
    frame_two = tmp_path / "frame_000002.png"
    frame_one.write_bytes(b"frame-one")
    frame_two.write_bytes(b"frame-two")

    monkeypatch.setattr(pipeline, "_prepare_frame_paths", lambda video_id, source_path: [frame_one, frame_two])
    monkeypatch.setattr(pipeline, "keep_distinct_frames", lambda frames, distance_threshold: frames)
    monkeypatch.setattr(
        pipeline,
        "copy_or_create_thumbnail",
        lambda image_path, thumb_path: thumb_path.parent.mkdir(parents=True, exist_ok=True) or thumb_path.write_bytes(b"thumb") or thumb_path,
    )

    class _OpenClip:
        def embed_image(self, image_path: str) -> object:
            return type("Embedding", (), {"model_name": "stub", "values": [0.1, 0.2, 0.3]})()

        def embed_images(self, image_paths: list[str]) -> list[object]:
            return [self.embed_image(image_path) for image_path in image_paths]

        def embed_text(self, text: str) -> object:
            return type("Embedding", (), {"model_name": "stub", "values": [0.4, 0.5, 0.6]})()

    class _Caption:
        def caption(self, image_path: str) -> dict[str, object]:
            return {"caption": f"caption {Path(image_path).name}", "model_name": "stub"}

    class _Ocr:
        engine_name = "stub-ocr"

        def extract_text(self, image_path: str) -> dict[str, object]:
            return {"text": "frame placeholder car", "tokens": ["frame", "placeholder", "car"], "raw": []}

    class _Detector:
        model_name = "stub-yolo-world"

        def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
            label = "car" if classes and "car" in classes else "person"
            return [{"label": label, "matched_prompt": label, "score": 0.9, "bbox": [0, 0, 1, 1]}]

    class _InternVL:
        def describe_image(self, image_path: str) -> dict[str, object]:
            return {
                "caption": "car near sign",
                "tags": ["car", "sign"],
                "entities": [{"label": "car", "aliases": ["automobile"]}],
                "model_name": "stub-internvl",
            }

    monkeypatch.setattr(pipeline, "OpenClipAdapter", _OpenClip)
    monkeypatch.setattr(pipeline, "CaptionAdapter", _Caption)
    monkeypatch.setattr(pipeline, "PaddleOcrAdapter", _Ocr)
    monkeypatch.setattr(pipeline, "build_object_detector", lambda: _Detector())
    monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)
    monkeypatch.setattr(search_service, "run_openai_vision_rerank", lambda query, candidates, query_image_paths=None: {})
    monkeypatch.setattr(question_search, "run_openai_vision_rerank", lambda query, candidates, query_image_paths=None: {})


def test_search_endpoint_returns_ranked_results(monkeypatch, tmp_path: Path) -> None:
    _reset_search_api_state()
    _stub_indexing_runtime(monkeypatch, tmp_path)
    client = TestClient(app)
    created = client.post(
        "/videos",
        json={"filename": "search.mp4", "source_path": "./data/videos/search.mp4"},
    )
    job_id = created.json()["job"]["id"]
    client.post(f"/jobs/{job_id}/run")
    response = client.post(
        "/search",
        json={"query": "frame placeholder", "object_labels": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "frame placeholder"
    assert isinstance(payload["results"], list)
    assert payload["results"]
    assert payload["results"][0]["thumb_url"].startswith("/media/frames/")
    assert payload["results"][0]["image_url"].startswith("/media/frames/")
    assert payload["results"][0]["preview_url"].startswith("/media/frames/")
    assert isinstance(payload["results"][0]["object_counts"], dict)
    assert "object_refinement_score" in payload["results"][0]["diagnostics"]
    assert payload["results"][0]["diagnostics"]["object_detector_family"] == search_service.settings.object_detector_family
    assert payload["results"][0]["diagnostics"]["object_detector_model"] == "stub-yolo-world"


def test_search_endpoint_applies_query_conditioned_object_refinement(monkeypatch, tmp_path: Path) -> None:
    _reset_search_api_state()
    _stub_indexing_runtime(monkeypatch, tmp_path)

    class _RefinementDetector:
        def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
            if classes == ["car"]:
                return [{"label": "car", "matched_prompt": "car", "score": 0.95, "bbox": [0, 0, 1, 1]}]
            return []

    monkeypatch.setattr(object_refinement, "YoloDetectionAdapter", _RefinementDetector)

    client = TestClient(app)
    created = client.post(
        "/videos",
        json={"filename": "search-refine.mp4", "source_path": "./data/videos/search-refine.mp4"},
    )
    job_id = created.json()["job"]["id"]
    client.post(f"/jobs/{job_id}/run")
    response = client.post(
        "/search",
        json={"query": "find automobile", "object_labels": ["automobile"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    assert payload["results"][0]["diagnostics"]["object_refinement_score"] == 0.95


def test_image_search_returns_ranked_results(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "app.api.routes.search.run_image_search",
        lambda db, image_path: {
            "mode": "image",
            "query": "query.png",
            "expanded_queries": [],
            "parsed_query": None,
            "results": [{"segment_id": 7, "score": 0.51}],
        },
    )

    response = client.post(
        "/search/image",
        files={"file": ("query.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "image"
    assert response.json()["results"][0]["segment_id"] == 7


def test_image_search_rejects_non_image_upload() -> None:
    client = TestClient(app)

    response = client.post(
        "/search/image",
        files={"file": ("query.txt", b"not-image", "text/plain")},
    )

    assert response.status_code == 400


def test_video_query_search_route_is_registered() -> None:
    client = TestClient(app)

    response = client.options(
        "/search/video-query",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code != 404


def test_video_query_search_rejects_too_long_clip(tmp_path: Path) -> None:
    client = TestClient(app)
    clip_path = tmp_path / "query.mp4"
    clip_path.write_bytes(b"fake-video")

    response = client.post(
        "/search/video-query",
        files={"file": ("query.mp4", clip_path.read_bytes(), "video/mp4")},
    )

    assert response.status_code in {400, 422}


def test_video_query_search_returns_results(monkeypatch, tmp_path: Path) -> None:
    client = TestClient(app)
    clip_path = tmp_path / "query.mp4"
    clip_path.write_bytes(b"fake-video")

    monkeypatch.setattr(
        "app.api.routes.search.run_video_query_search",
        lambda db, upload, **_kwargs: {
            "mode": "video",
            "query": "query.mp4",
            "expanded_queries": [],
            "results": [{"frame_id": 11, "score": 0.9, "thumb_url": "/media/frames/11/thumb"}],
            "parsed_query": None,
        },
        raising=False,
    )

    response = client.post(
        "/search/video-query",
        files={"file": ("query.mp4", clip_path.read_bytes(), "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "video"
    assert response.json()["results"][0]["frame_id"] == 11


def test_question_search_route_is_registered() -> None:
    client = TestClient(app)
    response = client.options(
        "/search/question",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code != 404


def test_question_search_returns_results(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "app.api.routes.search.run_question_search",
        lambda db, question, **_kwargs: {
            "mode": "question",
            "query": question,
            "expanded_queries": [],
            "results": [{"frame_id": 11, "score": 0.81, "thumb_url": "/media/frames/11/thumb"}],
            "parsed_query": None,
        },
        raising=False,
    )

    response = client.post("/search/question", json={"question": "What is the virus name?"})

    assert response.status_code == 200
    assert response.json()["mode"] == "question"
    assert response.json()["results"][0]["frame_id"] == 11


def test_question_search_endpoint_returns_ranked_results(monkeypatch, tmp_path: Path) -> None:
    client = TestClient(app)

    frame_path = tmp_path / "question-frame.png"
    thumb_path = tmp_path / "question-thumb.png"
    frame_path.write_bytes(b"frame")
    thumb_path.write_bytes(b"thumb")

    monkeypatch.setattr(
        question_search,
        "_load_segment_rows",
        lambda db: [
            {
                "segment_id": 1,
                "video_id": 1,
                "segment_index": 0,
                "start_timestamp_sec": 0.0,
                "end_timestamp_sec": 2.0,
                "keyframe_id": 11,
                "caption_text": "screen with vaccine text",
                "ocr_text": "virus vaccine information",
                "labels": [],
                "object_counts": {},
                "object_positions": {},
                "semantic_entities": [],
                "semantic_counts": {},
            }
        ],
    )
    monkeypatch.setattr(
        question_search,
        "fetch_frame_media_map",
        lambda db, frame_ids: {
            11: {
                "image_path": str(frame_path),
                "thumb_path": str(thumb_path),
            }
        },
    )

    response = client.post("/search/question", json={"question": "What virus vaccine information is shown?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "question"
    assert payload["results"]


def test_search_endpoint_returns_results_when_openai_rerank_falls_back(monkeypatch, tmp_path: Path) -> None:
    _reset_search_api_state()
    _stub_indexing_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr("app.services.search_service.run_openai_vision_rerank", lambda query, candidates: {})

    client = TestClient(app)
    created = client.post(
        "/videos",
        json={"filename": "search-rerank.mp4", "source_path": "./data/videos/search-rerank.mp4"},
    )
    job_id = created.json()["job"]["id"]
    client.post(f"/jobs/{job_id}/run")

    response = client.post("/search", json={"query": "frame placeholder", "object_labels": []})

    assert response.status_code == 200
    assert response.json()["results"]


def test_run_scripts_exist() -> None:
    assert Path("scripts/run_api.sh").exists()
    assert Path("scripts/run_worker.sh").exists()


def test_search_endpoint_allows_cors_preflight() -> None:
    client = TestClient(app)
    response = client.options(
        "/search",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8080"


def test_media_endpoint_serves_frame_thumbnail(monkeypatch, tmp_path: Path) -> None:
    _reset_search_api_state()
    _stub_indexing_runtime(monkeypatch, tmp_path)
    client = TestClient(app)
    created = client.post(
        "/videos",
        json={"filename": "search-thumb.mp4", "source_path": "./data/videos/search-thumb.mp4"},
    )
    job_id = created.json()["job"]["id"]
    client.post(f"/jobs/{job_id}/run")
    search = client.post("/search", json={"query": "frame placeholder", "object_labels": []})
    thumb_url = search.json()["results"][0]["thumb_url"]

    response = client.get(thumb_url)

    assert response.status_code == 200
    assert response.headers["content-type"] in {"image/webp", "image/png", "application/octet-stream"}


def test_media_endpoint_serves_segment_preview_clip(monkeypatch, tmp_path: Path) -> None:
    _reset_search_api_state()
    _stub_indexing_runtime(monkeypatch, tmp_path)
    client = TestClient(app)
    created = client.post(
        "/videos",
        json={"filename": "search-preview.mp4", "source_path": "./data/videos/search-preview.mp4"},
    )
    job_id = created.json()["job"]["id"]
    client.post(f"/jobs/{job_id}/run")
    search = client.post("/search", json={"query": "frame placeholder", "object_labels": []})
    preview_url = search.json()["results"][0]["preview_url"]

    preview_file = tmp_path / "preview.mp4"
    preview_file.write_bytes(b"preview")

    from app.api.routes import media as media_routes

    monkeypatch.setattr(media_routes, "_build_preview_clip", lambda *args, **kwargs: preview_file, raising=False)

    response = client.get(preview_url)

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["cache-control"] == "no-store"
