from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import object_refinement
from worker import pipeline


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
    monkeypatch.setattr(pipeline, "YoloDetectionAdapter", _Detector)
    monkeypatch.setattr(pipeline, "InternvlAdapter", _InternVL)


def test_search_endpoint_returns_ranked_results(monkeypatch, tmp_path: Path) -> None:
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


def test_search_endpoint_applies_query_conditioned_object_refinement(monkeypatch, tmp_path: Path) -> None:
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
