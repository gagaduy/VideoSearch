from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_search_endpoint_returns_ranked_results() -> None:
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


def test_media_endpoint_serves_frame_thumbnail() -> None:
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
