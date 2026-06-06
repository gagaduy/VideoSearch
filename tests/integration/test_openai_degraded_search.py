from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_search_without_openai_returns_ranked_results(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "")

    client = TestClient(app)
    created = client.post(
        "/videos",
        json={"filename": "degraded-search.mp4", "source_path": "./data/videos/degraded-search.mp4"},
    )
    job_id = created.json()["job"]["id"]
    client.post(f"/jobs/{job_id}/run")

    response = client.post("/search", json={"query": "red boat with text", "object_labels": []})
    payload = response.json()

    assert response.status_code == 200
    assert payload["results"]
    assert payload["expanded_queries"] == ["red boat with text"]
