from fastapi.testclient import TestClient

import app.main as app_main
from app.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_lifespan_prewarms_search_runtime(monkeypatch) -> None:
    called = {"value": False}

    def _prewarm() -> None:
        called["value"] = True

    monkeypatch.setattr(app_main, "prewarm_search_runtime", _prewarm)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert called["value"] is True
