from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from worker.tasks import run_pending_jobs


def test_create_video_job_returns_pending_job() -> None:
    client = TestClient(app)
    response = client.post(
        "/videos",
        json={"filename": "demo.mp4", "source_path": "./data/videos/demo.mp4"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["video"]["filename"] == "demo.mp4"
    assert payload["job"]["status"] == "pending"

    run_response = client.post(f"/jobs/{payload['job']['id']}/run")
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "completed"


def test_index_job_flow_records_frame_artifacts() -> None:
    client = TestClient(app)
    created = client.post(
        "/videos",
        json={"filename": "demo-2.mp4", "source_path": "./data/videos/demo-2.mp4"},
    )
    job_id = created.json()["job"]["id"]
    client.post(f"/jobs/{job_id}/run")
    search = client.post("/search", json={"query": "frame placeholder", "object_labels": []})
    artifacts = search.json()["results"][0]

    assert "caption" in artifacts
    assert "object_labels" in artifacts


def test_upload_video_file_creates_index_job() -> None:
    for leftover in Path("data/videos").glob("upload-demo*.mp4"):
        leftover.unlink()

    client = TestClient(app)
    response = client.post(
        "/videos/upload",
        files={"file": ("upload-demo.mp4", b"fake video bytes", "video/mp4")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["video"]["filename"] == "upload-demo.mp4"
    assert payload["video"]["source_path"].endswith("upload-demo.mp4")
    assert payload["job"]["status"] == "pending"


def test_worker_processes_pending_uploaded_job() -> None:
    for leftover in Path("data/videos").glob("auto-run*.mp4"):
        leftover.unlink()

    client = TestClient(app)
    response = client.post(
        "/videos/upload",
        files={"file": ("auto-run.mp4", b"fake video bytes", "video/mp4")},
    )

    assert response.status_code == 201
    payload = response.json()
    job_id = payload["job"]["id"]

    processed = run_pending_jobs()
    job_response = client.get(f"/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_id in processed["processed_job_ids"]
    assert job_response.json()["status"] == "completed"
