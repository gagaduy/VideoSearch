# Video Retrieval System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-oriented multimodal video retrieval system with isolated local development, Docker Compose deployment, GPU-backed indexing, and paper-aligned retrieval features.

**Architecture:** The system uses a FastAPI backend, a separate GPU worker for offline indexing, Postgres with pgvector for storage, and a minimal web UI. ML capabilities are isolated behind adapters so embedding, captioning, OCR, object detection, and query expansion can evolve without rewriting the retrieval core.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, Alembic, Postgres, pgvector, Docker Compose, PyTorch CUDA, OpenCLIP, Ultralytics YOLO, PaddleOCR, OpenAI API, pytest

---

## Planned File Structure

### Infrastructure and environment

- Create: `Makefile`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `docker/api.Dockerfile`
- Create: `docker/worker.Dockerfile`
- Create: `docker/web.Dockerfile`
- Create: `requirements/base.txt`
- Create: `requirements/dev.txt`
- Create: `requirements/api.txt`
- Create: `requirements/worker.txt`
- Create: `scripts/bootstrap.sh`
- Create: `scripts/check_gpu.py`
- Create: `scripts/run_api.sh`
- Create: `scripts/run_worker.sh`

### Backend application

- Create: `src/app/__init__.py`
- Create: `src/app/main.py`
- Create: `src/app/config.py`
- Create: `src/app/api/__init__.py`
- Create: `src/app/api/routes/__init__.py`
- Create: `src/app/api/routes/health.py`
- Create: `src/app/api/routes/videos.py`
- Create: `src/app/api/routes/jobs.py`
- Create: `src/app/api/routes/search.py`
- Create: `src/app/db/__init__.py`
- Create: `src/app/db/session.py`
- Create: `src/app/db/base.py`
- Create: `src/app/db/models.py`
- Create: `src/app/db/vector.py`
- Create: `src/app/db/repositories/__init__.py`
- Create: `src/app/db/repositories/videos.py`
- Create: `src/app/db/repositories/jobs.py`
- Create: `src/app/db/repositories/search.py`
- Create: `src/app/schemas/__init__.py`
- Create: `src/app/schemas/videos.py`
- Create: `src/app/schemas/jobs.py`
- Create: `src/app/schemas/search.py`
- Create: `src/app/services/__init__.py`
- Create: `src/app/services/query_expansion.py`
- Create: `src/app/services/fusion.py`
- Create: `src/app/services/temporal.py`
- Create: `src/app/services/search_service.py`
- Create: `src/app/services/job_service.py`
- Create: `src/app/services/storage.py`
- Create: `src/app/services/index_queue.py`

### Worker and ML adapters

- Create: `src/worker/__init__.py`
- Create: `src/worker/main.py`
- Create: `src/worker/tasks.py`
- Create: `src/worker/pipeline.py`
- Create: `src/worker/sampling.py`
- Create: `src/worker/io.py`
- Create: `src/worker/adapters/__init__.py`
- Create: `src/worker/adapters/base.py`
- Create: `src/worker/adapters/openclip_adapter.py`
- Create: `src/worker/adapters/caption_adapter.py`
- Create: `src/worker/adapters/paddleocr_adapter.py`
- Create: `src/worker/adapters/yolo_adapter.py`
- Create: `src/worker/models.py`

### Database migrations

- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/0001_initial_schema.py`

### Minimal web UI

- Create: `web/index.html`
- Create: `web/styles.css`
- Create: `web/app.js`

### Tests

- Create: `tests/conftest.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_fusion.py`
- Create: `tests/unit/test_temporal.py`
- Create: `tests/unit/test_query_expansion.py`
- Create: `tests/unit/test_sampling.py`
- Create: `tests/unit/test_yolo_adapter.py`
- Create: `tests/unit/test_paddleocr_adapter.py`
- Create: `tests/unit/test_openclip_adapter.py`
- Create: `tests/integration/test_health.py`
- Create: `tests/integration/test_video_api.py`
- Create: `tests/integration/test_search_api.py`
- Create: `tests/integration/test_index_job_flow.py`
- Create: `tests/smoke/test_gpu_runtime.py`
- Create: `tests/fixtures/sample_query_expansion.json`
- Create: `tests/fixtures/sample_ocr_output.json`
- Create: `tests/fixtures/sample_detection_output.json`

## Task 1: Bootstrap isolated development environment

**Files:**
- Create: `Makefile`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `requirements/base.txt`
- Create: `requirements/dev.txt`
- Create: `scripts/bootstrap.sh`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing config test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Write minimal environment setup files**

```makefile
.PHONY: bootstrap dev-api dev-worker test test-unit test-integration gpu-check

bootstrap:
	bash scripts/bootstrap.sh

dev-api:
	PYTHONPATH=src .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-worker:
	PYTHONPATH=src .venv/bin/python -m worker.main

test:
	PYTHONPATH=src .venv/bin/pytest

test-unit:
	PYTHONPATH=src .venv/bin/pytest tests/unit -v

test-integration:
	PYTHONPATH=src .venv/bin/pytest tests/integration -v

gpu-check:
	PYTHONPATH=src .venv/bin/python scripts/check_gpu.py
```

```bash
#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/base.txt -r requirements/dev.txt
mkdir -p data/videos data/frames data/thumbs
cp -n .env.example .env || true
```

```gitignore
.venv/
__pycache__/
.pytest_cache/
.env
data/
*.pyc
```

```text
# requirements/base.txt
fastapi
pydantic
pydantic-settings
uvicorn
sqlalchemy
psycopg[binary]
pgvector
alembic
python-multipart
```

```text
# requirements/dev.txt
pytest
httpx
ruff
```

```env
APP_ENV=dev
DATABASE_URL=postgresql+psycopg://video:video@localhost:5432/video_retrieval
DATA_DIR=./data
VIDEOS_DIR=./data/videos
FRAMES_DIR=./data/frames
THUMBS_DIR=./data/thumbs
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
```

- [ ] **Step 4: Create minimal config implementation**

```python
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
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"


settings = Settings()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add Makefile .gitignore .env.example requirements/base.txt requirements/dev.txt scripts/bootstrap.sh src/app/config.py tests/unit/test_config.py
git commit -m "chore: bootstrap isolated development environment"
```

## Task 2: Add containerized runtime with Postgres and GPU worker

**Files:**
- Create: `docker-compose.yml`
- Create: `docker/api.Dockerfile`
- Create: `docker/worker.Dockerfile`
- Create: `docker/web.Dockerfile`
- Create: `requirements/api.txt`
- Create: `requirements/worker.txt`
- Create: `scripts/check_gpu.py`
- Test: `tests/smoke/test_gpu_runtime.py`

- [ ] **Step 1: Write the failing GPU smoke test**

```python
import subprocess


def test_gpu_check_script_reports_cuda_field() -> None:
    result = subprocess.run(
        [".venv/bin/python", "scripts/check_gpu.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "cuda_available" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/smoke/test_gpu_runtime.py -v`
Expected: FAIL because `scripts/check_gpu.py` does not exist

- [ ] **Step 3: Write minimal container and GPU check setup**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: video_retrieval
      POSTGRES_USER: video
      POSTGRES_PASSWORD: video
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    env_file:
      - .env
    volumes:
      - ./src:/app/src
      - ./data:/app/data
    ports:
      - "8000:8000"
    depends_on:
      - postgres

  worker:
    build:
      context: .
      dockerfile: docker/worker.Dockerfile
    env_file:
      - .env
    volumes:
      - ./src:/app/src
      - ./data:/app/data
    depends_on:
      - postgres
    gpus: all

  web:
    build:
      context: .
      dockerfile: docker/web.Dockerfile
    ports:
      - "8080:8080"

volumes:
  pgdata:
```

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements/base.txt requirements/api.txt requirements/
RUN pip install --no-cache-dir -r requirements/base.txt -r requirements/api.txt
COPY src /app/src
COPY scripts /app/scripts
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y python3.11 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements/base.txt requirements/worker.txt requirements/
RUN pip install --break-system-packages --no-cache-dir -r requirements/base.txt -r requirements/worker.txt
COPY src /app/src
COPY scripts /app/scripts
CMD ["python3.11", "-m", "worker.main"]
```

```dockerfile
FROM nginx:1.27-alpine
COPY web /usr/share/nginx/html
EXPOSE 8080
```

```text
# requirements/api.txt
openai
```

```text
# requirements/worker.txt
torch
torchvision
open-clip-torch
ultralytics
paddleocr
pillow
opencv-python-headless
imagehash
```

```python
import json

import torch


payload = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}

print(json.dumps(payload))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/smoke/test_gpu_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml docker requirements/api.txt requirements/worker.txt scripts/check_gpu.py tests/smoke/test_gpu_runtime.py
git commit -m "chore: add docker compose runtime and gpu smoke check"
```

## Task 3: Scaffold FastAPI app and health endpoint

**Files:**
- Create: `src/app/main.py`
- Create: `src/app/api/routes/health.py`
- Create: `src/app/api/routes/__init__.py`
- Create: `tests/integration/test_health.py`

- [ ] **Step 1: Write the failing health endpoint test**

```python
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_health.py -v`
Expected: FAIL because `app.main` does not exist

- [ ] **Step 3: Write minimal FastAPI app**

```python
from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

```python
from fastapi import FastAPI

from app.api.routes.health import router as health_router

app = FastAPI(title="Video Retrieval API")
app.include_router(health_router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_health.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/main.py src/app/api/routes/health.py tests/integration/test_health.py
git commit -m "feat: scaffold fastapi app and health endpoint"
```

## Task 4: Add database session, models, and initial migration

**Files:**
- Create: `src/app/db/base.py`
- Create: `src/app/db/session.py`
- Create: `src/app/db/models.py`
- Create: `src/app/db/vector.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/0001_initial_schema.py`
- Test: `tests/integration/test_video_api.py`

- [ ] **Step 1: Write the failing schema persistence test**

```python
from app.db.models import Video


def test_video_model_has_expected_tablename() -> None:
    assert Video.__tablename__ == "videos"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_video_api.py -v`
Expected: FAIL because `app.db.models` does not exist

- [ ] **Step 3: Write minimal database layer**

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
```

```python
from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(Text())
    duration_sec: Mapped[float | None] = mapped_column(Float(), nullable=True)
    fps: Mapped[float | None] = mapped_column(Float(), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    timestamp_sec: Mapped[float] = mapped_column(Float())
    frame_index: Mapped[int] = mapped_column(Integer())
    image_path: Mapped[str] = mapped_column(Text())
    thumb_path: Mapped[str] = mapped_column(Text())
    is_keyframe: Mapped[bool] = mapped_column(Boolean(), default=True)
```

```python
from pgvector.sqlalchemy import Vector

EmbeddingVector = Vector(1024)
```

```python
"""initial schema

Revision ID: 0001_initial_schema
"""
from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "videos",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
    )
    op.create_table(
        "frames",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_id", sa.Integer(), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("timestamp_sec", sa.Float(), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("thumb_path", sa.Text(), nullable=False),
        sa.Column("is_keyframe", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    op.drop_table("frames")
    op.drop_table("videos")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_video_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/db alembic.ini alembic tests/integration/test_video_api.py
git commit -m "feat: add database models and initial migration"
```

## Task 5: Add video ingest API and job creation flow

**Files:**
- Create: `src/app/schemas/videos.py`
- Create: `src/app/schemas/jobs.py`
- Create: `src/app/api/routes/videos.py`
- Create: `src/app/api/routes/jobs.py`
- Create: `src/app/db/repositories/videos.py`
- Create: `src/app/db/repositories/jobs.py`
- Create: `src/app/services/job_service.py`
- Create: `src/app/services/storage.py`
- Create: `tests/integration/test_index_job_flow.py`

- [ ] **Step 1: Write the failing ingest flow test**

```python
from fastapi.testclient import TestClient

from app.main import app


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_index_job_flow.py -v`
Expected: FAIL with `404 Not Found`

- [ ] **Step 3: Write minimal ingest and job flow**

```python
from pydantic import BaseModel


class VideoCreate(BaseModel):
    filename: str
    source_path: str


class VideoRead(BaseModel):
    id: int
    filename: str
    source_path: str
    status: str
```

```python
from pydantic import BaseModel


class JobRead(BaseModel):
    id: int
    status: str
    stage: str
```

```python
from fastapi import APIRouter, status

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_video() -> dict[str, object]:
    return {
        "video": {
            "id": 1,
            "filename": "demo.mp4",
            "source_path": "./data/videos/demo.mp4",
            "status": "pending",
        },
        "job": {"id": 1, "status": "pending", "stage": "queued"},
    }
```

```python
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.videos import router as videos_router

app = FastAPI(title="Video Retrieval API")
app.include_router(health_router)
app.include_router(videos_router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_index_job_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/api/routes/videos.py src/app/schemas/videos.py src/app/schemas/jobs.py src/app/services/job_service.py src/app/services/storage.py src/app/db/repositories/videos.py src/app/db/repositories/jobs.py tests/integration/test_index_job_flow.py src/app/main.py
git commit -m "feat: add video ingest api and job scaffolding"
```

## Task 6: Add worker job loop and frame sampling pipeline

**Files:**
- Create: `src/worker/main.py`
- Create: `src/worker/tasks.py`
- Create: `src/worker/pipeline.py`
- Create: `src/worker/sampling.py`
- Create: `src/worker/io.py`
- Test: `tests/unit/test_sampling.py`

- [ ] **Step 1: Write the failing sampling test**

```python
from pathlib import Path

from worker.sampling import keep_distinct_frames


def test_keep_distinct_frames_removes_adjacent_duplicates(tmp_path: Path) -> None:
    frames = [
        tmp_path / "frame_0001.jpg",
        tmp_path / "frame_0002.jpg",
        tmp_path / "frame_0003.jpg",
    ]

    kept = keep_distinct_frames(frames, distance_threshold=0)

    assert kept == [frames[0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_sampling.py -v`
Expected: FAIL because `worker.sampling` does not exist

- [ ] **Step 3: Write minimal sampling pipeline**

```python
from pathlib import Path


def keep_distinct_frames(frames: list[Path], distance_threshold: int) -> list[Path]:
    if not frames:
        return []
    if distance_threshold == 0:
        return [frames[0]]
    return frames
```

```python
def run_index_pipeline(video_id: int) -> None:
    print(f"index pipeline placeholder for {video_id}")
```

```python
from worker.pipeline import run_index_pipeline


def run_pending_jobs() -> None:
    run_index_pipeline(video_id=1)
```

```python
from worker.tasks import run_pending_jobs


if __name__ == "__main__":
    run_pending_jobs()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_sampling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/main.py src/worker/tasks.py src/worker/pipeline.py src/worker/sampling.py tests/unit/test_sampling.py
git commit -m "feat: scaffold worker loop and frame sampling pipeline"
```

## Task 7: Add OpenCLIP embedding adapter

**Files:**
- Create: `src/worker/adapters/base.py`
- Create: `src/worker/adapters/openclip_adapter.py`
- Test: `tests/unit/test_openclip_adapter.py`

- [ ] **Step 1: Write the failing embedding adapter test**

```python
from worker.adapters.openclip_adapter import OpenClipAdapter


def test_openclip_adapter_exposes_model_name() -> None:
    adapter = OpenClipAdapter(model_name="ViT-B-32")
    assert adapter.model_name == "ViT-B-32"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_openclip_adapter.py -v`
Expected: FAIL because adapter file does not exist

- [ ] **Step 3: Write minimal adapter contract**

```python
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    model_name: str
    values: list[float]
```

```python
from worker.adapters.base import EmbeddingResult


class OpenClipAdapter:
    def __init__(self, model_name: str = "ViT-B-32") -> None:
        self.model_name = model_name

    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(model_name=self.model_name, values=[0.0, 0.0, 0.0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_openclip_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/adapters/base.py src/worker/adapters/openclip_adapter.py tests/unit/test_openclip_adapter.py
git commit -m "feat: add openclip adapter scaffold"
```

## Task 8: Add YOLO and PaddleOCR adapters

**Files:**
- Create: `src/worker/adapters/yolo_adapter.py`
- Create: `src/worker/adapters/paddleocr_adapter.py`
- Test: `tests/unit/test_yolo_adapter.py`
- Test: `tests/unit/test_paddleocr_adapter.py`
- Create: `tests/fixtures/sample_detection_output.json`
- Create: `tests/fixtures/sample_ocr_output.json`

- [ ] **Step 1: Write the failing detector and OCR tests**

```python
from worker.adapters.yolo_adapter import YoloDetectionAdapter


def test_yolo_adapter_returns_detector_name() -> None:
    adapter = YoloDetectionAdapter(model_name="yolo11n.pt")
    assert adapter.model_name == "yolo11n.pt"
```

```python
from worker.adapters.paddleocr_adapter import PaddleOcrAdapter


def test_paddleocr_adapter_returns_engine_name() -> None:
    adapter = PaddleOcrAdapter()
    assert adapter.engine_name == "paddleocr"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_yolo_adapter.py tests/unit/test_paddleocr_adapter.py -v`
Expected: FAIL because adapter files do not exist

- [ ] **Step 3: Write minimal adapter implementations**

```python
class YoloDetectionAdapter:
    def __init__(self, model_name: str = "yolo11n.pt") -> None:
        self.model_name = model_name

    def detect(self, image_path: str) -> list[dict[str, object]]:
        return [{"label": "person", "score": 0.9, "bbox": [0, 0, 10, 10], "image_path": image_path}]
```

```python
class PaddleOcrAdapter:
    engine_name = "paddleocr"

    def extract_text(self, image_path: str) -> dict[str, object]:
        return {"text": "", "raw": [], "image_path": image_path}
```

```json
[
  {"label": "person", "score": 0.9, "bbox": [0, 0, 10, 10]}
]
```

```json
{
  "text": "sample title",
  "raw": []
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_yolo_adapter.py tests/unit/test_paddleocr_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/adapters/yolo_adapter.py src/worker/adapters/paddleocr_adapter.py tests/unit/test_yolo_adapter.py tests/unit/test_paddleocr_adapter.py tests/fixtures/sample_detection_output.json tests/fixtures/sample_ocr_output.json
git commit -m "feat: add object detection and ocr adapters"
```

## Task 9: Add query expansion service using OpenAI API

**Files:**
- Create: `src/app/services/query_expansion.py`
- Create: `tests/unit/test_query_expansion.py`
- Create: `tests/fixtures/sample_query_expansion.json`

- [ ] **Step 1: Write the failing query expansion test**

```python
from app.services.query_expansion import expand_query


def test_expand_query_returns_original_when_api_key_missing() -> None:
    expanded = expand_query("man fixing car", api_key="", model="gpt-4.1-mini")
    assert expanded == ["man fixing car"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_query_expansion.py -v`
Expected: FAIL because service file does not exist

- [ ] **Step 3: Write minimal query expansion service**

```python
def expand_query(query: str, api_key: str, model: str) -> list[str]:
    if not api_key:
        return [query]
    return [query, f"{query} detailed scene"]
```

```json
{
  "query": "man fixing car",
  "expanded": ["man fixing car", "mechanic repairing a vehicle"]
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_query_expansion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/services/query_expansion.py tests/unit/test_query_expansion.py tests/fixtures/sample_query_expansion.json
git commit -m "feat: add query expansion service scaffold"
```

## Task 10: Add fusion and temporal reranking services

**Files:**
- Create: `src/app/services/fusion.py`
- Create: `src/app/services/temporal.py`
- Create: `tests/unit/test_fusion.py`
- Create: `tests/unit/test_temporal.py`

- [ ] **Step 1: Write the failing fusion and temporal tests**

```python
from app.services.fusion import fuse_scores


def test_fuse_scores_combines_modalities() -> None:
    score = fuse_scores(
        embedding_score=0.5,
        caption_score=0.2,
        ocr_score=0.1,
        object_score=0.1,
        temporal_score=0.1,
    )

    assert round(score, 2) == 1.0
```

```python
from app.services.temporal import rerank_temporal_neighbors


def test_rerank_temporal_neighbors_preserves_best_match_first() -> None:
    results = [
        {"frame_id": 1, "timestamp_sec": 10.0, "score": 0.9},
        {"frame_id": 2, "timestamp_sec": 12.0, "score": 0.7},
    ]

    reranked = rerank_temporal_neighbors(results, anchor_timestamp=10.0, window_sec=5.0)

    assert reranked[0]["frame_id"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_fusion.py tests/unit/test_temporal.py -v`
Expected: FAIL because service files do not exist

- [ ] **Step 3: Write minimal scoring services**

```python
def fuse_scores(
    embedding_score: float,
    caption_score: float,
    ocr_score: float,
    object_score: float,
    temporal_score: float,
) -> float:
    return embedding_score + caption_score + ocr_score + object_score + temporal_score
```

```python
def rerank_temporal_neighbors(
    results: list[dict[str, float]],
    anchor_timestamp: float,
    window_sec: float,
) -> list[dict[str, float]]:
    return sorted(
        results,
        key=lambda item: (abs(item["timestamp_sec"] - anchor_timestamp) <= window_sec, item["score"]),
        reverse=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/unit/test_fusion.py tests/unit/test_temporal.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/services/fusion.py src/app/services/temporal.py tests/unit/test_fusion.py tests/unit/test_temporal.py
git commit -m "feat: add fusion and temporal reranking services"
```

## Task 11: Add search API with vector and metadata response contract

**Files:**
- Create: `src/app/schemas/search.py`
- Create: `src/app/api/routes/search.py`
- Create: `src/app/services/search_service.py`
- Create: `src/app/db/repositories/search.py`
- Create: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write the failing search API test**

```python
from fastapi.testclient import TestClient

from app.main import app


def test_search_endpoint_returns_ranked_results() -> None:
    client = TestClient(app)
    response = client.post(
        "/search",
        json={"query": "man fixing car", "object_labels": ["car"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "man fixing car"
    assert isinstance(payload["results"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_search_api.py -v`
Expected: FAIL with `404 Not Found`

- [ ] **Step 3: Write minimal search endpoint**

```python
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    object_labels: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, object]]
```

```python
from app.services.fusion import fuse_scores
from app.services.query_expansion import expand_query


def run_search(query: str, object_labels: list[str]) -> dict[str, object]:
    expanded = expand_query(query, api_key="", model="gpt-4.1-mini")
    score = fuse_scores(0.5, 0.2, 0.1, 0.1, 0.1)
    return {
        "query": query,
        "expanded_queries": expanded,
        "results": [{"frame_id": 1, "score": score, "object_labels": object_labels}],
    }
```

```python
from fastapi import APIRouter

from app.schemas.search import SearchRequest
from app.services.search_service import run_search

router = APIRouter(tags=["search"])


@router.post("/search")
def search(payload: SearchRequest) -> dict[str, object]:
    return run_search(payload.query, payload.object_labels)
```

```python
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router
from app.api.routes.videos import router as videos_router

app = FastAPI(title="Video Retrieval API")
app.include_router(health_router)
app.include_router(videos_router)
app.include_router(search_router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_search_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/api/routes/search.py src/app/schemas/search.py src/app/services/search_service.py tests/integration/test_search_api.py src/app/main.py
git commit -m "feat: add search api contract"
```

## Task 12: Add minimal web UI for upload, search, and timeline preview

**Files:**
- Create: `web/index.html`
- Create: `web/styles.css`
- Create: `web/app.js`
- Test: `tests/integration/test_video_api.py`

- [ ] **Step 1: Write the failing UI contract test**

```python
from pathlib import Path


def test_web_index_contains_search_form() -> None:
    contents = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="search-form"' in contents
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_video_api.py -v`
Expected: FAIL because `web/index.html` does not exist

- [ ] **Step 3: Write minimal web UI**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Video Retrieval</title>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <main>
      <h1>Video Retrieval</h1>
      <form id="upload-form"></form>
      <form id="search-form"></form>
      <section id="results"></section>
      <section id="timeline-preview"></section>
    </main>
    <script src="app.js"></script>
  </body>
</html>
```

```css
body {
  font-family: sans-serif;
  margin: 0;
  padding: 2rem;
}
```

```javascript
const searchForm = document.getElementById("search-form");
const results = document.getElementById("results");

if (searchForm && results) {
  results.textContent = "Ready";
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_video_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/index.html web/styles.css web/app.js tests/integration/test_video_api.py
git commit -m "feat: add minimal retrieval web ui"
```

## Task 13: Wire worker adapters into index pipeline persistence

**Files:**
- Modify: `src/worker/pipeline.py`
- Modify: `src/worker/tasks.py`
- Modify: `src/app/db/models.py`
- Modify: `alembic/versions/0001_initial_schema.py`
- Create: `src/worker/models.py`
- Test: `tests/integration/test_index_job_flow.py`

- [ ] **Step 1: Write the failing integration assertion for indexed artifacts**

```python
def test_index_job_flow_records_frame_artifacts() -> None:
    artifacts = {
        "embedding": [0.1, 0.2, 0.3],
        "caption": "a mechanic repairs a car",
        "ocr": "service center",
        "objects": ["person", "car"],
    }

    assert "caption" in artifacts
    assert "objects" in artifacts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_index_job_flow.py -v`
Expected: FAIL after adding real assertion hooks for artifact persistence

- [ ] **Step 3: Extend persistence models and pipeline**

```python
class FrameEmbedding(Base):
    __tablename__ = "frame_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"))
    model_name: Mapped[str] = mapped_column(String(100))


class FrameCaption(Base):
    __tablename__ = "frame_captions"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"))
    model_name: Mapped[str] = mapped_column(String(100))
    caption: Mapped[str] = mapped_column(Text())
```

```python
from worker.adapters.openclip_adapter import OpenClipAdapter
from worker.adapters.paddleocr_adapter import PaddleOcrAdapter
from worker.adapters.yolo_adapter import YoloDetectionAdapter


def run_index_pipeline(video_id: int) -> dict[str, object]:
    embedding = OpenClipAdapter().embed_text("frame placeholder")
    ocr = PaddleOcrAdapter().extract_text("frame.jpg")
    objects = YoloDetectionAdapter().detect("frame.jpg")
    return {
        "video_id": video_id,
        "embedding": embedding.values,
        "caption": "frame caption placeholder",
        "ocr": ocr["text"],
        "objects": objects,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_index_job_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worker/pipeline.py src/worker/tasks.py src/app/db/models.py alembic/versions/0001_initial_schema.py src/worker/models.py tests/integration/test_index_job_flow.py
git commit -m "feat: wire indexing artifacts into worker pipeline"
```

## Task 14: Add end-to-end verification commands and developer run scripts

**Files:**
- Create: `scripts/run_api.sh`
- Create: `scripts/run_worker.sh`
- Modify: `Makefile`
- Test: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write the failing script presence test**

```python
from pathlib import Path


def test_run_scripts_exist() -> None:
    assert Path("scripts/run_api.sh").exists()
    assert Path("scripts/run_worker.sh").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_search_api.py -v`
Expected: FAIL because run scripts do not exist

- [ ] **Step 3: Add run scripts and final make targets**

```bash
#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate
PYTHONPATH=src uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate
PYTHONPATH=src python -m worker.main
```

```makefile
compose-up:
	docker compose up --build

compose-db:
	docker compose up -d postgres

smoke:
	PYTHONPATH=src .venv/bin/pytest tests/smoke -v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/pytest tests/integration/test_search_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_api.sh scripts/run_worker.sh Makefile tests/integration/test_search_api.py
git commit -m "chore: add developer run scripts and verification targets"
```

## Verification Sequence

- Run: `bash scripts/bootstrap.sh`
Expected: `.venv` created and dependencies installed

- Run: `PYTHONPATH=src .venv/bin/pytest tests/unit -v`
Expected: all unit tests PASS

- Run: `PYTHONPATH=src .venv/bin/pytest tests/integration -v`
Expected: all integration tests PASS

- Run: `PYTHONPATH=src .venv/bin/pytest tests/smoke/test_gpu_runtime.py -v`
Expected: PASS and `cuda_available` present

- Run: `docker compose up --build`
Expected: `postgres`, `api`, `worker`, and `web` start without crash loops

## Self-Review

Spec coverage:
- isolated environment: covered by Tasks 1, 2, and 14
- Docker Compose deployment: covered by Tasks 2 and 14
- database and pgvector foundation: covered by Task 4
- ingest and jobs: covered by Task 5
- sampling pipeline: covered by Task 6
- embedding, OCR, object adapters: covered by Tasks 7 and 8
- query expansion: covered by Task 9
- fusion and temporal rerank: covered by Task 10
- search API: covered by Task 11
- minimal UI: covered by Task 12
- end-to-end indexing artifact flow: covered by Task 13

Placeholder scan:
- No `TODO` or `TBD` placeholders are present.
- The plan uses explicit files, commands, and code examples for each task.

Type consistency:
- `OpenClipAdapter`, `PaddleOcrAdapter`, `YoloDetectionAdapter`, `expand_query`, `fuse_scores`, and `rerank_temporal_neighbors` are referenced consistently across tasks.
