# Video Retrieval System Design

**Date:** 2026-06-02

## Goal

Build a production-oriented multimodal video retrieval system that stays as close as practical to the VBS 2025 paper direction while still running reliably on a single GPU server. The system must support offline indexing, multimodal search, object filtering, temporal refinement, and OpenAI-powered query expansion, with an isolated local environment that does not conflict with the user's global Python setup.

## Scope

This spec covers the first full system architecture for:

- video ingest and indexing
- keyframe extraction and storage
- embedding, caption, OCR, and object extraction
- multimodal retrieval and score fusion
- temporal reranking
- minimal web UI
- Docker Compose deployment on one GPU server
- isolated local development environment

This spec does not include:

- user authentication
- multi-tenant management
- distributed multi-GPU scheduling
- Stable Diffusion visual query generation in the first release
- full competition-only tuning workflows

## Recommended Approach

The system should use a balanced production architecture:

- `FastAPI` for the core backend
- `Postgres + pgvector` for metadata and vector storage
- `Docker Compose` for deployability on one GPU server
- a separate indexing worker for GPU-heavy offline processing
- adapter-based ML components so models can be swapped later
- `OpenAI API` for query expansion

This is the best tradeoff between:

- staying close to the paper's multimodal design
- keeping operations simple enough for a real project
- preserving a path to later competition-grade upgrades

## Architecture

The system consists of six primary parts:

1. `web`
Minimal browser UI for upload, indexing control, text search, object filtering, result browsing, and timeline preview.

2. `api`
FastAPI service that handles search requests, result fusion, query expansion, filtering, and orchestration.

3. `worker`
Background processing service that performs frame extraction, keyframe selection, embedding, captioning, OCR, object detection, and persistence.

4. `postgres`
Single relational database used for metadata, inference artifacts, jobs, and vector search through `pgvector`.

5. `storage`
Local mounted directories for source videos, frames, thumbnails, logs, and cached model outputs.

6. `openai integration`
Used for query expansion in online retrieval. The architecture must keep this integration isolated behind a dedicated service interface.

## Core Design Principles

- Keep offline indexing and online retrieval strictly separated.
- Do not bind retrieval quality to a single model.
- Normalize all model outputs into stable internal contracts.
- Prefer explicit metadata tables over opaque blobs for searchable fields.
- Preserve raw model outputs for debugging and future reprocessing.
- Make every ML component replaceable without breaking API contracts.
- Optimize first for correctness and operability, then for benchmark tuning.

## Data Flow

### Offline Indexing Flow

1. User uploads or registers a video.
2. The API creates an `index_job`.
3. The worker extracts frames at `1 fps`.
4. The sampling stage removes near-duplicate frames.
5. Selected keyframes are stored as image files and WebP thumbnails.
6. The worker runs:
   - image embedding
   - caption generation
   - OCR
   - object detection
7. The worker writes artifacts and metadata into Postgres.
8. The job status is updated for monitoring and retry handling.

### Online Retrieval Flow

1. User submits a text query and optional filters.
2. The API sends the query to the query expansion module.
3. The system generates multiple expanded queries using OpenAI.
4. The API runs vector search for each query variant.
5. The API evaluates auxiliary matches from caption, OCR, and object metadata.
6. The fusion module normalizes and combines scores.
7. The temporal search module reranks nearby frames or segments when applicable.
8. The API returns grouped results with previews and timeline context.

## Services And Responsibilities

### Web

Responsibilities:

- upload video or register local file
- trigger indexing
- monitor job state
- submit search query
- apply object and metadata filters
- view result thumbnails
- inspect timeline around a result
- switch between basic mode and advanced mode

The first release UI should stay minimal and operational, not stylistically elaborate.

### API

Responsibilities:

- expose REST endpoints for ingest, jobs, search, and result inspection
- validate input
- call query expansion
- run retrieval and fusion
- expose result grouping and timeline preview
- return structured diagnostics for advanced mode

The API should be stateless and should not perform heavy offline inference work directly.

### Worker

Responsibilities:

- consume indexing jobs
- extract frames
- remove redundant frames
- generate thumbnails
- run ML inference stages
- persist outputs
- retry failed stages safely

The worker is the main GPU consumer and must be deployable with NVIDIA runtime support.

## ML Component Contracts

Every model-backed module must be accessed through an adapter interface.

### Embedding Adapter

Input:
- image path or in-memory image
- text query

Output:
- normalized embedding vector
- model name

Initial backend:
- `OpenCLIP`

### Caption Adapter

Input:
- image path

Output:
- caption text
- optional confidence
- model name

Initial backend:
- local VLM backend

### OCR Adapter

Input:
- image path

Output:
- extracted text
- optional structured OCR data
- engine name

Initial backend:
- `PaddleOCR`

### Object Detection Adapter

Input:
- image path

Output:
- list of objects with `label`, `score`, `bbox`
- detector name

Initial backend:
- `YOLO`

The object detection contract must remain backend-neutral so future upgrades to stronger detectors do not require retrieval-layer rewrites.

### Query Expansion Adapter

Input:
- original user query
- optional retrieval context

Output:
- list of expanded queries
- provider name

Initial backend:
- `OpenAI API`

## Database Design

The system should use Postgres with `pgvector` enabled.

### `videos`

- `id`
- `filename`
- `source_path`
- `duration_sec`
- `fps`
- `status`
- `created_at`
- `updated_at`

### `frames`

- `id`
- `video_id`
- `timestamp_sec`
- `frame_index`
- `image_path`
- `thumb_path`
- `is_keyframe`
- `created_at`

### `frame_embeddings`

- `id`
- `frame_id`
- `model_name`
- `embedding`
- `created_at`

### `frame_captions`

- `id`
- `frame_id`
- `model_name`
- `caption`
- `confidence`
- `raw_json`
- `created_at`

### `frame_ocr`

- `id`
- `frame_id`
- `engine_name`
- `text`
- `raw_json`
- `created_at`

### `frame_objects`

- `id`
- `frame_id`
- `detector_name`
- `label`
- `score`
- `bbox`
- `raw_json`
- `created_at`

### `index_jobs`

- `id`
- `video_id`
- `status`
- `stage`
- `error_message`
- `attempt_count`
- `created_at`
- `updated_at`

### `query_logs`

- `id`
- `original_query`
- `expanded_queries`
- `filters_json`
- `weights_json`
- `top_results_json`
- `created_at`

## Retrieval Strategy

The retrieval layer should follow the paper's core philosophy: combine multiple weak-to-strong signals instead of trusting one model path.

The first release should include:

- text-to-image vector retrieval over keyframes
- caption text matching
- OCR text matching
- object presence filtering
- result fusion
- temporal reranking

Initial fusion should support configurable weights for:

- embedding score
- caption score
- OCR score
- object score
- temporal score

These weights must be surfaced in advanced mode so the system can later be tuned for competition use cases without rewriting core logic.

## Temporal Search

Temporal search should be implemented as a reranking stage instead of a separate full search engine.

Expected behavior:

- start from strong initial matches
- inspect nearby timestamps from the same video
- score local temporal neighborhoods against follow-up or adjacent query intent
- promote sequences that maintain semantic coherence across nearby frames

The first version does not need full sequence reasoning, but it must define a clean module boundary so future multi-stage temporal refinement can be added.

## Sampling Strategy

The initial indexing strategy should use:

- frame extraction at `1 fps`
- near-duplicate filtering
- WebP thumbnails for efficient browsing

This is intentionally chosen as a practical baseline. The module must be isolated so a future upgrade to shot-boundary detection can replace or augment the current sampler without changing the rest of the indexing flow.

## UI Requirements

The first release UI must support:

- video upload or registration
- indexing trigger
- job status display
- search box
- object filter input
- result grid
- grouped results by video or nearby timestamps
- frame preview
- timeline context around a selected result
- advanced mode with score weight controls

The first release does not need:

- authentication
- multi-user roles
- extensive analytics dashboards

## Environment Isolation

The project must not rely on the user's global Python or Conda environments.

Local development requirements:

- create a project-local `.venv`
- pin Python-compatible dependency versions
- use repo-local scripts or `make` targets for setup and run flows
- keep secrets in `.env`

Deployment requirements:

- all services run through `Docker Compose`
- Python dependencies are installed inside service images
- GPU access is explicitly configured for the worker

## GPU Strategy

The detected hardware is a single `NVIDIA GeForce RTX 5060` with driver support available through `nvidia-smi`.

The system should therefore:

- assign heavy inference to the worker service
- keep the API lightweight
- use CUDA-enabled PyTorch builds in the isolated environment
- verify GPU access through explicit health or smoke tests

The design must assume one GPU and prevent accidental reliance on the global host environment.

## Error Handling

The system must handle:

- failed video ingest
- partial inference failures
- bad OCR or detector outputs
- OpenAI query expansion timeouts or failures
- retryable worker crashes
- corrupted or unsupported media files

Design rules:

- each indexing stage records status
- failures must be visible through job APIs
- non-critical enrichment failures should be captured without silently corrupting the full index
- query expansion failure must degrade gracefully to the original query

## Testing Strategy

The implementation must include:

- unit tests for adapters and fusion logic
- integration tests for ingest and retrieval flows
- GPU smoke tests for containerized inference
- regression fixtures with a small sample dataset

Success is not just "containers run"; success means:

- indexing completes end-to-end
- a stored video becomes searchable
- GPU inference executes inside the isolated runtime
- the search API returns stable structured results

## Deployment Strategy

The first deployment target is a single GPU server using Docker Compose.

Required services:

- `postgres`
- `api`
- `worker`
- `web`

Required mounts:

- source videos
- extracted frames
- thumbnails
- optional model cache

The deployment should be simple enough for one-machine operation but structured so later extraction of services is possible.

## Future Extensions

The architecture must leave room for:

- stronger object detectors
- multiple embedding backends and score fusion
- shot-boundary detection
- sequence-aware temporal search
- Stable Diffusion or other generated visual query methods
- competition tuning workflows

These are future capabilities, not first-release obligations.

## Acceptance Criteria

The design is considered satisfied when the implementation can:

- run in isolated local and containerized environments
- use the machine GPU for worker inference
- ingest and index a video end-to-end
- persist multimodal artifacts in Postgres with `pgvector`
- accept a text query and return ranked video frame results
- apply object filtering
- perform OpenAI-powered query expansion
- expose timeline-aware result browsing in the minimal UI

## Implementation Direction

The implementation should start with a production-oriented first release rather than a throwaway prototype. That first release should optimize for:

- clean module boundaries
- reproducible environment setup
- stable indexing and retrieval flows
- observability and debuggability
- practical closeness to the paper's retrieval philosophy
