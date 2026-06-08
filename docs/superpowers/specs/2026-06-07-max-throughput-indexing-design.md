# Max Throughput Indexing Design

**Date:** 2026-06-07

## Goal

Increase indexing throughput as much as practical on a single GPU machine without intentionally degrading retrieval quality, by improving how the pipeline feeds the GPU rather than by reducing model quality or skipping enrich stages.

## Scope

- Keep the current model stack:
  - `OpenCLIP ViT-H-14`
  - `InternVL2_5-1B`
  - OCR
  - object detection
- Optimize pipeline execution order and concurrency.
- Add timing instrumentation for stage-level benchmarking.
- Prefer batching and overlap over multiple heavy GPU worker processes.

## Non-Goals

- No downgrade to smaller models.
- No quality-oriented stage removal.
- No distributed multi-GPU architecture.
- No change to search ranking logic in this phase.

## Problem Statement

The current indexing pipeline appears to feed the GPU in small sequential units:

- per-frame embedding work happens one item at a time
- per-segment enrich work happens one segment at a time
- CPU image preparation and DB writes can leave the GPU underfed

This means GPU utilization may remain moderate even while wall-clock indexing time is high.

## Recommended Approach

Use a throughput-first single-worker architecture with bounded concurrency:

- keep one indexing worker process
- batch `OpenCLIP` work
- prefetch and prepare image inputs on CPU ahead of GPU inference
- overlap lighter CPU-bound stages where safe
- keep only one heavy VLM inference lane active at a time unless measurement proves a wider setting helps

This aims to maximize sustained GPU throughput without destabilizing the machine through VRAM contention.

## Architecture

### 1. Stage-Level Timing

Add timing around:

- frame extraction
- frame embedding
- segment building
- OCR
- object detection
- `InternVL` description
- DB write/commit time

These timings must be visible in logs or structured debug output so the pipeline can identify the true bottleneck.

### 2. OpenCLIP Batch Embedding

`OpenCLIP` is the safest first target for batching.

The pipeline should:

- collect a batch of frame paths
- preprocess them together
- encode them in a single model call
- normalize and split outputs back into per-frame rows

This should apply to:

- image embedding during frame indexing
- text embedding where branch-B semantic text is embedded later

### 3. CPU Prefetch For Enrich

Before a GPU-heavy step runs, the pipeline should already have:

- image bytes or decoded images ready
- thumbnails prepared if needed
- lightweight metadata available in memory

This reduces GPU idle gaps caused by file I/O and image decoding.

### 4. Bounded Overlap

Allow a small amount of concurrency for lighter tasks such as:

- OCR
- object detection preparation
- image loading

But do not launch multiple uncontrolled heavy `InternVL` jobs on the same GPU in the first iteration.

The default throughput guard rails should be:

- one worker process
- one active `InternVL` lane
- configurable `OpenCLIP` batch size
- configurable CPU prefetch worker count

### 5. Config Surface

Add throughput-oriented settings such as:

- `openclip_batch_size`
- `enrich_prefetch_workers`
- `enable_stage_timing`

Defaults should aim for high throughput on a single-GPU workstation while remaining configurable.

## Risks

- Too-large `OpenCLIP` batches may spike VRAM and fail on some machines.
- CPU-side concurrency can create contention if image preprocessing becomes too aggressive.
- If `InternVL` batching is not naturally supported by the adapter, trying to force it in the first pass may create instability for limited speed gain.

## Acceptance Criteria

- The pipeline emits stage timing data that identifies time spent per major sub-stage.
- `OpenCLIP` frame embedding is batched instead of one-image-at-a-time.
- The machine does not rely on multiple heavy GPU worker processes for throughput.
- Indexing throughput measurably improves on the same clip/model configuration without intentionally reducing enrich quality.
