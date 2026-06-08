# Enrich Throughput Without Quality Tradeoff

**Date:** 2026-06-07

## Goal

Reduce end-to-end enrich time for indexing without intentionally lowering retrieval quality, by improving how enrich stages are scheduled and fed rather than by removing or downgrading enrich outputs.

## Scope

- Keep the current enrich stack:
  - `InternVL2_5-1B`
  - OCR
  - object detection
  - semantic enrichment outputs
- Measure enrich sub-stage timing explicitly.
- Optimize enrich throughput with prefetch and bounded overlap.
- Preserve current enrich behavior and output semantics as much as possible.

## Non-Goals

- No downgrade to lighter models.
- No switch from `full` back to `local` for quality reasons in this phase.
- No dropping OCR or detector as a shortcut.
- No multi-GPU or distributed execution.

## Problem Statement

The current indexing bottleneck is likely inside `enriching_segments`, not `embedding_frames`.

Even after improving `OpenCLIP`, enrich still appears slow because:

- segment processing is largely sequential
- CPU image loading and preprocessing can block model work
- `InternVL`, OCR, and detector work are not clearly timed, so the slowest sub-stage is not visible

Without visibility and better staging, throughput remains low even if the GPU is capable of more.

## Recommended Approach

Use a throughput-first enrich pipeline with bounded concurrency:

- measure each enrich sub-stage separately
- prefetch enrich inputs before heavy model calls
- overlap lighter CPU-bound preparation where safe
- keep only one heavy `InternVL` inference lane active by default
- only parallelize work that does not alter model outputs

This preserves quality while reducing dead time between enrich steps.

## Architecture

### 1. Enrich Timing

Record timings for:

- OCR
- object detection
- `InternVL` description
- branch-B text embedding
- DB persistence/commit

Timing data should be attached to debug output or returned pipeline metadata so the bottleneck is observable per run.

### 2. Enrich Input Prefetch

Before processing enrich steps, prepare the per-segment payloads up front:

- locate keyframe rows
- collect image paths
- optionally decode or validate images ahead of time

This reduces per-segment lookup overhead and keeps the enrich loop focused on model work.

### 3. Bounded Overlap For Lighter Stages

Allow small bounded concurrency for lighter or more I/O-bound enrich sub-steps when possible:

- image loading
- OCR preparation
- detector preparation

The overlap must stay bounded so the workstation remains stable.

### 4. Single Heavy VLM Lane By Default

Do not open multiple uncontrolled `InternVL` lanes on the same GPU in the first iteration.

The default rule is:

- one worker process
- one active `InternVL` execution lane
- bounded CPU overlap around it

This avoids VRAM contention and quality-neutral slowdown from oversubscription.

### 5. Config Surface

Add enrich throughput settings such as:

- `enrich_prefetch_workers`
- `enable_stage_timing`
- optional small-stage overlap flags if needed

Defaults should favor throughput improvement while remaining safe on a single-GPU workstation.

## Risks

- Too much CPU-side overlap can create contention and make the machine feel unstable.
- Timing instrumentation adds a small amount of overhead, but the visibility gain outweighs it.
- If the true bottleneck is almost entirely `InternVL`, prefetch alone may improve throughput only moderately.

## Acceptance Criteria

- Enrich runs emit sub-stage timing data.
- Enrich input preparation is moved out of the hottest inner loop where practical.
- Throughput improves on the same clip/model configuration without intentionally reducing output quality.
- No model downgrade or enrich stage removal is used to achieve the speedup.
