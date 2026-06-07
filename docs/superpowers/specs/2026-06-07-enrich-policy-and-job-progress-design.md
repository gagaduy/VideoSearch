# Enrich Policy And Job Progress Design

**Date:** 2026-06-07

## Goal

Clean up the indexing enrich policy so `balanced` no longer exists in behavior or naming, slightly improve retrieval quality by relying on the existing `InternVL2_5-1B` path more consistently, and make indexing progress visible in the web UI without checking the database manually.

## Scope

- Remove `balanced` from enrich behavior, labels, and tests.
- Keep two enrich modes only: `local` and `full`.
- Preserve the current worker and schema boundaries.
- Improve the ingest progress panel in the existing web UI using current job polling.
- Investigate and fix the misleading search score presentation and the OCR scoring bug during implementation.

## Non-Goals

- No schema redesign.
- No websocket or server-sent event progress channel.
- No full retrieval architecture rewrite.
- No new VLM model introduction.

## Enrich Policy

### Profiles

The system keeps exactly two meaningful enrich modes:

- `local`
  Run `InternVL2_5-1B` on a sparse schedule using the existing stride logic.

- `full`
  Run `InternVL2_5-1B` on every segment.

`balanced` must be removed completely from enrich control flow. No fallback naming, heuristic labels, or tests may refer to it.

### Segment Decision Rules

`_should_run_vlm_enrichment(...)` remains the single gate for whether a segment should receive VLM enrichment.

- In `full`, always return `True`.
- In `local`, use the existing sparse decision strategy:
  - first segment
  - last segment
  - sparse stride hit
  - segments with weak supporting evidence such as missing OCR or missing detections

This keeps indexing cost bounded while slightly improving quality relative to the current mixed semantics around `balanced`.

### Behavior When VLM Is Not Scheduled

If a segment is not selected for VLM enrichment because the profile is `local`, the worker may still populate lower-cost evidence:

- OCR text and OCR tokens
- object detections
- object counts and coarse positions
- semantic counts derived from object evidence when appropriate

Caption text for such segments may be generated from a neutral local fallback helper if needed, but it must not reference `balanced` in any naming or diagnostics.

## Failure Semantics

If a segment was selected for VLM enrichment and `InternVL2_5-1B` fails:

- do not fall back to caption generation from the caption adapter
- do not fall back to semantic entity extraction from the semantic entity adapter
- keep `caption_text` and semantic enrich fields empty if no good output exists
- still keep OCR and detector outputs if those stages succeeded
- record the failure in `stage_failures_json["branch_b"]`

This makes the stored data honest about quality loss instead of masking a VLM failure with lower-quality substitutes.

## Search Follow-Up Included In This Work

The implementation should also address two search quality issues already identified during investigation:

- `ocr_score` is currently computed from `caption` instead of `ocr_text`
- the UI presents `score` like a percentage even though it is an internal rerank score with a naturally low numeric range

The score scale itself is not inherently broken, but the presentation is misleading and dense ranking is currently understated because RRF produces very small values relative to the lexical and object signals.

## Job Progress UI

### Current Constraint

The first iteration must use the existing `/jobs/{id}` polling flow and `job.stage` field. No new transport or background event channel is required.

### UI Goals

The ingest panel must let the user tell, at a glance:

- which stage is running
- whether progress is advancing
- how many frames or segments have been processed when available
- whether the job completed, failed, or timed out in polling

### UI Content

The job status panel should show:

- human-readable stage label
- `current/total` counter when the stage carries quantitative progress
- numeric percent label derived from stage position
- progress bar as a secondary indicator
- short status note describing what the worker is doing

Example states:

- `Queued`
- `Extracting frames`
- `Embedding frames 37/120`
- `Building segments`
- `Enriching segments 8/24`
- `Completed`
- `Failed`

### Progress Mapping

The progress bar should keep a coarse staged mapping similar to today, but text becomes the primary source of truth.

- `queued`
- `processing`
- `extracting_frames`
- `embedding_frames:x/y`
- `building_segments`
- `enriching_segments:x/y`
- `done`
- `error`

The UI should parse the `x/y` values and render them directly rather than leaving users to infer progress from bar width alone.

### Failure And Timeout Messaging

The panel must distinguish:

- worker failed
- upload failed
- polling timed out while the job may still be running

That distinction should appear in the status note instead of collapsing all non-success states into a generic message.

## Testing Strategy

Implementation must follow TDD.

Expected test coverage:

- unit tests for enrich gating without `balanced`
- unit or integration coverage for VLM failure behavior without low-quality fallback
- web UI tests asserting the progress panel contains the new progress fields
- search tests covering the `ocr_score` fix or at minimum a targeted unit test around score input construction

## Risks

- Sparse `local` scheduling may still miss some fine-grained semantic scenes; this is accepted for bounded cost.
- Leaving score scale unchanged while relabeling presentation improves clarity, but may not fully solve ranking quality if dense weighting remains too weak.
- Reusing `job.stage` strings keeps backend changes small, but the frontend parser must stay aligned with worker stage formats.

## Acceptance Criteria

- No enrich logic path references `balanced`.
- `local` and `full` are the only supported enrich behaviors.
- If scheduled VLM enrichment fails, the segment does not silently downgrade into caption/entity fallback.
- The web ingest panel shows stage, counter when available, percent label, and clearer status messaging.
- A user can tell indexing state from the UI alone without opening the database.
- Search no longer computes OCR contribution from caption text.
