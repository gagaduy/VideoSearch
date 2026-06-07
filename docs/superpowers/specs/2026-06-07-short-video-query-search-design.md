# Short Video Query Search Design

## Summary

Replace the current `Search By Image` mode with `Search By Video Clip`, where the user uploads a short query clip and receives the most visually similar indexed frames. The system will treat the uploaded clip as a multi-frame visual query, run local retrieval against existing frame embeddings, and then rerank the top `8` candidates with OpenAI vision for better precision.

This design keeps the current indexed database intact, does not require re-indexing existing videos, and reuses the existing result grid and timeline preview UX.

## Goals

- Replace image query search with short video clip query search.
- Accept uploaded query clips from the web UI.
- Return the most similar indexed frames, not full matched clips.
- Reuse the current preview workflow so users can inspect a matching frame in timeline context.
- Use local retrieval first and OpenAI vision reranking second.
- Keep the OpenAI rerank scope capped to `top 8` candidates.

## Non-Goals

- No long-video query workflow.
- No audio-based search.
- No motion-specific video encoder in this iteration.
- No clip-to-clip temporal localization output.
- No permanent database storage for uploaded query clips.
- No simultaneous image and video query modes in the same sidebar section.

## User Experience

### Sidebar Changes

Replace the current `Search By Image` panel with `Search By Video Clip`.

The new panel will contain:

- a file input that accepts video uploads
- a submit button such as `Find Similar Frames`
- a compact preview area for the selected query clip
- a short helper note that this mode is meant for short clips

### Query Result Experience

- Results continue to render in the existing results grid.
- Clicking a result continues to populate the preview pane on the right.
- Search submit is blocked while a clip-query request is in progress.
- If no strong matches remain after thresholding, the UI should show a clear empty-state message instead of weak filler results.

## Query Clip Constraints

This iteration is explicitly for short uploaded clips.

- intended duration: under `10` seconds
- backend rejects clips above the configured duration limit
- UI should display a clear validation error when the uploaded clip exceeds the limit

The exact limit should live in configuration so it can be tuned without redesigning the flow.

## Retrieval Architecture

### High-Level Flow

1. User uploads a short query clip from the UI.
2. Backend stores the upload temporarily.
3. Backend extracts multiple representative frames from the query clip.
4. Each extracted frame is embedded with the same OpenCLIP pipeline used for indexed frames.
5. Local retrieval gathers matching indexed frames/segments for each query frame.
6. Backend aggregates local candidate scores across all query frames.
7. The combined candidate list is filtered and reduced to the strongest local candidates.
8. OpenAI vision reranks the top `8` candidates using the query clip frames and candidate frames.
9. Final results are returned in the same response shape used by the existing search UI.

### Why Multi-Frame Query Instead of Single Keyframe

The query clip is short enough that extracting multiple frames is affordable. Using multiple frames provides better coverage than relying on a single snapshot, especially when:

- the clip contains slight camera movement
- the best visual cue appears only in part of the clip
- a single frame does not fully represent the query scene

## Query Frame Extraction

The backend should sample a small but meaningful set of frames distributed across the query clip.

Design constraints:

- enough frames to represent the whole clip
- bounded count to keep latency and OpenAI cost under control
- deterministic sampling so repeated searches behave consistently

The exact extraction count can remain configurable, but this design assumes a fixed bounded count rather than adaptive scene segmentation in the first iteration.

## Local Retrieval Strategy

### Retrieval Basis

The local retrieval phase should use the existing indexed visual embeddings. No schema expansion is required for this iteration.

For each query frame:

- compute an embedding with OpenCLIP
- retrieve similar indexed frame or segment candidates from the current search corpus

### Candidate Aggregation

Because a query clip will yield multiple frame-level retrieval lists, the backend needs to merge them into one candidate set.

Recommended aggregation behavior:

- union candidates across all query frames
- accumulate evidence when the same indexed frame/segment is retrieved by multiple query frames
- preserve a strong score when a candidate matches one query frame very well
- reward candidates that match consistently across several query frames

This should be implemented as score aggregation, not as a hard requirement that all query frames must match the same candidate.

## OpenAI Vision Rerank

### Scope

After local aggregation, rerank only the strongest `top 8` local candidates.

### Inputs

The reranker receives:

- multiple extracted frames from the uploaded query clip
- the `top 8` candidate indexed frames
- a prompt instructing the model to judge which candidate frame best matches the visual content of the query clip

### Output

The reranker returns a score or ranking for the candidate frames, which is blended with the local retrieval score.

### Blend Policy

Reuse the current rerank philosophy:

- local retrieval preserves recall
- OpenAI vision improves precision

The exact weighting may differ from text-query rerank later, but this iteration should keep the same pattern of local-first retrieval with OpenAI refinement.

### Failure Handling

If OpenAI times out, fails, or is unavailable:

- do not fail the search request
- return local aggregated results instead
- optionally record diagnostics for observability

## API Design

Add a dedicated clip-query search endpoint rather than overloading the text-search route.

Expected request shape:

- multipart form upload
- one short video clip file

Expected response shape:

- same result schema as existing search endpoints where practical
- include ranked frames and media URLs compatible with the current UI

This keeps the frontend implementation simple and allows reuse of existing result rendering code.

## Temporary File Handling

Uploaded query clips and extracted query frames should be treated as temporary artifacts.

Requirements:

- store them in a temporary query area
- clean them up after the search completes or on a bounded retention window
- do not insert them into the main indexed video tables

This prevents the search feature from polluting the persistent index.

## Result Filtering

Keep the current relevance-threshold behavior:

- filter weak matches
- cap visible results
- avoid flooding the grid with low-signal frames

This should happen after final scoring so the UI only shows meaningful matches.

## Configuration

Add configuration for the new mode, including:

- max query clip duration
- max extracted query frames
- local candidate pool size before rerank
- OpenAI rerank enabled flag
- OpenAI rerank top-k, fixed at `8` by default
- timeout for clip-query rerank

The goal is to make tuning practical without redesigning the feature.

## Error Handling

### Invalid or Unsupported Video

- reject unsupported or unreadable uploads
- return a short user-facing error
- avoid partial search state

### Clip Too Long

- reject with a clear validation message
- do not attempt partial processing in this iteration

### No Query Frames Extracted

- return an explicit error instead of silently returning no results

### No Strong Matches

- return an empty result set with a clear UI message
- do not synthesize filler results

## Testing Strategy

### Backend Tests

- unit tests for query clip frame extraction policy
- unit tests for multi-frame local score aggregation
- unit tests for top-8 rerank candidate selection
- unit tests for OpenAI rerank fallback behavior
- API tests for clip upload validation and successful search flow

### Frontend Tests

- web asset tests for replacing `Search By Image` with `Search By Video Clip`
- loading/disabled-state tests for duplicate submit blocking
- empty-state and validation-state coverage where feasible

### Regression Tests

- existing text search must continue to work unchanged
- current timeline preview and result click behavior must remain intact

## Rollout Notes

This feature should replace the old image-query mode in the UI rather than coexist with it in this iteration. The scope stays intentionally tight:

- one uploaded query clip
- local multi-frame retrieval
- OpenAI rerank top `8`
- existing result grid and preview panel reused

That keeps the feature aligned with the current product direction without expanding database shape or indexing cost.
