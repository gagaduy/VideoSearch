# OpenAI Vision Rerank Top 8 Design

## Summary

This design adds an online reranking stage that uses the OpenAI API to visually rerank the top 8 locally retrieved frame candidates.

The existing local retrieval stack remains the recall engine. OpenAI vision reranking is added only after the first-stage candidates have already been selected. This improves precision at the top of the results without changing indexing, schema, or database size.

## Goals

- improve result quality for complex queries without reindexing
- use OpenAI only on a small top-K set
- keep local retrieval as the main recall stage
- preserve graceful fallback when OpenAI is unavailable

## Non-goals

- reranking the full candidate pool
- changing indexing artifacts or schema
- introducing a new asynchronous search queue
- replacing the existing local retrieval stack

## Retrieval Strategy

### Stage 1: Local Retrieval

The current local retrieval path remains unchanged:

- dense embedding retrieval
- caption / text matching
- OCR matching
- object-aware scoring
- fusion and local scoring

This stage produces the ranked candidate list used today.

### Stage 2: OpenAI Vision Rerank

Take the top 8 locally ranked candidates and send them to an OpenAI vision-capable model for a second-pass relevance judgment.

The vision model should look directly at the frame images rather than relying only on caption or OCR metadata.

The reranker should be instructed to score each frame for:

- object correctness
- object count correctness when the query implies quantities
- scene relationship correctness
- action and layout match quality

This stage is intended to fix errors caused by:

- weak captions
- imperfect OCR
- incorrect object detection
- local ranking confusion between visually similar scenes

## Payload Design

For each rerank call, the API will send:

- the user query
- up to 8 frame images
- frame IDs
- optional lightweight metadata such as timestamps or captions if useful for debugging, but the image itself remains primary

The model should return a compact machine-readable structure for each candidate:

- `frame_id`
- `vision_score` in the range `0.0..1.0`
- optional short reasoning text if enabled for debug mode

The default production path should avoid verbose reasoning payloads unless that information is explicitly useful.

## Final Scoring

The final score will blend local retrieval and OpenAI vision rerank:

- `final_score = 0.45 * local_score + 0.55 * vision_score`

Rationale:

- local retrieval remains important for recall and stability
- vision rerank gets the higher weight because the system currently needs stronger top-result precision

The reranked list is then sorted by this blended score and returned to the UI.

## Failure Handling

OpenAI rerank must never be a single point of failure for search.

Behavior:

- if no OpenAI API key is configured, skip rerank and return local results
- if the OpenAI request times out, skip rerank and return local results
- if the OpenAI request fails or rate-limits, skip rerank and return local results
- log rerank failure for debugging, but do not fail the search request

If the local result set has fewer than 8 items, rerank only the available items.

## Cost And Latency Guardrails

To keep cost controlled:

- rerank only top 8 candidates
- do not rerank additional pages or deep candidates
- use a short request timeout
- avoid unnecessary verbose reasoning output

This keeps the OpenAI stage bounded and practical for interactive use.

## API And Service Scope

This change should be implemented as an additional service-layer rerank step after local retrieval and before response return.

The search API contract should remain stable from the UI perspective. The UI does not need to know whether rerank was active unless debug information is deliberately exposed.

The design should allow the rerank stage to be feature-flagged or disabled cleanly.

## Testing

Add tests for:

- rerank only processes at most 8 candidates
- score blending uses `0.45 / 0.55`
- fallback behavior when OpenAI is unavailable
- fallback behavior on timeout or API failure
- search response shape remains valid with rerank enabled or skipped

Regression coverage should confirm that normal local search still works when rerank is disabled.

## Expected Outcome

This design should improve the quality of top-ranked results, especially for:

- multi-object queries
- count-sensitive queries
- relationship-heavy natural language queries
- visually confusing scenes where caption or object metadata is noisy

The system remains lightweight operationally because:

- indexing is unchanged
- database growth is unchanged
- only the top 8 search candidates incur external rerank cost
