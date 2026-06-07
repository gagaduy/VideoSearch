# Question To Evidence Search Design

## Summary

Add a dedicated `Search By Question` mode that accepts a full natural-language question and returns ranked frames that are most likely to contain the evidence needed to answer it. This mode is distinct from scene-oriented text search and focuses on evidence-bearing frames, especially OCR-heavy or information-dense content.

The system will not generate the final answer in this iteration. It will only retrieve and rank the frames that best support answering the question.

## Goals

- Add a dedicated `Q&A Search` workflow separate from normal text search.
- Accept a full natural-language question without requiring the user to manually extract keywords.
- Return ranked frames that most likely contain answer-bearing evidence.
- Prioritize OCR/textual evidence, semantic captions, and information-rich frames.
- Use OpenAI reranking on top candidates to improve `answer-bearing relevance`.

## Non-Goals

- No final answer generation in this iteration.
- No chain-of-thought solving pipeline.
- No conversational follow-up reasoning.
- No page-level document OCR subsystem beyond what is already indexed.
- No new indexing schema specifically for Q&A.

## Why This Needs Its Own Search Mode

The current text search mode is designed primarily for:

- visual scene similarity
- object/entity matching
- caption-level semantic retrieval

Question-driven retrieval is a different problem. A good evidence frame may:

- contain text that answers the question
- contain part of the question context but not describe a visible scene well
- be visually plain but semantically critical

Because of that, Q&A search should not be treated as just another free-text search box. It needs its own ranking objective: `which frame is most useful for answering this question?`

## User Experience

### Sidebar Panel

Add a new panel named `Search By Question`.

The panel will contain:

- a multiline or comfortably sized input for the full question
- a submit button
- a loading/disabled state while the request is running
- a short helper note explaining that the system finds likely evidence frames, not the final answer

This should be a separate panel from:

- normal text search
- video clip search

### Result Experience

Results should reuse the existing grid and preview workflow:

- ranked frame cards in the results pane
- click a card to open timeline preview on the right
- no answer text shown yet

If no useful evidence is found, show a clear empty state rather than weak filler results.

## Retrieval Objective

This mode should rank frames by `answer-bearing evidence`, not by generic scene similarity.

For example, if the question is about:

- text shown on screen
- a number hidden in a sentence
- a sign, scoreboard, or overlay
- a statement that must be read rather than inferred visually

then the best frame may be one with:

- strong OCR content
- dense readable text
- captions or semantic text that strongly overlap the question context

## High-Level Flow

1. User enters a full question in `Search By Question`.
2. Backend interprets the question as an evidence retrieval request.
3. Backend extracts evidence-oriented terms and intent from the question.
4. Local retrieval ranks frames/segments with strong OCR/caption/semantic support.
5. Backend selects the strongest local candidates.
6. OpenAI reranks the top candidates by `how useful is this frame for answering the question?`
7. Final ranked frames are returned to the UI.

## Question Understanding

The user will provide the entire question, not a manual keyword list.

The backend should derive:

- likely evidence terms
- textual anchors
- any soft semantic variants helpful for recall
- a notion of whether the question is likely OCR-heavy or scene-heavy

This stage should not over-commit to hard structured constraints unless confidence is high. The example question from the screenshot is a good reminder that retrieval should preserve candidate evidence instead of pruning too early.

## Local Retrieval Strategy

### Core Principle

The local phase should prioritize evidence-bearing signals:

1. OCR text
2. caption text / semantic description
3. semantic entities and aliases
4. object cues only when they help, not as the primary signal

### Ranking Bias

Compared with the normal text search pipeline, Q&A search should bias toward:

- OCR-heavy frames
- frames with rich caption/description overlap
- frames whose metadata suggests answer-bearing content

This mode should not be tuned primarily for:

- generic visual similarity
- object-only matching
- scene composition alone

### Candidate Pool

Local retrieval should gather a sufficiently wide candidate set before reranking. The goal is to keep recall high enough that the answer-bearing frame survives to the rerank stage.

## OpenAI Rerank

### Purpose

Use OpenAI to score which top candidate frames are most likely to help answer the question.

This is different from ordinary visual reranking:

- the prompt should emphasize `answer usefulness`
- not merely visual similarity to a query

### Inputs

The reranker should receive:

- the full original question
- candidate frame image(s)
- optionally candidate metadata such as OCR/caption summaries if that helps the prompt

### Output

Return a ranked or scored list of candidate frames. Blend it with the local retrieval score so that:

- local retrieval preserves recall
- OpenAI rerank improves top precision

### Fallback

If OpenAI fails or times out:

- return local ranked results
- do not fail the entire request

## API Design

Add a dedicated endpoint for Q&A search rather than overloading the standard text-search route.

The request should contain:

- the full question string

The response should reuse the current result schema as much as possible:

- ranked frame results
- media URLs
- score
- diagnostics if already part of the system

This keeps the frontend simple and avoids a second rendering model.

## UI Copy

The panel should frame expectations clearly. It should communicate:

- this mode searches for evidence frames
- it may help answer the question
- it does not claim to produce the final answer yet

This is important for user trust and avoids implying a stronger QA system than what exists.

## Error Handling

### Empty Question

- UI blocks empty submit
- backend validates again

### No Strong Matches

- return empty result set with a clear evidence-search empty state
- do not show unrelated filler frames

### OpenAI Failure

- return local evidence ranking
- do not break the user flow

### Ambiguous or Long Questions

- still allow search
- do not reject merely because the question is long
- treat it as a best-effort evidence search

## Configuration

Add configuration for Q&A search tuning, such as:

- local candidate pool size
- top-k rerank size
- rerank timeout
- mode enabled flag if needed

This iteration should keep the knobs modest and avoid introducing a large configuration surface.

## Testing Strategy

### Backend Tests

- unit tests for question-to-evidence term extraction
- unit tests showing OCR/caption-biased local ranking behavior
- unit tests for rerank candidate selection and fallback behavior
- integration tests for the dedicated Q&A search endpoint

### Frontend Tests

- web asset test for the new `Search By Question` panel
- submit blocking/loading state coverage where feasible
- empty-state rendering expectations

### Regression Tests

- standard text search still works
- video clip search still works
- preview pane behavior remains unchanged

## Rollout Notes

This mode should ship as a new sidebar panel instead of replacing existing search modes. The system then has:

- normal text search for scene/object retrieval
- video clip search for visual query clips
- question search for evidence retrieval

This separation keeps each retrieval objective understandable and tunable.
