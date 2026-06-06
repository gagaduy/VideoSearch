# Paper-Grade Video Retrieval Design

**Date:** 2026-06-06

## Goal

Build a single video retrieval system that is as close as practical to the VBS 2025 / MMM paper direction, with emphasis on retrieval quality over indexing cost, while still running on one GPU server. The system must support both benchmark-style known-item search and natural-language video search without relying on OpenAI as the core retrieval engine.

## Constraints

- One GPU deployment target.
- Retrieval quality takes priority over indexing speed.
- A single architecture must support both benchmark-style and natural-language queries.
- OpenAI API is optional enhancement only:
  - allowed for query understanding
  - allowed for query expansion
  - allowed for top-K reranking
  - not allowed as the primary semantic retrieval backbone
- The design should stay close to the paper not only in architecture, but also in model roles and overall retrieval behavior.

## Non-Goals

- Multi-tenant SaaS concerns.
- Distributed indexing across multiple GPUs.
- Full reproduction of every competition-only trick from the winning team.
- A lightweight MVP optimized for minimum engineering effort.

## Recommended Approach

Use a multi-branch retrieval engine with a shared segment-centric index. Each video segment must be represented by multiple retrieval views instead of a single embedding:

- visual-semantic dense retrieval
- second dense semantic branch from a stronger VLM family
- OCR-aware text retrieval
- caption and semantic-entity retrieval
- object-count and object-position retrieval
- temporal step retrieval across ordered segments

The system should generate candidates independently from these branches, then fuse and rerank them using a retrieval core that works without OpenAI. OpenAI sits on top of this core as a quality amplifier for ambiguous or compositional queries.

## Target Architecture

### 1. Segment-Centric Retrieval Core

The primary search unit remains `segment`, not raw frame. A segment contains:

- ordered frame span
- representative keyframe
- pooled segment embedding for each dense branch
- keyframe embedding for each dense branch
- OCR text
- caption text
- semantic entities and aliases
- object labels, counts, and positions
- temporal neighbors and local sequence metadata

This keeps retrieval video-aware and avoids overfitting the engine to isolated keyframes.

### 2. Multi-Branch Candidate Generation

The retrieval engine must query several branches in parallel:

- `Dense branch A`
  CLIP-family retrieval for image-text alignment.

- `Dense branch B`
  stronger VLM-derived semantic retrieval to recover scenes that CLIP misses.

- `Text branch`
  caption + OCR lexical and semantic retrieval.

- `Object/entity branch`
  structured retrieval by object name, count, region, and semantic aliases.

- `Temporal branch`
  sequence-aware retrieval over ordered segment candidates for multi-step queries.

No single branch is allowed to dominate the architecture the way the current `OpenCLIP-first` pipeline does.

### 3. Two-Stage Ranking

Ranking must happen in two stages:

- `Stage 1: branch fusion`
  gather top candidates per branch, normalize scores, then combine with reciprocal-rank fusion and query-type-aware weighting.

- `Stage 2: final rerank`
  use structured features from the fused candidate set to compute a final score. This stage must work locally without OpenAI.

Optional:

- `Stage 3: LLM rerank`
  apply OpenAI only to a small top-K set for difficult semantic or temporal cases.

### 4. Query Modes Without Separate Systems

There is one retrieval engine, but it adapts behavior based on query structure:

- `Benchmark-oriented behavior`
  emphasizes object precision, temporal order, OCR clues, and exact scene constraints.

- `Natural-query behavior`
  emphasizes semantic recall, caption/entity matching, and looser compositional intent.

This is mode switching inside one engine, not two separate stacks.

## Model Stack

The system should follow paper-like model roles as closely as practical.

### Dense Visual-Semantic Branch A

- Primary choice: `OpenCLIP ViT-H/14`
- Purpose:
  strong text-image retrieval backbone
- Indexed outputs:
  - keyframe embedding
  - pooled segment embedding

This replaces the current `ViT-B/32` baseline as the main CLIP branch.

### Dense Semantic Branch B

- Primary choice: `InternVL` family used as semantic enrichment and secondary retrieval branch
- Fallback if direct retrieval embeddings are awkward:
  use InternVL to generate normalized scene descriptions and semantic tags, then index those outputs separately
- Purpose:
  recover semantic cases where CLIP-style retrieval is weak

This branch is required to move the system closer to the paper’s multi-model fusion behavior.

### Object Detection

- Primary choice: stronger YOLO variant such as `yolo11l` or `yolo11x`
- Purpose:
  support object presence, count, and coarse position constraints
- Indexed outputs:
  - label
  - confidence
  - bbox
  - count per segment
  - region map per segment

Current small-detector quality is not sufficient for the target.

### OCR

- Primary choice: `PaddleOCR`
- Purpose:
  reliable text extraction from frames
- Indexed outputs:
  - raw OCR text
  - normalized tokens
  - optional line-level metadata if useful

`tesseract` is not acceptable as the main OCR engine for this target design.

### Caption / Semantic Enrichment

- Primary choice:
  a local VLM-capable path for indexing-time captioning and semantic normalization
- Acceptable role for OpenAI:
  quality-enhancement layer only, not the only source of captions or semantic structure

If the chosen local VLM is weak for caption quality, OpenAI may be used as a secondary enrichment pass for top-value data only, but the core indexing design must not depend on it.

### Query Understanding / Expansion / Rerank

- Primary choice for enhancement: `OpenAI API`
- Allowed tasks:
  - structured query parsing
  - paraphrase generation
  - top-K reranking
- Forbidden role:
  - acting as the main semantic retrieval engine

## Offline Indexing Design

### 1. Video Segmentation

The current embedding-distance grouping is too weak as the long-term segment builder. The new pipeline should:

- extract frames at a configurable base rate
- score visual change over time
- build segments from visual continuity and duration constraints
- choose a representative keyframe per segment

The segment builder does not need to be academically perfect, but it must be more stable than simple adjacent embedding thresholding.

### 2. Multi-View Segment Representation

For each segment, persist:

- segment metadata
- keyframe path
- keyframe timestamp
- start/end timestamp
- dense embedding from branch A
- dense embedding from branch B or branch-B semantic representation
- caption text
- OCR text
- object detections
- object counts
- object regions
- semantic entities
- semantic aliases
- temporal neighbor references

The system must preserve enough raw artifacts to allow re-fusion and re-ranking without re-running the entire video pipeline.

### 3. Storage Layout

Persist retrieval artifacts in a way that supports both vector and structured lookup:

- `Postgres + pgvector` remains the default store
- separate vector columns or vector tables per dense branch
- structured searchable fields on segments
- raw per-frame enrichment tables for debugging and future upgrades

The schema must stop assuming a single dominant embedding.

## Online Retrieval Design

### 1. Structured Query Parsing

Every query is parsed into:

- semantic core text
- object constraints
- count constraints
- region constraints
- OCR-sensitive terms
- temporal steps
- hard constraints vs soft hints

This parsing must have a deterministic local fallback even when OpenAI is unavailable.

### 2. Branch Candidate Retrieval

The system retrieves candidates independently from:

- CLIP branch
- secondary semantic branch
- caption retrieval
- OCR retrieval
- object/entity retrieval
- temporal step retrieval

Each branch returns top-K candidates with branch-specific diagnostics.

### 3. Fusion

Fusion combines branch results using:

- reciprocal rank fusion
- calibrated branch weights
- query-type-aware weight adjustments
- penalties for hard-constraint violations

Weighting must be explicit and inspectable, not hidden in ad hoc logic spread across the codebase.

### 4. Temporal Retrieval

Temporal retrieval must be upgraded from neighbor score propagation to true ordered-step reasoning:

- detect multi-step queries
- score each step against candidate segments
- search valid ordered paths within each video
- reward local continuity and ordering consistency
- allow bounded gaps between steps

This is a required paper-alignment feature, not an optional refinement.

### 5. Final Local Rerank

Before any OpenAI call, the engine computes a final local score from:

- dense retrieval signals
- text similarity signals
- OCR overlap
- object match quality
- entity/alias match quality
- temporal-path quality

This local rerank is the real backbone of accuracy.

### 6. Optional OpenAI Rerank

OpenAI can rerank a small top-K candidate set using structured evidence:

- caption
- OCR
- object counts
- positions
- semantic entities
- temporal summaries

If OpenAI is unavailable, retrieval quality should degrade gracefully rather than collapse.

## Query-Type Adaptation

The retrieval engine should adapt weights by query pattern:

- `Object-heavy`
  increase object/entity branch weight and hard-constraint enforcement.

- `OCR-heavy`
  increase OCR and caption-text weight.

- `Temporal multi-step`
  increase temporal path scoring and reduce dependence on isolated single-segment scores.

- `Broad semantic scene query`
  increase dense semantic branch influence and keep object constraints soft unless explicitly stated.

This adaptation makes one system work for both VBS-style search and natural-language search.

## Required Codebase Changes

### Retrieval Core

Rewrite the current retrieval path so it no longer assumes:

- one dominant dense embedding branch
- object filters as mostly post-hoc scoring
- temporal reasoning as neighbor bonus propagation

The new retrieval core should have explicit branch interfaces and a branch-fusion layer.

### Index Schema

Extend the schema to support:

- multiple dense branches per segment
- richer segment metadata
- stronger entity and OCR indexing
- temporal path metadata

### Worker Pipeline

Replace or upgrade:

- current OCR adapter
- current detector tier
- current segment builder
- current semantic enrichment contract

### Search Service

Refactor search into clearly separated stages:

1. query parsing
2. branch candidate retrieval
3. branch fusion
4. temporal path scoring
5. final local rerank
6. optional LLM rerank

## Error Handling

- If one retrieval branch fails, the engine should still return results from remaining branches.
- If OpenAI fails, the local ranking pipeline still completes.
- If one enrichment artifact is missing for a segment, retrieval should use available evidence instead of dropping the segment outright.
- Indexing jobs must record which enrichment stages failed so artifacts can be backfilled later.

## Testing Strategy

Testing must validate retrieval behavior, not just API responses.

### Unit Tests

- branch scoring
- fusion math
- query parsing fallback
- object/count/region matching
- temporal path construction

### Integration Tests

- indexing a sample video with all retrieval artifacts
- branch-specific search results
- fused retrieval behavior on compositional queries
- graceful degradation when OpenAI is disabled

### Evaluation Harness

Add an offline retrieval evaluation harness with labeled queries so model and fusion changes can be compared over time. This is required because a paper-grade retrieval stack cannot be tuned safely by intuition alone.

## Implementation Priority

Highest priority:

1. replace OCR with PaddleOCR
2. upgrade main dense branch to OpenCLIP H/14
3. redesign schema for multi-branch segment retrieval
4. rewrite search into explicit multi-branch fusion
5. replace temporal bonus with temporal path retrieval

Second priority:

1. add stronger object detector tier
2. add secondary semantic branch from InternVL-style enrichment
3. add local final reranker
4. add optional OpenAI top-K rerank

Third priority:

1. add offline evaluation harness
2. tune branch weights by query class
3. add selective high-value enrichment passes

## Expected Outcome

If implemented as specified, the system should move from:

- a single-embedding-first retrieval system with auxiliary signals

to:

- a paper-like multi-branch retrieval engine
- a retrieval core that remains strong without OpenAI
- a design that supports both benchmark search and natural-language search
- a system that is realistically in the 85-90% paper-alignment range for retrieval architecture and model-role fidelity on a one-GPU deployment
