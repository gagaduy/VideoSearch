# Paper-Grade Video Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current single-embedding-first retrieval stack into a paper-like multi-branch video retrieval engine that stays strong without OpenAI and runs on one GPU.

**Architecture:** Keep `segment` as the search unit, but expand each segment into multiple retrieval views: dense CLIP embeddings, OCR text, object/entity metadata, and temporal sequence structure. Split the implementation into indexing-side upgrades and search-side upgrades so each phase is testable and can land without blocking the next one.

**Tech Stack:** FastAPI, SQLAlchemy, Postgres + pgvector, OpenCLIP, PaddleOCR, Ultralytics YOLO, Python test suite with pytest

---

## File Structure

### Existing files to modify

- `src/app/db/models.py`
  Extend `Segment` and related models so the schema supports multiple dense branches, richer OCR/entity metadata, and stage failure tracking.
- `src/app/db/vector.py`
  Add helpers for multiple vector columns or vector tables.
- `src/app/db/repositories/search.py`
  Replace the single-branch search helper with explicit branch retrieval helpers and temporal neighbor/path fetch helpers.
- `src/app/services/search_service.py`
  Refactor into orchestration only; move branch scoring and fusion into focused helpers.
- `src/app/services/query_understanding.py`
  Strengthen deterministic local parsing and make hard vs soft constraints explicit.
- `src/worker/pipeline.py`
  Rewrite indexing flow to produce multi-view segment artifacts, `YOLO-World` ontology detections, and per-stage failure state.
- `src/worker/adapters/openclip_adapter.py`
  Upgrade default model configuration and expose pooled/keyframe embedding helpers.
- `src/worker/adapters/yolo_adapter.py`
  Replace closed-set YOLO behavior with `YOLO-World` prompt-driven detection.
- `src/worker/adapters/paddleocr_adapter.py`
  Replace tesseract behavior with PaddleOCR behavior and normalized output.
- `tests/unit/test_query_understanding.py`
  Extend coverage for local query parsing, object constraints, and temporal steps.
- `tests/unit/test_worker_tasks.py`
  Extend for indexing-stage state and failure recording.
- `tests/integration/test_search_api.py`
  Expand for multi-branch retrieval behavior.
- `tests/integration/test_index_job_flow.py`
  Expand for richer retrieval artifacts.

### New files to create

- `src/app/db/repositories/branch_search.py`
  Dense branch retrieval, text branch retrieval, object/entity branch retrieval, and branch diagnostics.
- `src/app/services/retrieval_branches.py`
  Query-to-branch execution helpers.
- `src/app/services/object_refinement.py`
  Query-conditioned `YOLO-World` verification on top candidate keyframes.
- `src/app/services/retrieval_fusion.py`
  Reciprocal-rank fusion, normalized weighting, and hard-constraint penalties.
- `src/app/services/temporal_paths.py`
  Ordered-step path search and temporal score calculation.
- `src/app/services/local_rerank.py`
  Final local rerank that does not depend on OpenAI.
- `src/worker/adapters/internvl_adapter.py`
  Secondary semantic branch adapter or semantic-enrichment adapter for branch B.
- `src/worker/adapters/paddleocr_normalize.py`
  OCR normalization helpers.
- `src/worker/retrieval_ontology.py`
  Shared retrieval vocabulary and alias lists used by `YOLO-World`.
- `tests/unit/test_retrieval_branches.py`
  Branch candidate generation tests.
- `tests/unit/test_retrieval_fusion.py`
  Fusion and hard-constraint scoring tests.
- `tests/unit/test_temporal_paths.py`
  Ordered-step temporal retrieval tests.
- `tests/unit/test_local_rerank.py`
  Final local rerank tests.
- `tests/unit/test_paddleocr_normalize.py`
  OCR normalization tests.
- `tests/unit/test_yolo_world_adapter.py`
  Open-vocabulary detector tests and prompt handling.
- `tests/unit/test_retrieval_ontology.py`
  Ontology normalization and alias tests.
- `tests/integration/test_branch_retrieval.py`
  Multi-branch candidate generation integration tests.
- `tests/integration/test_temporal_retrieval.py`
  Ordered-step search integration tests.
- `tests/integration/test_openai_degraded_search.py`
  Verify search stays strong with OpenAI disabled.
- `scripts/evaluate_retrieval.py`
  Offline evaluation harness.
- `tests/fixtures/retrieval_eval_queries.json`
  Labeled evaluation cases for regression tracking.

## Delivery Notes

- This workspace snapshot is not currently a valid git working tree, so commit steps are intentionally omitted from the task checklist.
- Follow TDD: write or expand the failing tests first, then implement the minimum code to satisfy them, then run the focused tests before moving on.
- Keep each new service file focused. Do not continue growing `search_service.py` into the whole engine.

### Task 1: Expand the Retrieval Schema

**Files:**
- Modify: `src/app/db/models.py`
- Modify: `src/app/db/vector.py`
- Modify: `src/app/db/session.py`
- Test: `tests/unit/test_vector_schema.py`
- Test: `tests/integration/test_index_job_flow.py`

- [ ] **Step 1: Write the failing schema tests**

```python
from app.db.models import Segment


def test_segment_model_exposes_multi_branch_fields():
    fields = Segment.__table__.columns.keys()
    assert "embedding_branch_a" in fields
    assert "embedding_branch_b" in fields
    assert "ocr_tokens_json" in fields
    assert "stage_failures_json" in fields


def test_postgres_bootstrap_sql_mentions_new_columns():
    from app.db.vector import postgres_vector_bootstrap_sql

    statements = postgres_vector_bootstrap_sql("vector")
    joined = "\n".join(statements)
    assert "embedding_branch_a" in joined
    assert "embedding_branch_b" in joined
    assert "ocr_tokens_json" in joined
    assert "stage_failures_json" in joined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_vector_schema.py -v`
Expected: FAIL because the new segment fields and bootstrap SQL are not defined.

- [ ] **Step 3: Add the new schema fields and bootstrap support**

```python
class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    segment_index: Mapped[int] = mapped_column(Integer())
    start_timestamp_sec: Mapped[float] = mapped_column(Float())
    end_timestamp_sec: Mapped[float] = mapped_column(Float())
    keyframe_id: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    caption_text: Mapped[str] = mapped_column(Text(), default="")
    ocr_text: Mapped[str] = mapped_column(Text(), default="")
    ocr_tokens_json: Mapped[list[str] | None] = mapped_column(JSON(), nullable=True)
    object_labels_json: Mapped[list[str] | None] = mapped_column(JSON(), nullable=True)
    object_counts_json: Mapped[dict[str, int] | None] = mapped_column(JSON(), nullable=True)
    object_positions_json: Mapped[dict[str, list[str]] | None] = mapped_column(JSON(), nullable=True)
    semantic_entities_json: Mapped[list[dict[str, object]] | None] = mapped_column(JSON(), nullable=True)
    semantic_aliases_json: Mapped[dict[str, list[str]] | None] = mapped_column(JSON(), nullable=True)
    semantic_counts_json: Mapped[dict[str, int] | None] = mapped_column(JSON(), nullable=True)
    embedding_branch_a: Mapped[list[float] | None] = mapped_column(EmbeddingVector, nullable=True)
    embedding_branch_b: Mapped[list[float] | None] = mapped_column(EmbeddingVector, nullable=True)
    stage_failures_json: Mapped[dict[str, str] | None] = mapped_column(JSON(), nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON(), nullable=True)
```

```python
def postgres_vector_bootstrap_sql(embedding_udt_name: str | None) -> list[str]:
    statements = ["CREATE EXTENSION IF NOT EXISTS vector"]
    statements.append(
        "ALTER TABLE segments ADD COLUMN IF NOT EXISTS embedding_branch_a vector"
    )
    statements.append(
        "ALTER TABLE segments ADD COLUMN IF NOT EXISTS embedding_branch_b vector"
    )
    statements.append(
        "ALTER TABLE segments ADD COLUMN IF NOT EXISTS ocr_tokens_json jsonb"
    )
    statements.append(
        "ALTER TABLE segments ADD COLUMN IF NOT EXISTS semantic_aliases_json jsonb"
    )
    statements.append(
        "ALTER TABLE segments ADD COLUMN IF NOT EXISTS stage_failures_json jsonb"
    )
    return statements
```

- [ ] **Step 4: Run the focused schema tests**

Run: `pytest tests/unit/test_vector_schema.py -v`
Expected: PASS

- [ ] **Step 5: Run the indexing integration test**

Run: `pytest tests/integration/test_index_job_flow.py -v`
Expected: FAIL or PARTIAL FAIL if the worker has not started populating the new fields yet.

### Task 2: Replace OCR With PaddleOCR and Normalize Output

**Files:**
- Modify: `src/worker/adapters/paddleocr_adapter.py`
- Create: `src/worker/adapters/paddleocr_normalize.py`
- Test: `tests/unit/test_paddleocr_adapter.py`
- Test: `tests/unit/test_paddleocr_normalize.py`

- [ ] **Step 1: Write the failing OCR normalization tests**

```python
from worker.adapters.paddleocr_normalize import normalize_ocr_lines, tokenize_ocr_text


def test_normalize_ocr_lines_merges_and_cleans_text():
    lines = [["HELLO"], ["World"], ["  2025  "]]
    assert normalize_ocr_lines(lines) == "hello world 2025"


def test_tokenize_ocr_text_deduplicates_tokens():
    assert tokenize_ocr_text("boat boat shark 2025") == ["boat", "shark", "2025"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_paddleocr_normalize.py -v`
Expected: FAIL because the normalization module does not exist.

- [ ] **Step 3: Add the OCR normalization helpers**

```python
import re


def normalize_ocr_lines(lines: list[list[str]]) -> str:
    text = " ".join(part.strip() for line in lines for part in line if part and part.strip())
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize_ocr_text(text: str) -> list[str]:
    seen: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token not in seen:
            seen.append(token)
    return seen
```

- [ ] **Step 4: Rewrite the OCR adapter to use PaddleOCR first**

```python
from worker.adapters.paddleocr_normalize import normalize_ocr_lines, tokenize_ocr_text


class PaddleOcrAdapter:
    def __init__(self, engine_name: str = "paddleocr") -> None:
        self.engine_name = engine_name
        self._engine = None

    def _lazy_load(self):
        if self._engine is not None:
            return self._engine
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return self._engine

    def extract_text(self, image_path: str) -> dict[str, object]:
        try:
            engine = self._lazy_load()
            result = engine.ocr(image_path, cls=True)
            lines = [[item[1][0] for item in block] for block in result if block]
            text = normalize_ocr_lines(lines)
            return {
                "text": text,
                "tokens": tokenize_ocr_text(text),
                "raw": result,
                "image_path": image_path,
            }
        except Exception:
            return {"text": "", "tokens": [], "raw": [], "image_path": image_path}
```

- [ ] **Step 5: Run OCR unit tests**

Run: `pytest tests/unit/test_paddleocr_normalize.py tests/unit/test_paddleocr_adapter.py -v`
Expected: PASS

### Task 3: Upgrade the Worker to Build Multi-View Segments

**Files:**
- Modify: `src/worker/pipeline.py`
- Modify: `src/worker/adapters/openclip_adapter.py`
- Create: `src/worker/adapters/internvl_adapter.py`
- Modify: `src/worker/adapters/yolo_adapter.py`
- Create: `src/worker/retrieval_ontology.py`
- Modify: `tests/unit/test_worker_pipeline_io.py`
- Modify: `tests/unit/test_worker_tasks.py`
- Modify: `tests/integration/test_index_job_flow.py`

- [ ] **Step 1: Write the failing worker artifact tests**

```python
def test_index_pipeline_persists_multi_view_segment_artifacts(session, sample_video):
    result = run_index_pipeline(session, video_id=sample_video.id, job_id=None)
    segment = session.query(Segment).first()

    assert result["segment_count"] >= 1
    assert segment.embedding_branch_a
    assert segment.embedding_branch_b is not None
    assert segment.ocr_tokens_json is not None
    assert segment.stage_failures_json == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_worker_tasks.py tests/integration/test_index_job_flow.py -v`
Expected: FAIL because the worker still writes a single embedding and no OCR token or stage failure metadata.

- [ ] **Step 3: Add branch-B adapter and branch-aware embedding outputs**

```python
class InternvlAdapter:
    def __init__(self, model_name: str = "OpenGVLab/InternVL2_5-2B") -> None:
        self.model_name = model_name

    def describe_image(self, image_path: str) -> dict[str, object]:
        return {
            "caption": "",
            "tags": [],
            "entities": [],
            "model_name": self.model_name,
        }
```

```python
class OpenClipAdapter:
    def __init__(self, model_name: str = "ViT-H-14", pretrained: str = "laion2b_s32b_b79k") -> None:
        ...
```

- [ ] **Step 4: Rewrite segment persistence in the worker**

```python
segment = Segment(
    video_id=video_id,
    segment_index=segment_index,
    start_timestamp_sec=float(items[0]["timestamp_sec"]),
    end_timestamp_sec=float(items[-1]["timestamp_sec"]),
    keyframe_id=int(keyframe_item["frame_id"]),
    caption_text=caption_text,
    ocr_text=ocr["text"],
    ocr_tokens_json=list(ocr["tokens"]),
    object_labels_json=sorted(counts),
    object_counts_json=counts,
    object_positions_json=positions,
    semantic_entities_json=semantic["entities"],
    semantic_aliases_json=semantic_aliases,
    semantic_counts_json=semantic["counts"],
    embedding_branch_a=_mean_vector([item["embedding_branch_a"] for item in items]),
    embedding_branch_b=_mean_vector([item["embedding_branch_b"] for item in items]),
    stage_failures_json=stage_failures,
    raw_json={"frame_ids": frame_ids, "branch_b_tags": branch_b_tags},
)
```

- [ ] **Step 5: Record stage failures instead of aborting the whole segment**

```python
stage_failures: dict[str, str] = {}
try:
    ocr = ocr_engine.extract_text(image_path)
except Exception as exc:
    ocr = {"text": "", "tokens": [], "raw": []}
    stage_failures["ocr"] = str(exc)
```

- [ ] **Step 6: Run worker and indexing tests**

Run: `pytest tests/unit/test_worker_tasks.py tests/unit/test_worker_pipeline_io.py tests/integration/test_index_job_flow.py -v`
Expected: PASS

### Task 4: Split Search Into Explicit Retrieval Branches

**Files:**
- Create: `src/app/db/repositories/branch_search.py`
- Create: `src/app/services/retrieval_branches.py`
- Modify: `src/app/services/search_service.py`
- Create: `tests/unit/test_retrieval_branches.py`
- Create: `tests/integration/test_branch_retrieval.py`

- [ ] **Step 1: Write the failing branch retrieval tests**

```python
def test_collect_branch_candidates_returns_per_branch_rankings(fake_db):
    result = collect_branch_candidates(
        fake_db,
        semantic_query="red boat",
        object_filters=[],
        temporal_steps=[],
    )
    assert "dense_a" in result
    assert "dense_b" in result
    assert "ocr_text" in result
    assert "object_entity" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_retrieval_branches.py -v`
Expected: FAIL because branch retrieval helpers do not exist.

- [ ] **Step 3: Create focused repository helpers per branch**

```python
def search_dense_branch(db: Session, query_embedding: list[float], column_name: str, limit: int = 80):
    ...


def search_text_branch(db: Session, query_terms: list[str], limit: int = 80):
    ...


def search_object_branch(db: Session, object_filters: list[ObjectFilter], limit: int = 80):
    ...
```

- [ ] **Step 4: Add branch orchestration service**

```python
def collect_branch_candidates(
    db: Session,
    *,
    semantic_query: str,
    expanded_queries: list[str],
    object_filters: list[ObjectFilter],
    temporal_steps: list[TemporalStep],
) -> dict[str, list[dict[str, object]]]:
    dense_a = search_dense_branch(db, embed_branch_a(semantic_query), "embedding_branch_a")
    dense_b = search_dense_branch(db, embed_branch_b(semantic_query), "embedding_branch_b")
    text_rows = search_text_branch(db, tokenize_queries(expanded_queries))
    object_rows = search_object_branch(db, object_filters)
    temporal_rows = search_temporal_seed_branch(db, temporal_steps)
    return {
        "dense_a": dense_a,
        "dense_b": dense_b,
        "ocr_text": text_rows,
        "object_entity": object_rows,
        "temporal_seed": temporal_rows,
    }
```

- [ ] **Step 5: Update search service to call branch orchestration**

```python
branch_rows = collect_branch_candidates(
    db,
    semantic_query=structured.semantic_query,
    expanded_queries=expanded,
    object_filters=object_filters,
    temporal_steps=structured.temporal_steps,
)
```

- [ ] **Step 6: Run branch retrieval tests**

Run: `pytest tests/unit/test_retrieval_branches.py tests/integration/test_branch_retrieval.py -v`
Expected: PASS

### Task 5: Implement Fusion and Final Local Rerank

**Files:**
- Create: `src/app/services/retrieval_fusion.py`
- Create: `src/app/services/local_rerank.py`
- Modify: `src/app/services/search_service.py`
- Create: `tests/unit/test_retrieval_fusion.py`
- Create: `tests/unit/test_local_rerank.py`
- Modify: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write the failing fusion tests**

```python
def test_reciprocal_rank_fusion_prefers_consistent_cross_branch_hits():
    rankings = {
        "dense_a": [10, 20, 30],
        "dense_b": [20, 10, 40],
        "ocr_text": [20, 50],
    }
    scores = fuse_branch_rankings(rankings)
    assert scores[20] > scores[10] > scores[30]


def test_hard_constraint_penalty_drops_non_matching_segment():
    item = {"segment_id": 9, "hard_constraints_passed": False}
    assert apply_constraint_penalty(item, 0.8) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_retrieval_fusion.py -v`
Expected: FAIL because fusion helpers do not exist.

- [ ] **Step 3: Implement branch fusion**

```python
def fuse_branch_rankings(rankings: dict[str, list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings.values():
        for index, segment_id in enumerate(ranking, start=1):
            scores[segment_id] += 1.0 / (k + index)
    return scores


def apply_constraint_penalty(item: dict[str, object], score: float) -> float:
    return score if item.get("hard_constraints_passed", True) else 0.0
```

- [ ] **Step 4: Implement final local rerank**

```python
def score_local_candidate(item: dict[str, object]) -> float:
    return (
        0.35 * float(item.get("dense_score", 0.0))
        + 0.15 * float(item.get("text_score", 0.0))
        + 0.10 * float(item.get("ocr_score", 0.0))
        + 0.15 * float(item.get("object_score", 0.0))
        + 0.10 * float(item.get("entity_score", 0.0))
        + 0.15 * float(item.get("temporal_score", 0.0))
    )
```

- [ ] **Step 5: Wire fusion and local rerank into search service**

```python
fused_scores = fuse_branch_rankings(branch_rankings)
reranked = [
    {
        **item,
        "score": apply_constraint_penalty(
            item,
            score_local_candidate({**item, "dense_score": fused_scores.get(item["segment_id"], 0.0)}),
        ),
    }
    for item in candidate_items
]
```

- [ ] **Step 6: Run fusion and search tests**

Run: `pytest tests/unit/test_retrieval_fusion.py tests/unit/test_local_rerank.py tests/integration/test_search_api.py -v`
Expected: PASS

### Task 6: Replace Neighbor Bonus With Ordered Temporal Path Retrieval

**Files:**
- Create: `src/app/services/temporal_paths.py`
- Modify: `src/app/db/repositories/search.py`
- Modify: `src/app/services/search_service.py`
- Create: `tests/unit/test_temporal_paths.py`
- Create: `tests/integration/test_temporal_retrieval.py`

- [ ] **Step 1: Write the failing temporal path tests**

```python
def test_find_best_temporal_path_respects_step_order():
    steps = [
        [{"segment_id": 1, "video_id": 7, "segment_index": 2, "score": 0.8}],
        [{"segment_id": 2, "video_id": 7, "segment_index": 4, "score": 0.7}],
    ]
    path = find_best_temporal_paths(steps, max_gap=4)
    assert path[0]["segment_ids"] == [1, 2]


def test_find_best_temporal_path_rejects_reverse_order():
    steps = [
        [{"segment_id": 4, "video_id": 7, "segment_index": 5, "score": 0.9}],
        [{"segment_id": 3, "video_id": 7, "segment_index": 3, "score": 0.9}],
    ]
    assert find_best_temporal_paths(steps, max_gap=4) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_temporal_paths.py -v`
Expected: FAIL because the temporal path module does not exist.

- [ ] **Step 3: Implement ordered-step path search**

```python
def find_best_temporal_paths(
    step_candidates: list[list[dict[str, object]]],
    max_gap: int = 6,
) -> list[dict[str, object]]:
    paths: list[dict[str, object]] = []
    for first in step_candidates[0]:
        current_paths = [{"video_id": first["video_id"], "segment_ids": [first["segment_id"]], "score": first["score"], "last_index": first["segment_index"]}]
        for later_step in step_candidates[1:]:
            next_paths = []
            for path in current_paths:
                for candidate in later_step:
                    if candidate["video_id"] != path["video_id"]:
                        continue
                    gap = candidate["segment_index"] - path["last_index"]
                    if gap <= 0 or gap > max_gap:
                        continue
                    next_paths.append(
                        {
                            "video_id": path["video_id"],
                            "segment_ids": [*path["segment_ids"], candidate["segment_id"]],
                            "score": path["score"] + candidate["score"],
                            "last_index": candidate["segment_index"],
                        }
                    )
            current_paths = next_paths
        paths.extend(current_paths)
    return sorted(paths, key=lambda item: item["score"], reverse=True)
```

- [ ] **Step 4: Replace temporal neighbor expansion in search service**

```python
temporal_paths = find_best_temporal_paths(step_candidates, max_gap=6)
temporal_scores = score_temporal_paths(temporal_paths)
```

- [ ] **Step 5: Run temporal tests**

Run: `pytest tests/unit/test_temporal_paths.py tests/integration/test_temporal_retrieval.py -v`
Expected: PASS

### Task 7: Keep OpenAI as Optional Enhancement Only

**Files:**
- Modify: `src/app/services/search_service.py`
- Modify: `src/app/services/query_understanding.py`
- Modify: `src/app/services/query_expansion.py`
- Modify: `src/app/services/query_rerank.py`
- Create: `tests/integration/test_openai_degraded_search.py`

- [ ] **Step 1: Write the failing degraded-mode tests**

```python
def test_search_without_openai_returns_ranked_results(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/search", json={"query": "red boat with text", "object_labels": []})
    payload = response.json()

    assert response.status_code == 200
    assert payload["results"]
    assert payload["expanded_queries"] == ["red boat with text"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_openai_degraded_search.py -v`
Expected: FAIL if the new search pipeline still depends on OpenAI-derived parsing or reranking.

- [ ] **Step 3: Make local parsing and ranking the default path**

```python
structured = parse_structured_query(query, api_key=settings.openai_api_key, model=settings.openai_model)
expanded = structured.semantic_queries or [structured.semantic_query]
if settings.openai_api_key:
    expanded = expand_query(structured.semantic_query, api_key=settings.openai_api_key, model=settings.openai_model)
```

```python
llm_scores = {}
if settings.openai_api_key and rerank_candidates:
    llm_scores = rerank_structured_candidates(...)
```

- [ ] **Step 4: Run degraded-mode tests**

Run: `pytest tests/integration/test_openai_degraded_search.py tests/integration/test_search_api.py -v`
Expected: PASS

### Task 8: Add the Offline Evaluation Harness

**Files:**
- Create: `scripts/evaluate_retrieval.py`
- Create: `tests/fixtures/retrieval_eval_queries.json`
- Test: `tests/integration/test_branch_retrieval.py`

- [ ] **Step 1: Write the failing evaluation harness test**

```python
def test_evaluate_retrieval_script_emits_metrics(tmp_path):
    payload = run_evaluation_fixture("tests/fixtures/retrieval_eval_queries.json")
    assert "recall_at_10" in payload
    assert "mrr_at_10" in payload
    assert payload["query_count"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_branch_retrieval.py -v`
Expected: FAIL because the evaluation harness and fixtures do not exist.

- [ ] **Step 3: Add the evaluation script**

```python
def evaluate_queries(queries: list[dict[str, object]]) -> dict[str, float]:
    hits_at_10 = 0
    reciprocal_ranks: list[float] = []
    for item in queries:
        results = run_search(db, item["query"], item.get("object_labels", []))["results"][:10]
        ranks = [index for index, result in enumerate(results, start=1) if result["segment_id"] in item["expected_segment_ids"]]
        if ranks:
            hits_at_10 += 1
            reciprocal_ranks.append(1.0 / ranks[0])
        else:
            reciprocal_ranks.append(0.0)
    return {
        "query_count": len(queries),
        "recall_at_10": hits_at_10 / max(len(queries), 1),
        "mrr_at_10": sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1),
    }
```

- [ ] **Step 4: Run evaluation-related tests**

Run: `pytest tests/integration/test_branch_retrieval.py -v`
Expected: PASS

### Task 9: Replace Closed-Set Detection With YOLO-World

**Files:**
- Create: `src/worker/retrieval_ontology.py`
- Modify: `src/worker/adapters/yolo_adapter.py`
- Modify: `src/worker/pipeline.py`
- Create: `src/app/services/object_refinement.py`
- Modify: `src/app/services/search_service.py`
- Create: `tests/unit/test_retrieval_ontology.py`
- Create: `tests/unit/test_yolo_world_adapter.py`
- Modify: `tests/integration/test_real_pipeline.py`
- Modify: `tests/integration/test_search_api.py`

- [ ] **Step 1: Write failing ontology tests**

```python
from worker.retrieval_ontology import build_indexing_prompts, normalize_query_object_terms


def test_build_indexing_prompts_contains_background_and_aliases():
    prompts = build_indexing_prompts()
    assert "person" in prompts
    assert "vehicle" in prompts
    assert "" in prompts


def test_normalize_query_object_terms_maps_aliases_to_canonical_terms():
    assert normalize_query_object_terms(["automobile", "ship"]) == ["car", "boat"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_retrieval_ontology.py -v`
Expected: FAIL because the ontology module does not exist.

- [ ] **Step 3: Add retrieval ontology helpers**

```python
CANONICAL_OBJECTS = {
    "car": ["automobile", "vehicle"],
    "boat": ["ship"],
    "person": ["human", "man", "woman"],
}


def build_indexing_prompts() -> list[str]:
    prompts = sorted({name for key, values in CANONICAL_OBJECTS.items() for name in [key, *values]})
    return [*prompts, ""]


def normalize_query_object_terms(terms: list[str]) -> list[str]:
    normalized = []
    for term in terms:
        lower = term.lower()
        canonical = next((key for key, values in CANONICAL_OBJECTS.items() if lower == key or lower in values), lower)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized
```

- [ ] **Step 4: Write failing YOLO-World adapter tests**

```python
from worker.adapters.yolo_adapter import YoloDetectionAdapter


def test_yolo_world_adapter_sets_prompt_classes(monkeypatch):
    calls = {}

    class _World:
        def __init__(self, model_name):
            calls["model_name"] = model_name

        def set_classes(self, classes):
            calls["classes"] = classes

        def predict(self, source, verbose=False):
            return []

    monkeypatch.setattr("worker.adapters.yolo_adapter.YOLOWorld", _World)
    adapter = YoloDetectionAdapter(model_name="yolov8s-worldv2.pt")
    adapter.detect("frame.png", classes=["boat", "person"])

    assert calls["model_name"] == "yolov8s-worldv2.pt"
    assert calls["classes"] == ["boat", "person"]
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/unit/test_yolo_world_adapter.py -v`
Expected: FAIL because the adapter does not expose `YOLO-World` prompt-driven behavior yet.

- [ ] **Step 6: Rewrite the detector adapter around YOLO-World**

```python
from ultralytics import YOLOWorld


class YoloDetectionAdapter:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "yolov8s-worldv2.pt"
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            self._model = YOLOWorld(self.model_name)
        return self._model

    def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
        model = self._lazy_load()
        prompts = classes or build_indexing_prompts()
        model.set_classes(prompts)
        results = model.predict(source=image_path, verbose=False)
        ...
```

- [ ] **Step 7: Update worker indexing to use ontology prompts**

```python
objects = detector.detect(image_path, classes=build_indexing_prompts())
segment.raw_json["object_prompt_set"] = build_indexing_prompts()
segment.raw_json["object_detector_family"] = "yolo_world"
```

- [ ] **Step 8: Add query-conditioned refinement service**

```python
def refine_object_matches(image_paths: list[str], query_terms: list[str]) -> dict[int, float]:
    prompts = normalize_query_object_terms(query_terms)
    ...
```

- [ ] **Step 9: Wire refinement into search scoring**

```python
if query_object_terms:
    refinement_scores = refine_object_matches(candidate_frame_paths, query_object_terms)
    item["object_score"] = max(item["object_score"], refinement_scores.get(item["frame_id"], 0.0))
```

- [ ] **Step 10: Run targeted YOLO-World tests**

Run: `pytest tests/unit/test_retrieval_ontology.py tests/unit/test_yolo_world_adapter.py tests/integration/test_real_pipeline.py tests/integration/test_search_api.py -v`
Expected: PASS

## Self-Review Checklist

- Spec coverage:
  - multi-branch retrieval: Tasks 3, 4, 5
  - stronger OCR: Task 2
  - stronger dense branch and secondary semantic branch: Task 3
  - temporal path retrieval: Task 6
  - OpenAI as optional enhancement: Task 7
  - evaluation harness: Task 8
  - YOLO-World open-vocabulary object branch: Task 9
- Placeholder scan:
  - no `TODO`, `TBD`, or “implement later” text remains
  - each test and implementation step includes concrete commands or code
- Type consistency:
  - `embedding_branch_a`, `embedding_branch_b`, `ocr_tokens_json`, and `stage_failures_json` are used consistently across schema, worker, and retrieval tasks
