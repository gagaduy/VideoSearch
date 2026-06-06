from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.db.repositories.branch_search import (
    search_dense_branch,
    search_object_branch,
    search_temporal_seed_branch,
    search_text_branch,
)
from app.services.query_understanding import ObjectFilter, TemporalStep
from worker.adapters.openclip_adapter import OpenClipAdapter


def _tokenize_queries(queries: list[str]) -> list[str]:
    seen: list[str] = []
    for query in queries:
        for token in re.findall(r"[a-z0-9]+", query.lower()):
            if token not in seen:
                seen.append(token)
    return seen


def collect_branch_candidates(
    db: Session,
    *,
    semantic_query: str,
    expanded_queries: list[str],
    object_filters: list[ObjectFilter],
    temporal_steps: list[TemporalStep],
    dense_encoder: OpenClipAdapter | None = None,
) -> dict[str, list[dict[str, object]]]:
    dense_encoder = dense_encoder or OpenClipAdapter()
    dense_embedding = dense_encoder.embed_text(semantic_query).values
    text_terms = _tokenize_queries(expanded_queries or [semantic_query])

    return {
        "dense_a": search_dense_branch(db, dense_embedding, "embedding_branch_a"),
        "dense_b": search_dense_branch(db, dense_embedding, "embedding_branch_b"),
        "ocr_text": search_text_branch(db, text_terms),
        "object_entity": search_object_branch(db, object_filters),
        "temporal_seed": search_temporal_seed_branch(db, temporal_steps),
    }
