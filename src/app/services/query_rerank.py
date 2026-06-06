from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.services.query_understanding import StructuredQuery


def rerank_structured_candidates(
    structured: StructuredQuery,
    candidates: list[dict[str, Any]],
    api_key: str,
    model: str,
) -> dict[int, float]:
    if not api_key or not candidates:
        return {}

    prompt = (
        "Score each video retrieval candidate from 0.0 to 1.0 for how well it satisfies the structured query. "
        "Use the candidate caption, OCR, semantic entities, semantic counts, object labels, object counts, and object positions. "
        "Return JSON only as an array of objects with keys segment_id and score. "
        "Be strict about object count and temporal intent when present. Prefer semantically correct marine or domain-specific entities over noisy detector labels.\n"
        f"Structured query: {json.dumps(structured.model_dump(), ensure_ascii=True)}\n"
        f"Candidates: {json.dumps(candidates, ensure_ascii=True)}"
    )
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=model, input=prompt)
        text = response.output_text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < start:
            return {}
        payload = json.loads(text[start : end + 1])
        scores: dict[int, float] = {}
        for item in payload:
            segment_id = int(item["segment_id"])
            score = float(item["score"])
            scores[segment_id] = max(0.0, min(score, 1.0))
        return scores
    except Exception:
        return {}
