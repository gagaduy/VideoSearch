from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from worker.retrieval_ontology import canonicalize_object_label


class ObjectFilter(BaseModel):
    label: str
    min_count: int = 1
    max_count: int | None = None
    regions: list[str] = Field(default_factory=list)


class TemporalStep(BaseModel):
    text: str
    object_filters: list[ObjectFilter] = Field(default_factory=list)


class StructuredQuery(BaseModel):
    original_query: str
    semantic_query: str
    semantic_queries: list[str] = Field(default_factory=list)
    object_filters: list[ObjectFilter] = Field(default_factory=list)
    must_terms: list[str] = Field(default_factory=list)
    soft_terms: list[str] = Field(default_factory=list)
    temporal_steps: list[TemporalStep] = Field(default_factory=list)


def _extract_json_payload(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no json object found")
    return json.loads(text[start : end + 1])


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}
_REGIONS = {"left", "right", "center", "top", "middle", "bottom"}
_STOPWORDS = {
    "a",
    "an",
    "the",
    "on",
    "in",
    "at",
    "with",
    "then",
    "first",
    "next",
    "later",
    "after",
    "before",
}


def _split_temporal_steps(query: str) -> list[str]:
    steps = [part.strip(" ,.") for part in re.split(r"\b(?:then|next|later|after that)\b", query, flags=re.IGNORECASE) if part.strip(" ,.")]
    if len(steps) <= 1:
        return []
    cleaned: list[str] = []
    for step in steps:
        normalized = re.sub(r"^\b(first|finally)\b\s*", "", step, flags=re.IGNORECASE).strip(" ,.")
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _extract_object_filters(text: str) -> list[ObjectFilter]:
    lowered = text.lower()
    regions = [region for region in _REGIONS if region in lowered]
    tokens = re.findall(r"[a-z0-9]+", lowered)
    has_count_hint = any(token in _NUMBER_WORDS or token.isdigit() for token in tokens)
    if not has_count_hint and not regions:
        return []

    filters: list[ObjectFilter] = []
    for index, token in enumerate(tokens):
        if token in _NUMBER_WORDS or token.isdigit():
            if index + 1 >= len(tokens):
                continue
            label = tokens[index + 1]
            if label in _STOPWORDS or label in _REGIONS or label in _NUMBER_WORDS:
                continue
            label = canonicalize_object_label(label)
            min_count = _NUMBER_WORDS[token] if token in _NUMBER_WORDS else int(token)
            filters.append(ObjectFilter(label=label, min_count=min_count, regions=regions))
            break

    if filters:
        return filters

    for token in tokens:
        if token in _STOPWORDS or token in _REGIONS or token in _NUMBER_WORDS:
            continue
        filters.append(ObjectFilter(label=canonicalize_object_label(token), min_count=1, regions=regions))
        break
    return filters


def _local_fallback(query: str) -> StructuredQuery:
    temporal_steps = [
        TemporalStep(text=step, object_filters=_extract_object_filters(step))
        for step in _split_temporal_steps(query)
    ]
    return StructuredQuery(
        original_query=query,
        semantic_query=query,
        semantic_queries=[query],
        object_filters=_extract_object_filters(query),
        must_terms=[],
        soft_terms=[],
        temporal_steps=temporal_steps,
    )


def parse_structured_query(query: str, api_key: str, model: str) -> StructuredQuery:
    fallback = _local_fallback(query)
    if not api_key:
        return fallback

    prompt = (
        "Convert this video retrieval query into a compact JSON object. "
        "Return JSON only with keys: semantic_query, semantic_queries, object_filters, must_terms, soft_terms, temporal_steps. "
        "Each object_filter must have label, min_count, max_count, regions. "
        "Each temporal step must have text and object_filters. "
        "Use empty arrays when not applicable. Do not invent constraints unless implied by the query.\n"
        f"Query: {query}"
    )
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=model, input=prompt)
        payload = _extract_json_payload(response.output_text.strip())
        semantic_query = str(payload.get("semantic_query") or query).strip() or query
        semantic_queries = [str(item).strip() for item in payload.get("semantic_queries", []) if str(item).strip()]
        normalized = StructuredQuery(
            original_query=query,
            semantic_query=semantic_query,
            semantic_queries=[semantic_query, *semantic_queries],
            object_filters=[ObjectFilter.model_validate(item) for item in payload.get("object_filters", [])],
            must_terms=[str(item).strip() for item in payload.get("must_terms", []) if str(item).strip()],
            soft_terms=[str(item).strip() for item in payload.get("soft_terms", []) if str(item).strip()],
            temporal_steps=[TemporalStep.model_validate(item) for item in payload.get("temporal_steps", [])],
        )
        unique_queries: list[str] = []
        for item in normalized.semantic_queries:
            if item not in unique_queries:
                unique_queries.append(item)
        normalized.semantic_queries = unique_queries or [semantic_query]
        return normalized
    except Exception:
        return fallback
