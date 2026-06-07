from __future__ import annotations

import re

CANONICAL_OBJECTS: dict[str, list[str]] = {
    "person": ["human", "man", "woman", "people"],
    "car": ["automobile", "sedan", "taxi"],
    "boat": ["ship", "vessel"],
    "bicycle": ["bike"],
    "motorcycle": ["motorbike"],
    "phone": ["smartphone", "mobile phone"],
    "laptop": ["notebook computer"],
    "screen": ["display", "monitor"],
    "sign": ["poster", "board", "banner"],
    "store": ["shop", "market"],
    "beach": ["shore", "coast"],
    "underwater": ["ocean floor", "sea floor"],
    "fish": ["sea creature"],
}

ADDITIONAL_PROMPTS = [
    "vehicle",
    "bag",
    "bottle",
    "bus",
    "truck",
    "train",
    "airplane",
    "menu",
    "label",
    "text",
    "document",
    "book",
    "keyboard",
    "desk",
    "table",
    "chair",
    "street",
    "road",
    "building",
    "room",
    "kitchen",
    "classroom",
    "restaurant",
    "coral",
    "shark",
    "eel",
    "ray",
]


def _normalize_term(term: str) -> str:
    normalized = re.sub(r"\s+", " ", term.strip().lower())
    return normalized


def canonicalize_object_label(label: str) -> str:
    normalized = _normalize_term(label)
    if not normalized:
        return ""
    for canonical, aliases in CANONICAL_OBJECTS.items():
        if normalized == canonical:
            return canonical
        if normalized in {_normalize_term(alias) for alias in aliases}:
            return canonical
    return normalized


def build_indexing_prompts() -> list[str]:
    prompts = {
        canonical
        for canonical in CANONICAL_OBJECTS
    }
    for aliases in CANONICAL_OBJECTS.values():
        prompts.update(_normalize_term(alias) for alias in aliases if _normalize_term(alias))
    prompts.update(_normalize_term(prompt) for prompt in ADDITIONAL_PROMPTS if _normalize_term(prompt))
    return [*sorted(prompts), ""]


def normalize_query_object_terms(terms: list[str]) -> list[str]:
    normalized: list[str] = []
    for term in terms:
        canonical = canonicalize_object_label(term)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized
