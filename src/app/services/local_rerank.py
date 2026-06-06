def score_local_candidate(item: dict[str, object]) -> float:
    return (
        0.35 * float(item.get("dense_score", 0.0))
        + 0.15 * float(item.get("text_score", 0.0))
        + 0.10 * float(item.get("ocr_score", 0.0))
        + 0.15 * float(item.get("object_score", 0.0))
        + 0.10 * float(item.get("entity_score", 0.0))
        + 0.15 * float(item.get("temporal_score", 0.0))
    )
