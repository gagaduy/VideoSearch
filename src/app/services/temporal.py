def expand_temporal_neighbors(
    ranked_segments: list[dict[str, object]],
    neighbors_by_segment: dict[int, list[dict[str, object]]],
    decay: float = 0.85,
) -> list[dict[str, object]]:
    expanded: dict[int, dict[str, object]] = {int(item["segment_id"]): dict(item) for item in ranked_segments}
    primary_segment_ids = [int(item["segment_id"]) for item in ranked_segments]

    for item in ranked_segments[:10]:
        base_score = float(item["score"])
        for neighbor in neighbors_by_segment.get(int(item["segment_id"]), []):
            segment_id = int(neighbor["segment_id"])
            gap = abs(int(neighbor["segment_index"]) - int(item["segment_index"]))
            propagated_score = base_score * (decay / max(gap, 1))
            current = expanded.get(segment_id)
            if current is None or float(current["score"]) < propagated_score:
                expanded[segment_id] = {
                    **(current or {}),
                    **neighbor,
                    "score": propagated_score,
                    "source": "temporal_neighbor",
                }

    primary_results = [expanded[segment_id] for segment_id in primary_segment_ids if segment_id in expanded]
    neighbor_results = [
        row
        for segment_id, row in expanded.items()
        if segment_id not in primary_segment_ids
    ]
    neighbor_results.sort(
        key=lambda row: (float(row["score"]), -float(row["start_timestamp_sec"])),
        reverse=True,
    )
    return [*primary_results, *neighbor_results]
