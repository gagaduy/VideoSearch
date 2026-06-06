from __future__ import annotations


def find_best_temporal_paths(
    step_candidates: list[list[dict[str, object]]],
    max_gap: int = 6,
) -> list[dict[str, object]]:
    if not step_candidates:
        return []

    paths: list[dict[str, object]] = []
    for first in step_candidates[0]:
        current_paths = [
            {
                "video_id": first["video_id"],
                "segment_ids": [first["segment_id"]],
                "score": float(first["score"]),
                "last_index": int(first["segment_index"]),
            }
        ]
        for later_step in step_candidates[1:]:
            next_paths: list[dict[str, object]] = []
            for path in current_paths:
                for candidate in later_step:
                    if candidate["video_id"] != path["video_id"]:
                        continue
                    gap = int(candidate["segment_index"]) - int(path["last_index"])
                    if gap <= 0 or gap > max_gap:
                        continue
                    next_paths.append(
                        {
                            "video_id": path["video_id"],
                            "segment_ids": [*path["segment_ids"], candidate["segment_id"]],
                            "score": float(path["score"]) + float(candidate["score"]),
                            "last_index": int(candidate["segment_index"]),
                        }
                    )
            current_paths = next_paths
            if not current_paths:
                break
        paths.extend(current_paths)
    return sorted(paths, key=lambda item: float(item["score"]), reverse=True)
