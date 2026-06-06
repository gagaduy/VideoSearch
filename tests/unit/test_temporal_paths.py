from app.services.temporal_paths import find_best_temporal_paths


def test_find_best_temporal_path_respects_step_order() -> None:
    steps = [
        [{"segment_id": 1, "video_id": 7, "segment_index": 2, "score": 0.8}],
        [{"segment_id": 2, "video_id": 7, "segment_index": 4, "score": 0.7}],
    ]

    path = find_best_temporal_paths(steps, max_gap=4)

    assert path[0]["segment_ids"] == [1, 2]


def test_find_best_temporal_path_rejects_reverse_order() -> None:
    steps = [
        [{"segment_id": 4, "video_id": 7, "segment_index": 5, "score": 0.9}],
        [{"segment_id": 3, "video_id": 7, "segment_index": 3, "score": 0.9}],
    ]

    assert find_best_temporal_paths(steps, max_gap=4) == []
