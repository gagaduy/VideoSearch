from app.services.temporal import expand_temporal_neighbors


def test_expand_temporal_neighbors_adds_adjacent_segment_context() -> None:
    ranked = [
        {"segment_id": 10, "segment_index": 4, "start_timestamp_sec": 10.0, "score": 0.9, "frame_id": 1},
    ]
    neighbors = {
        10: [
            {"segment_id": 11, "segment_index": 5, "start_timestamp_sec": 13.0, "frame_id": 2},
        ]
    }

    reranked = expand_temporal_neighbors(ranked, neighbors)

    assert reranked[0]["segment_id"] == 10
    assert reranked[1]["segment_id"] == 11


def test_expand_temporal_neighbors_preserves_existing_frame_metadata() -> None:
    ranked = [
        {
            "segment_id": 1,
            "segment_index": 1,
            "frame_id": 10,
            "start_timestamp_sec": 5.0,
            "score": 0.9,
            "image_url": "/media/frames/10/image",
            "thumb_url": "/media/frames/10/thumb",
        },
        {
            "segment_id": 2,
            "segment_index": 2,
            "frame_id": 20,
            "start_timestamp_sec": 15.0,
            "score": 0.2,
            "image_url": "/media/frames/20/image",
            "thumb_url": "/media/frames/20/thumb",
        },
    ]
    neighbors = {
        1: [
            {
                "segment_id": 2,
                "segment_index": 2,
                "start_timestamp_sec": 15.0,
                "end_timestamp_sec": 19.0,
                "keyframe_id": 20,
                "caption_text": "neighbor",
                "ocr_text": "",
                "labels": [],
            }
        ]
    }

    reranked = expand_temporal_neighbors(ranked, neighbors)

    segment_two = next(item for item in reranked if item["segment_id"] == 2)
    assert segment_two["frame_id"] == 20
    assert segment_two["image_url"] == "/media/frames/20/image"
    assert segment_two["thumb_url"] == "/media/frames/20/thumb"


def test_expand_temporal_neighbors_keeps_primary_results_ahead_of_neighbors() -> None:
    ranked = [
        {"segment_id": 10, "segment_index": 10, "start_timestamp_sec": 100.0, "score": 0.04, "frame_id": 10},
        {"segment_id": 20, "segment_index": 20, "start_timestamp_sec": 200.0, "score": 0.03, "frame_id": 20},
    ]
    neighbors = {
        10: [
            {"segment_id": 11, "segment_index": 11, "start_timestamp_sec": 101.0, "frame_id": 11},
        ]
    }

    reranked = expand_temporal_neighbors(ranked, neighbors, decay=3.5)

    assert [item["segment_id"] for item in reranked[:2]] == [10, 20]
    assert reranked[2]["segment_id"] == 11
    assert reranked[2]["source"] == "temporal_neighbor"
