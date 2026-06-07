from pathlib import Path

from app.services.object_refinement import refine_object_matches


def test_refine_object_matches_runs_query_conditioned_prompts(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class _Detector:
        def detect(self, image_path: str, classes: list[str] | None = None) -> list[dict[str, object]]:
            calls.setdefault("images", []).append(image_path)
            calls.setdefault("classes", []).append(classes)
            if image_path.endswith("frame_1.png"):
                return [{"label": "car", "score": 0.9, "bbox": [0, 0, 10, 10]}]
            return []

    monkeypatch.setattr("app.services.object_refinement.YoloDetectionAdapter", _Detector)

    first = tmp_path / "frame_1.png"
    second = tmp_path / "frame_2.png"
    first.write_bytes(b"frame-1")
    second.write_bytes(b"frame-2")

    scores = refine_object_matches(
        [
            {"frame_id": 11, "image_path": str(first)},
            {"frame_id": 12, "image_path": str(second)},
        ],
        ["automobile"],
    )

    assert scores == {11: 0.9}
    assert calls["images"] == [str(first), str(second)]
    assert calls["classes"] == [["car"], ["car"]]
