from pathlib import Path

from worker.adapters.openclip_adapter import OpenClipAdapter


def test_openclip_adapter_exposes_model_name() -> None:
    adapter = OpenClipAdapter(model_name="ViT-B-32")
    assert adapter.model_name == "ViT-B-32"


def test_openclip_adapter_can_embed_images_in_batch(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    adapter = OpenClipAdapter(model_name="ViT-B-32")
    results = adapter.embed_images([str(image_a), str(image_b)])

    assert len(results) == 2
    assert all(result.model_name == "ViT-B-32" for result in results)
