from worker.adapters.openclip_adapter import OpenClipAdapter


def test_openclip_adapter_exposes_model_name() -> None:
    adapter = OpenClipAdapter(model_name="ViT-B-32")
    assert adapter.model_name == "ViT-B-32"

