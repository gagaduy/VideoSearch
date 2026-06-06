from worker.adapters.internvl_adapter import InternvlAdapter


def test_internvl_adapter_parses_json_response(monkeypatch) -> None:
    adapter = InternvlAdapter(model_name="test-model", load_in_8bit=False)
    monkeypatch.setattr(
        adapter,
        "_run_prompt",
        lambda image_path, prompt: (
            '{"caption":"a red boat on the water","tags":["boat","water"],'
            '"entities":[{"label":"boat","aliases":["ship"],"attributes":["red"]}]}'
        ),
    )

    payload = adapter.describe_image("frame.png")

    assert payload["caption"] == "a red boat on the water"
    assert payload["tags"] == ["boat", "water"]
    assert payload["entities"][0]["label"] == "boat"
    assert payload["entities"][0]["aliases"] == ["ship"]


def test_internvl_adapter_falls_back_to_caption_text(monkeypatch) -> None:
    adapter = InternvlAdapter(model_name="test-model", load_in_8bit=False)
    responses = iter(["not json", "a blue car on a road"])
    monkeypatch.setattr(adapter, "_run_prompt", lambda image_path, prompt: next(responses))

    payload = adapter.describe_image("frame.png")

    assert payload["caption"] == "a blue car on a road"
    assert "blue" in payload["tags"]
