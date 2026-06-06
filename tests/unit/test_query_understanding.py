from app.services.query_understanding import StructuredQuery, parse_structured_query


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _Client:
    def __init__(self, output_text: str) -> None:
        self.responses = self
        self._output_text = output_text

    def create(self, **_: object) -> _Response:
        return _Response(self._output_text)


def test_parse_structured_query_extracts_object_count_position_and_temporal_steps(monkeypatch) -> None:
    payload = """
    {
      "semantic_query": "fish swimming together in dark water",
      "semantic_queries": ["fish swimming together in dark water", "a pair of fish underwater"],
      "object_filters": [{"label": "fish", "min_count": 2, "regions": ["left", "right"]}],
      "must_terms": ["fish"],
      "soft_terms": ["dark water"],
      "temporal_steps": [
        {"text": "fish appear", "object_filters": [{"label": "fish", "min_count": 1}]},
        {"text": "fish swim together", "object_filters": [{"label": "fish", "min_count": 2}]}
      ]
    }
    """

    monkeypatch.setattr(
        "app.services.query_understanding.OpenAI",
        lambda api_key: _Client(payload),
    )

    parsed = parse_structured_query("two fish swimming together", api_key="test-key", model="gpt-4.1-mini")

    assert isinstance(parsed, StructuredQuery)
    assert parsed.semantic_query == "fish swimming together in dark water"
    assert parsed.object_filters[0].label == "fish"
    assert parsed.object_filters[0].min_count == 2
    assert parsed.object_filters[0].regions == ["left", "right"]
    assert [step.text for step in parsed.temporal_steps] == ["fish appear", "fish swim together"]


def test_parse_structured_query_falls_back_to_semantic_only_when_api_disabled() -> None:
    parsed = parse_structured_query("a transparent fish", api_key="", model="gpt-4.1-mini")

    assert parsed.semantic_query == "a transparent fish"
    assert parsed.semantic_queries == ["a transparent fish"]
    assert parsed.object_filters == []
    assert parsed.temporal_steps == []


def test_parse_structured_query_uses_local_temporal_and_object_heuristics_without_api() -> None:
    parsed = parse_structured_query(
        "first two fish on the left, then shark on the right",
        api_key="",
        model="gpt-4.1-mini",
    )

    assert parsed.semantic_query == "first two fish on the left, then shark on the right"
    assert parsed.object_filters[0].label == "fish"
    assert parsed.object_filters[0].min_count == 2
    assert "left" in parsed.object_filters[0].regions
    assert [step.text for step in parsed.temporal_steps] == ["two fish on the left", "shark on the right"]
