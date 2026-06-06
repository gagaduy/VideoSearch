from app.services.query_expansion import expand_query


def test_expand_query_returns_original_when_api_key_missing() -> None:
    expanded = expand_query("man fixing car", api_key="", model="gpt-4.1-mini")
    assert expanded == ["man fixing car"]

