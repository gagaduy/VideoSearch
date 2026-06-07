from worker.retrieval_ontology import build_indexing_prompts, normalize_query_object_terms


def test_build_indexing_prompts_contains_background_and_aliases() -> None:
    prompts = build_indexing_prompts()

    assert "person" in prompts
    assert "vehicle" in prompts
    assert "" in prompts


def test_normalize_query_object_terms_maps_aliases_to_canonical_terms() -> None:
    assert normalize_query_object_terms(["automobile", "ship"]) == ["car", "boat"]
