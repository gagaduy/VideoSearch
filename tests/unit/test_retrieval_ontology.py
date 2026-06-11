from worker.retrieval_ontology import build_indexing_prompts, canonicalize_object_label, normalize_query_object_terms


def test_build_indexing_prompts_contains_background_and_aliases() -> None:
    prompts = build_indexing_prompts()

    assert "person" in prompts
    assert "vehicle" in prompts
    assert "" in prompts


def test_build_indexing_prompts_compact_profile_is_smaller_but_keeps_core_terms() -> None:
    full_prompts = build_indexing_prompts()
    compact_prompts = build_indexing_prompts("compact")

    assert "person" in compact_prompts
    assert "car" in compact_prompts
    assert "vehicle" in compact_prompts
    assert "" in compact_prompts
    assert len(compact_prompts) < len(full_prompts)


def test_normalize_query_object_terms_maps_aliases_to_canonical_terms() -> None:
    assert normalize_query_object_terms(["automobile", "ship"]) == ["car", "boat"]


def test_canonicalize_object_label_maps_vehicle_variants() -> None:
    assert canonicalize_object_label("sports car") == "car"
    assert canonicalize_object_label("race car") == "car"
    assert canonicalize_object_label("supercar") == "car"
    assert canonicalize_object_label("plane") == "airplane"
    assert canonicalize_object_label("aircraft") == "airplane"
