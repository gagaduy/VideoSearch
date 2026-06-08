from app.services.question_search import build_question_evidence_terms, rank_question_candidates, score_evidence_row


def test_build_question_evidence_terms_extracts_textual_clues() -> None:
    question = "Ten cua loai virus nay la gi? Doan video ve viec che tao vaccine phong mot loai virus."

    terms = build_question_evidence_terms(question)

    assert "virus" in terms
    assert "vaccine" in terms


def test_score_evidence_row_prefers_ocr_overlap() -> None:
    row = {
        "caption_text": "scientist explaining a vaccine",
        "ocr_text": "virus vaccine trial information",
        "labels": [],
        "object_counts": {},
        "object_positions": {},
        "semantic_entities": [],
        "semantic_counts": {},
    }

    score = score_evidence_row(row, ["virus", "vaccine"])

    assert score > 0.5


def test_question_search_prefers_answer_bearing_text_rows() -> None:
    rows = [
        {
            "segment_id": 1,
            "ocr_text": "virus vaccine information appears here",
            "caption_text": "screen with text",
            "labels": [],
            "object_counts": {},
            "object_positions": {},
            "semantic_entities": [],
            "semantic_counts": {},
        },
        {
            "segment_id": 2,
            "ocr_text": "",
            "caption_text": "people talking in a room",
            "labels": ["person"],
            "object_counts": {"person": 2},
            "object_positions": {},
            "semantic_entities": [],
            "semantic_counts": {},
        },
    ]

    ranked = rank_question_candidates(rows, "What is the virus name?")

    assert ranked[0]["segment_id"] == 1
