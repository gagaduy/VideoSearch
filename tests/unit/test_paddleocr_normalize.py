from worker.adapters.paddleocr_normalize import normalize_ocr_lines, tokenize_ocr_text


def test_normalize_ocr_lines_merges_and_cleans_text() -> None:
    lines = [["HELLO"], ["World"], ["  2025  "]]

    assert normalize_ocr_lines(lines) == "hello world 2025"


def test_tokenize_ocr_text_deduplicates_tokens() -> None:
    assert tokenize_ocr_text("boat boat shark 2025") == ["boat", "shark", "2025"]
