import re


def normalize_ocr_lines(lines: list[list[str]]) -> str:
    text = " ".join(part.strip() for line in lines for part in line if part and part.strip())
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize_ocr_text(text: str) -> list[str]:
    seen: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token not in seen:
            seen.append(token)
    return seen
