from worker.adapters.paddleocr_normalize import normalize_ocr_lines, tokenize_ocr_text


class PaddleOcrAdapter:
    def __init__(self, engine_name: str = "paddleocr") -> None:
        self.engine_name = engine_name
        self._engine = None

    def _lazy_load(self):
        if self._engine is not None:
            return self._engine
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return self._engine

    def extract_text(self, image_path: str) -> dict[str, object]:
        try:
            engine = self._lazy_load()
            result = engine.ocr(image_path, cls=True)
            lines = [[item[1][0] for item in block] for block in result if block]
            text = normalize_ocr_lines(lines)
            return {
                "text": text,
                "tokens": tokenize_ocr_text(text),
                "raw": result,
                "image_path": image_path,
            }
        except Exception:
            return {"text": "", "tokens": [], "raw": [], "image_path": image_path}
