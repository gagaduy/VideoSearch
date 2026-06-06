from worker.adapters.paddleocr_adapter import PaddleOcrAdapter


def test_paddleocr_adapter_returns_engine_name() -> None:
    adapter = PaddleOcrAdapter()
    assert adapter.engine_name == "paddleocr"
