from worker.adapters import codetr_adapter


def test_codetr_runtime_probe_reports_missing_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(codetr_adapter, "init_detector", None)
    adapter = codetr_adapter.CoDetrDetectionAdapter(
        config_path="cfg.py",
        checkpoint_path="model.pth",
        runtime_backend="direct",
    )

    try:
        adapter._lazy_load()
    except RuntimeError as exc:
        assert "Co-DETR dependencies are unavailable" in str(exc)
    else:
        raise AssertionError("expected runtime dependency error")
