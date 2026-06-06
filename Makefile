.PHONY: bootstrap dev-api dev-worker test test-unit test-integration smoke gpu-check compose-up compose-db

CONDA_PREFIX := $(CURDIR)/.conda
CONDA_RUN := conda run -p $(CONDA_PREFIX)

bootstrap:
	bash scripts/bootstrap.sh

dev-api:
	$(CONDA_RUN) env PYTHONPATH=src uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-worker:
	$(CONDA_RUN) env PYTHONPATH=src python -m worker.main

test:
	$(CONDA_RUN) env PYTHONPATH=src pytest

test-unit:
	$(CONDA_RUN) env PYTHONPATH=src pytest tests/unit -v

test-integration:
	$(CONDA_RUN) env PYTHONPATH=src pytest tests/integration -v

smoke:
	$(CONDA_RUN) env PYTHONPATH=src pytest tests/smoke -v

gpu-check:
	$(CONDA_RUN) env PYTHONPATH=src python scripts/check_gpu.py

compose-up:
	docker compose up --build

compose-db:
	docker compose up -d postgres

