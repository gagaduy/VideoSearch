#!/usr/bin/env bash
set -euo pipefail
conda run -p "${PWD}/.conda" env PYTHONPATH=src uvicorn app.main:app --host 0.0.0.0 --port 8000
