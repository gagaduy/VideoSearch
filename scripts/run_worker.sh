#!/usr/bin/env bash
set -euo pipefail
conda run -p "${PWD}/.conda" env PYTHONPATH=src python -m worker.main

