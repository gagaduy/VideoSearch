#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${PWD}/.conda"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not installed" >&2
  exit 1
fi

if [ ! -d "${ENV_PREFIX}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.11
fi

conda run -p "${ENV_PREFIX}" python -m pip install --upgrade pip
conda run -p "${ENV_PREFIX}" pip install -r requirements/base.txt -r requirements/dev.txt

mkdir -p data/videos data/frames data/thumbs

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Bootstrap complete. Local env: ${ENV_PREFIX}"

