FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libgl1 libglib2.0-0 tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements/base.txt requirements/worker.txt requirements/
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
RUN pip install -r requirements/base.txt -r requirements/worker.txt
COPY src /app/src
COPY scripts /app/scripts
CMD ["python", "-m", "worker.main"]
