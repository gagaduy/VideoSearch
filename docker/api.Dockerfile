FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements/base.txt requirements/api.txt requirements/
RUN pip install --no-cache-dir -r requirements/base.txt -r requirements/api.txt
COPY src /app/src
COPY scripts /app/scripts
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000"]
