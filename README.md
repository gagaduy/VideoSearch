# Video Retrieval Quickstart

## 1. Vao thu muc project

```bash
cd /home/duy/AI_Project/draft_v1
```

## 2. Bat database

```bash
docker compose up -d postgres
docker compose ps
```

Phai thay container `postgres` o trang thai `running`.

## 3. Chay API

Mo `terminal 1`:

```bash
conda run -p ./.conda env PYTHONPATH=src uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Kiem tra nhanh:

```bash
curl http://localhost:8000/health
```

Neu OK thi se tra ve:

```json
{"status":"ok"}
```

## 4. Chay worker

Mo `terminal 2`:

```bash
conda run -p ./.conda env PYTHONPATH=src python -m worker.main
```

Worker co the dung yen neu chua co `job`. Do la binh thuong.

## 5. Don sach database

```bash
docker compose exec -T postgres psql -U video -d video_retrieval -c "TRUNCATE TABLE query_logs, index_jobs, frame_objects, frame_ocr, frame_captions, frame_embeddings, segments, frames, videos RESTART IDENTITY CASCADE;"
```

Kiem tra lai:

```bash
docker compose exec -T postgres psql -U video -d video_retrieval -c "select (select count(*) from videos) as videos, (select count(*) from frames) as frames, (select count(*) from segments) as segments;"
```

## 6. Import 1 clip nho de test

Dang ky clip:

```bash
curl -X POST http://localhost:8000/videos \
  -H 'Content-Type: application/json' \
  -d '{"filename":"sample.mp4","source_path":"/duong/dan/toi/sample.mp4"}'
```

Lenh tren se tra ve JSON co `job.id`.

Chay job:

```bash
curl -X POST http://localhost:8000/jobs/1/run
```

Thay `1` bang `job.id` that su.

Kiem tra job:

```bash
curl http://localhost:8000/jobs/1
```

## 7. Search thu

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"car near sign","object_labels":[]}'
```

## 8. Import keyframe dataset da extract san

Lenh import:

```bash
conda run -p ./.conda env PYTHONPATH=src python scripts/import_keyframe_dataset.py /home/duy/Downloads/keyframe
```

Luu y:

- Dataset nay rat lon, import full se rat lau.
- Khong duoc xuong dong sai cho. `python scripts/import_keyframe_dataset.py` phai nam tren cung mot lenh.
- Neu muon chay nen va luu log:

```bash
nohup conda run -p ./.conda env PYTHONPATH=src python scripts/import_keyframe_dataset.py /home/duy/Downloads/keyframe > import_keyframe.log 2>&1 &
tail -f import_keyframe.log
```

## 9. Kiem tra tien do import

Xem process:

```bash
pgrep -af 'import_keyframe_dataset.py'
```

Xem DB da co bao nhieu ban ghi:

```bash
docker compose exec -T postgres psql -U video -d video_retrieval -c "select count(*) as total_videos, count(*) filter (where status='indexed') as indexed_videos, count(*) filter (where status='pending') as pending_videos from videos;"
```

```bash
docker compose exec -T postgres psql -U video -d video_retrieval -c "select (select count(*) from frames) as frames, (select count(*) from segments) as segments;"
```

## 10. Neu muon tat

Tat API va worker bang `Ctrl+C` trong tung terminal.

Tat Postgres:

```bash
docker compose stop postgres
```
