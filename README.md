# Video Retrieval Quickstart

README nay chi giu cac lenh chay hang ngay cho project.

## 1. Vao project

```bash
cd /home/duy/AI_Project/draft_v1
```

## 2. Bat cac service nen

### Chi can Postgres + Web

Day la mode dung nhieu nhat:

```bash
docker compose up -d postgres web
docker compose ps
```

Mac dinh:
- Web: `http://localhost:8080`
- Postgres: `localhost:5432`

## 3. Chay API local

Khuyen nghi chay API local bang env `.conda`, khong dung `--reload` khi test that.

```bash
PYTHONPATH=src ./.conda/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -sf http://localhost:8000/health
```

Ket qua dung:

```json
{"status":"ok"}
```

### Neu muon dung script

```bash
bash scripts/run_api.sh
```

Luu y: script nay can `conda` shell function hoat dong. Neu shell bao `conda: command not found` thi dung lenh Python local o tren.

## 4. Chay worker

Chi bat worker khi can import/index video moi.

```bash
PYTHONPATH=src ./.conda/bin/python -m worker.main
```

Hoac:

```bash
bash scripts/run_worker.sh
```

Neu chi test search tren du lieu da index san thi khong can worker.

## 5. Chay du an theo 2 mode thuong dung

### Mode A: Search-only

```bash
docker compose up -d postgres web
PYTHONPATH=src ./.conda/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mode nay nhe nhat de demo/search.

### Mode B: Import / Index / Enrich

Terminal 1:

```bash
docker compose up -d postgres web
PYTHONPATH=src ./.conda/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
PYTHONPATH=src ./.conda/bin/python -m worker.main
```

Mode nay dung khi import video moi.

## 6. Bat full Docker stack

Neu muon bat tat ca bang compose:

```bash
docker compose up -d
docker compose ps --all
```

Luu y:
- API trong Docker map ra cong `8001`
- Web van o `8080`
- Worker Docker nang hon, chi bat khi that su can

Health check Docker API:

```bash
curl -sf http://localhost:8001/health
```

## 7. Import video moi

Dang ky video:

```bash
curl -X POST http://localhost:8000/videos \
  -H 'Content-Type: application/json' \
  -d '{"filename":"sample.mp4","source_path":"/duong/dan/toi/sample.mp4"}'
```

Response se co `job.id`.

Chay job:

```bash
curl -X POST http://localhost:8000/jobs/1/run
```

Thay `1` bang `job.id` that.

Xem tien do:

```bash
curl http://localhost:8000/jobs/1
```

## 8. Search nhanh

### Text search

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"car near sign","object_labels":[]}'
```

### Video clip search

Mo UI o:

```text
http://localhost:8080
```

Roi dung o `Search By Video Clip`.

## 9. Reset du lieu DB

Xoa sach du lieu retrieval:

```bash
docker compose exec -T postgres psql -U video -d video_retrieval -c "TRUNCATE TABLE query_logs, index_jobs, frame_objects, frame_ocr, frame_captions, frame_embeddings, segments, frames, videos RESTART IDENTITY CASCADE;"
```

Kiem tra lai:

```bash
docker compose exec -T postgres psql -U video -d video_retrieval -c "select (select count(*) from videos) as videos, (select count(*) from frames) as frames, (select count(*) from segments) as segments;"
```

## 10. Tat project

### Neu chay local

- API / worker: `Ctrl+C` trong tung terminal

### Neu chay Docker

```bash
docker compose stop
```

Hoac chi tat DB + web:

```bash
docker compose stop postgres web
```

## 11. Troubleshooting nhanh

### API khong len

Kiem tra DB truoc:

```bash
docker compose ps
curl -sf http://localhost:8000/health
```

### Search duoc nhung import khong chay

Kha nang cao la chua bat worker.

### `conda: command not found`

Dung lenh truc tiep voi Python trong env:

```bash
PYTHONPATH=src ./.conda/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
PYTHONPATH=src ./.conda/bin/python -m worker.main
```

### May bi lag

- Tat worker neu chi search
- Khong dung `--reload` khi test that
- Tranh bat dong thoi API local, API docker va worker neu khong can
