from sqlalchemy.orm import Session

from app.db.session import get_session_factory
from app.db.repositories.jobs import create_job_record, get_job_by_id
from app.db.repositories.videos import create_video_record, get_video_by_id
from app.schemas.videos import VideoCreate
from app.services.index_queue import enqueue_index_job
from app.services.storage import ensure_data_dirs, reserve_video_path
from worker.pipeline import run_index_pipeline


def create_video_job(db: Session, payload: VideoCreate) -> tuple[object, object]:
    ensure_data_dirs()
    video = create_video_record(db, payload.filename, payload.source_path)
    job = create_job_record(db, int(video.id))
    enqueue_index_job(int(job.id))
    db.commit()
    db.refresh(video)
    db.refresh(job)
    return video, job


def create_uploaded_video_job(db: Session, filename: str, file_bytes: bytes) -> tuple[object, object]:
    destination = reserve_video_path(filename)
    destination.write_bytes(file_bytes)
    payload = VideoCreate(filename=destination.name, source_path=str(destination))
    return create_video_job(db, payload)


def run_job(db: Session, job_id: int) -> object:
    job = get_job_by_id(db, job_id)
    if job is None:
        raise ValueError(f"job {job_id} not found")
    video = get_video_by_id(db, int(job.video_id))
    if video is None:
        raise ValueError(f"video {job.video_id} not found")

    job.stage = "processing"
    job.status = "running"
    job.error_message = None
    job.attempt_count += 1
    db.commit()

    try:
        run_index_pipeline(db, video_id=int(video.id), job_id=int(job.id))
    except Exception as exc:
        db.rollback()
        failed_job = get_job_by_id(db, job_id)
        if failed_job is not None:
            failed_job.status = "failed"
            failed_job.stage = "error"
            failed_job.error_message = str(exc)
            db.commit()
        raise
    db.refresh(job)
    return job


def run_job_in_new_session(job_id: int) -> None:
    session = get_session_factory()()
    try:
        run_job(session, job_id)
    finally:
        session.close()
