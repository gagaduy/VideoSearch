from sqlalchemy.orm import Session

from app.db.models import IndexJob


def create_job_record(db: Session, video_id: int) -> IndexJob:
    job = IndexJob(video_id=video_id, status="pending", stage="queued", attempt_count=0)
    db.add(job)
    db.flush()
    return job


def get_job_by_id(db: Session, job_id: int) -> IndexJob | None:
    return db.get(IndexJob, job_id)
