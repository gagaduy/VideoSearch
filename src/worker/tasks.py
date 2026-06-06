from sqlalchemy import select

from app.db.models import IndexJob
from app.db.session import get_session_factory, init_db
from app.services.job_service import run_job


def run_pending_jobs() -> dict[str, object]:
    init_db()
    session = get_session_factory()()
    try:
        pending_jobs = session.scalars(
            select(IndexJob).where(IndexJob.status == "pending").order_by(IndexJob.id.asc())
        ).all()
        processed_ids: list[int] = []
        for job in pending_jobs:
            run_job(session, int(job.id))
            processed_ids.append(int(job.id))
        return {"processed_job_ids": processed_ids, "processed_count": len(processed_ids)}
    finally:
        session.close()
