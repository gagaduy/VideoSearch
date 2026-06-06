from sqlalchemy.orm import Session

from app.db.models import IndexJob, Video
from app.db.session import get_session_factory
from worker import tasks


def test_run_pending_jobs_processes_all_pending_jobs(monkeypatch) -> None:
    session: Session = get_session_factory()()
    try:
        video = Video(filename="queue.mp4", source_path="./data/videos/queue.mp4", status="pending")
        session.add(video)
        session.flush()
        first = IndexJob(video_id=video.id, status="pending", stage="queued", attempt_count=0)
        second = IndexJob(video_id=video.id, status="pending", stage="queued", attempt_count=0)
        session.add_all([first, second])
        session.commit()
    finally:
        session.close()

    called_job_ids: list[int] = []

    def _fake_run_job(db: Session, job_id: int) -> object:
        called_job_ids.append(job_id)
        job = db.get(IndexJob, job_id)
        assert job is not None
        job.status = "completed"
        job.stage = "done"
        db.commit()
        return job

    monkeypatch.setattr(tasks, "run_job", _fake_run_job)

    payload = tasks.run_pending_jobs()

    assert payload == {"processed_job_ids": called_job_ids, "processed_count": 2}
    assert called_job_ids == sorted(called_job_ids)
