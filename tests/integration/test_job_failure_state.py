from sqlalchemy.orm import Session

from app.db.models import IndexJob, Video
from app.db.session import get_session_factory
from app.services import job_service


def test_run_job_marks_failed_when_pipeline_raises(monkeypatch) -> None:
    session: Session = get_session_factory()()
    try:
        video = Video(filename="broken.mp4", source_path="./data/videos/broken.mp4", status="pending")
        job = IndexJob(video_id=1, status="pending", stage="queued", attempt_count=0)
        session.add(video)
        session.flush()
        job.video_id = video.id
        session.add(job)
        session.commit()

        def _raise_pipeline(db: Session, video_id: int, job_id: int | None = None) -> dict[str, object]:
            raise RuntimeError("pipeline exploded")

        monkeypatch.setattr(job_service, "run_index_pipeline", _raise_pipeline)

        try:
            job_service.run_job(session, int(job.id))
        except RuntimeError:
            pass

        failed_job = session.get(IndexJob, job.id)
        assert failed_job is not None
        assert failed_job.status == "failed"
        assert failed_job.stage == "error"
        assert failed_job.error_message == "pipeline exploded"
        assert failed_job.attempt_count == 1
    finally:
        session.close()
