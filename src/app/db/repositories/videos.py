from sqlalchemy.orm import Session

from app.db.models import Video


def create_video_record(db: Session, filename: str, source_path: str) -> Video:
    video = Video(filename=filename, source_path=source_path, status="pending")
    db.add(video)
    db.flush()
    return video


def get_video_by_id(db: Session, video_id: int) -> Video | None:
    return db.get(Video, video_id)
