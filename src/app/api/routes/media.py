from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.models import Frame, Segment, Video
from app.db.session import get_db
from app.services.media_preview import build_preview_clip

router = APIRouter(prefix="/media", tags=["media"])


def _resolve_frame_file(frame: Frame, variant: str) -> Path:
    candidate = Path(frame.thumb_path if variant == "thumb" else frame.image_path).resolve()
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="media file not found")
    return candidate


def _build_preview_clip(frame: Frame, db: Session) -> Path:
    video = db.get(Video, int(frame.video_id))
    if video is None:
        raise HTTPException(status_code=404, detail=f"video {frame.video_id} not found")
    segment = db.get(Segment, int(frame.segment_id)) if frame.segment_id is not None else None
    return build_preview_clip(video, frame, segment)


@router.get("/frames/{frame_id}/{variant}")
def get_frame_media(frame_id: int, variant: str, db: Session = Depends(get_db)) -> FileResponse:
    if variant not in {"thumb", "image", "preview"}:
        raise HTTPException(status_code=404, detail="unknown media variant")

    frame = db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"frame {frame_id} not found")

    if variant == "preview":
        return FileResponse(
            _build_preview_clip(frame, db),
            media_type="video/mp4",
            headers={"Cache-Control": "no-store"},
        )

    return FileResponse(_resolve_frame_file(frame, variant))
