from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.jobs import JobRead
from app.schemas.videos import VideoCreate, VideoJobCreateResponse, VideoRead
from app.services.job_service import create_uploaded_video_job, create_video_job

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("", response_model=VideoJobCreateResponse, status_code=status.HTTP_201_CREATED)
def create_video(
    payload: VideoCreate,
    db: Session = Depends(get_db),
) -> VideoJobCreateResponse:
    video, job = create_video_job(db, payload)
    return VideoJobCreateResponse(
        video=VideoRead.model_validate(video),
        job=JobRead.model_validate(job),
    )


@router.post("/upload", response_model=VideoJobCreateResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> VideoJobCreateResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="missing filename")

    video, job = create_uploaded_video_job(db, file.filename, await file.read())
    return VideoJobCreateResponse(
        video=VideoRead.model_validate(video),
        job=JobRead.model_validate(job),
    )
