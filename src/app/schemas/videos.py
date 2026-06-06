from pydantic import BaseModel

from app.schemas.jobs import JobRead


class VideoCreate(BaseModel):
    filename: str
    source_path: str


class VideoRead(BaseModel):
    id: int
    filename: str
    source_path: str
    status: str

    model_config = {"from_attributes": True}


class VideoJobCreateResponse(BaseModel):
    video: VideoRead
    job: JobRead
