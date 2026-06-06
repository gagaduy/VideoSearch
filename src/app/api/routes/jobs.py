from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.repositories.jobs import get_job_by_id
from app.db.session import get_db
from app.schemas.jobs import JobRead
from app.services.job_service import run_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: int, db: Session = Depends(get_db)) -> JobRead:
    job = get_job_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return JobRead.model_validate(job)


@router.post("/{job_id}/run", response_model=JobRead)
def run_job_endpoint(job_id: int, db: Session = Depends(get_db)) -> JobRead:
    try:
        job = run_job(db, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobRead.model_validate(job)
