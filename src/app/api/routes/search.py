import inspect
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import QuestionSearchRequest, SearchRequest, SearchResponse
from app.services.question_search import run_question_search
from app.services.search_service import run_image_search, run_search
from app.services.video_query_search import run_video_query_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    return SearchResponse.model_validate(run_search(db, payload.query, payload.object_labels))


@router.post("/search/question", response_model=SearchResponse)
def search_by_question(payload: QuestionSearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    return SearchResponse.model_validate(run_question_search(db, payload.question))


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> SearchResponse:
    if not str(file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    suffix = Path(file.filename or "query.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)

    try:
        return SearchResponse.model_validate(run_image_search(db, temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/search/video-query", response_model=SearchResponse)
async def search_by_video_query(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> SearchResponse:
    result = run_video_query_search(db, file)
    if inspect.isawaitable(result):
        result = await result
    return SearchResponse.model_validate(result)
