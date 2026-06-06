from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import run_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    return SearchResponse.model_validate(run_search(db, payload.query, payload.object_labels))
