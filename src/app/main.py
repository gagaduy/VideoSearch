from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.media import router as media_router
from app.api.routes.search import router as search_router
from app.api.routes.videos import router as videos_router
from app.db.session import init_db
from app.services.search_service import prewarm_search_runtime


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    prewarm_search_runtime()
    yield


app = FastAPI(title="Video Retrieval API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(videos_router)
app.include_router(jobs_router)
app.include_router(media_router)
app.include_router(search_router)
