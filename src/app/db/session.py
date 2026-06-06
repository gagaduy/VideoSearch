from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.base import Base
from app.db.vector import postgres_vector_bootstrap_sql


@lru_cache
def get_engine() -> Engine:
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(settings.database_url, future=True, connect_args=connect_args)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    from app.db import models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            embedding_udt_name = connection.execute(
                text(
                    """
                    SELECT udt_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'frame_embeddings'
                      AND column_name = 'embedding'
                    """
                )
            ).scalar_one_or_none()
            for statement in postgres_vector_bootstrap_sql(embedding_udt_name):
                connection.execute(text(statement))


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
