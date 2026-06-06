from sqlalchemy.dialects import postgresql, sqlite

from app.db.models import FrameEmbedding, Segment
from app.db.vector import postgres_vector_bootstrap_sql


def test_embedding_column_uses_vector_on_postgres_and_json_on_sqlite() -> None:
    embedding_type = FrameEmbedding.__table__.c.embedding.type

    postgres_sql = embedding_type.compile(dialect=postgresql.dialect())
    sqlite_sql = embedding_type.compile(dialect=sqlite.dialect())

    assert postgres_sql == "VECTOR"
    assert sqlite_sql == "JSON"


def test_postgres_bootstrap_adds_alter_when_legacy_json_column_detected() -> None:
    statements = postgres_vector_bootstrap_sql("json")

    assert any("CREATE EXTENSION IF NOT EXISTS vector" in statement for statement in statements)
    assert any("ALTER TABLE frame_embeddings" in statement for statement in statements)
    assert any("CREATE INDEX IF NOT EXISTS ix_frame_embeddings_frame_id_model_name" in statement for statement in statements)


def test_segment_model_exposes_multi_branch_fields() -> None:
    fields = Segment.__table__.columns.keys()

    assert "embedding_branch_a" in fields
    assert "embedding_branch_b" in fields
    assert "ocr_tokens_json" in fields
    assert "stage_failures_json" in fields


def test_postgres_bootstrap_sql_mentions_new_segment_columns() -> None:
    statements = postgres_vector_bootstrap_sql("vector")
    joined = "\n".join(statements)

    assert "embedding_branch_a" in joined
    assert "embedding_branch_b" in joined
    assert "ocr_tokens_json" in joined
    assert "stage_failures_json" in joined
