from sqlalchemy import JSON

from pgvector.sqlalchemy import Vector

# PostgreSQL uses pgvector natively; SQLite falls back to JSON for local tests.
EmbeddingVector = Vector(dim=None).with_variant(JSON(), "sqlite")


def postgres_vector_bootstrap_sql(embedding_udt_name: str | None) -> list[str]:
    statements = ["CREATE EXTENSION IF NOT EXISTS vector"]
    if embedding_udt_name and embedding_udt_name != "vector":
        statements.append(
            "ALTER TABLE frame_embeddings "
            "ALTER COLUMN embedding TYPE vector USING embedding::text::vector"
        )
    statements.append("ALTER TABLE frames ADD COLUMN IF NOT EXISTS segment_id integer")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS embedding_branch_a vector")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS embedding_branch_b vector")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS ocr_tokens_json jsonb")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS object_counts_json jsonb")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS object_positions_json jsonb")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS semantic_entities_json jsonb")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS semantic_aliases_json jsonb")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS semantic_counts_json jsonb")
    statements.append("ALTER TABLE segments ADD COLUMN IF NOT EXISTS stage_failures_json jsonb")
    statements.append(
        "CREATE INDEX IF NOT EXISTS ix_frame_embeddings_frame_id_model_name "
        "ON frame_embeddings (frame_id, model_name)"
    )
    statements.append(
        "CREATE INDEX IF NOT EXISTS ix_segments_video_id_segment_index "
        "ON segments (video_id, segment_index)"
    )
    return statements
