"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-02
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "videos",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
    )
    op.create_table(
        "frames",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_id", sa.Integer(), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("timestamp_sec", sa.Float(), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("thumb_path", sa.Text(), nullable=False),
        sa.Column("is_keyframe", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_table(
        "frame_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("frame_id", sa.Integer(), sa.ForeignKey("frames.id"), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("embedding", Vector(dim=None), nullable=False),
    )
    op.create_index(
        "ix_frame_embeddings_frame_id_model_name",
        "frame_embeddings",
        ["frame_id", "model_name"],
        unique=False,
    )
    op.create_table(
        "frame_captions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("frame_id", sa.Integer(), sa.ForeignKey("frames.id"), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("caption", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("raw_json", sa.JSON(), nullable=True),
    )
    op.create_table(
        "frame_ocr",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("frame_id", sa.Integer(), sa.ForeignKey("frames.id"), nullable=False),
        sa.Column("engine_name", sa.String(length=100), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=True),
    )
    op.create_table(
        "frame_objects",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("frame_id", sa.Integer(), sa.ForeignKey("frames.id"), nullable=False),
        sa.Column("detector_name", sa.String(length=100), nullable=False),
        sa.Column("label", sa.String(length=100), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=True),
    )
    op.create_table(
        "index_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_id", sa.Integer(), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("stage", sa.String(length=50), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("expanded_queries", sa.JSON(), nullable=False),
        sa.Column("filters_json", sa.JSON(), nullable=True),
        sa.Column("weights_json", sa.JSON(), nullable=True),
        sa.Column("top_results_json", sa.JSON(), nullable=True),
    )
    op.create_table(
        "segments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_id", sa.Integer(), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("start_timestamp_sec", sa.Float(), nullable=False),
        sa.Column("end_timestamp_sec", sa.Float(), nullable=False),
        sa.Column("keyframe_id", sa.Integer(), nullable=True),
        sa.Column("caption_text", sa.Text(), nullable=False),
        sa.Column("ocr_text", sa.Text(), nullable=False),
        sa.Column("object_labels_json", sa.JSON(), nullable=True),
        sa.Column("object_counts_json", sa.JSON(), nullable=True),
        sa.Column("object_positions_json", sa.JSON(), nullable=True),
        sa.Column("semantic_entities_json", sa.JSON(), nullable=True),
        sa.Column("semantic_counts_json", sa.JSON(), nullable=True),
        sa.Column("embedding", Vector(dim=None), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_segments_video_id_segment_index",
        "segments",
        ["video_id", "segment_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("segments")
    op.drop_table("query_logs")
    op.drop_table("index_jobs")
    op.drop_table("frame_objects")
    op.drop_table("frame_ocr")
    op.drop_table("frame_captions")
    op.drop_table("frame_embeddings")
    op.drop_table("frames")
    op.drop_table("videos")
