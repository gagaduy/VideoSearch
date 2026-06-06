from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.vector import EmbeddingVector


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(Text())
    duration_sec: Mapped[float | None] = mapped_column(Float(), nullable=True)
    fps: Mapped[float | None] = mapped_column(Float(), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    segment_id: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    timestamp_sec: Mapped[float] = mapped_column(Float())
    frame_index: Mapped[int] = mapped_column(Integer())
    image_path: Mapped[str] = mapped_column(Text())
    thumb_path: Mapped[str] = mapped_column(Text())
    is_keyframe: Mapped[bool] = mapped_column(Boolean(), default=True)


class Segment(Base):
    __tablename__ = "segments"
    __table_args__ = (
        Index("ix_segments_video_id_segment_index", "video_id", "segment_index"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    segment_index: Mapped[int] = mapped_column(Integer())
    start_timestamp_sec: Mapped[float] = mapped_column(Float())
    end_timestamp_sec: Mapped[float] = mapped_column(Float())
    keyframe_id: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    caption_text: Mapped[str] = mapped_column(Text(), default="")
    ocr_text: Mapped[str] = mapped_column(Text(), default="")
    ocr_tokens_json: Mapped[list[str] | None] = mapped_column(JSON(), nullable=True)
    object_labels_json: Mapped[list[str] | None] = mapped_column(JSON(), nullable=True)
    object_counts_json: Mapped[dict[str, int] | None] = mapped_column(JSON(), nullable=True)
    object_positions_json: Mapped[dict[str, list[str]] | None] = mapped_column(JSON(), nullable=True)
    semantic_entities_json: Mapped[list[dict[str, object]] | None] = mapped_column(JSON(), nullable=True)
    semantic_aliases_json: Mapped[dict[str, list[str]] | None] = mapped_column(JSON(), nullable=True)
    semantic_counts_json: Mapped[dict[str, int] | None] = mapped_column(JSON(), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector, nullable=True)
    embedding_branch_a: Mapped[list[float] | None] = mapped_column(EmbeddingVector, nullable=True)
    embedding_branch_b: Mapped[list[float] | None] = mapped_column(EmbeddingVector, nullable=True)
    stage_failures_json: Mapped[dict[str, str] | None] = mapped_column(JSON(), nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON(), nullable=True)


class FrameEmbedding(Base):
    __tablename__ = "frame_embeddings"
    __table_args__ = (
        Index("ix_frame_embeddings_frame_id_model_name", "frame_id", "model_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"))
    model_name: Mapped[str] = mapped_column(String(100))
    embedding: Mapped[list[float]] = mapped_column(EmbeddingVector)


class FrameCaption(Base):
    __tablename__ = "frame_captions"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"))
    model_name: Mapped[str] = mapped_column(String(100))
    caption: Mapped[str] = mapped_column(Text())
    confidence: Mapped[float | None] = mapped_column(Float(), nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON(), nullable=True)


class FrameOcr(Base):
    __tablename__ = "frame_ocr"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"))
    engine_name: Mapped[str] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text())
    raw_json: Mapped[dict | None] = mapped_column(JSON(), nullable=True)


class FrameObject(Base):
    __tablename__ = "frame_objects"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"))
    detector_name: Mapped[str] = mapped_column(String(100))
    label: Mapped[str] = mapped_column(String(100))
    score: Mapped[float] = mapped_column(Float())
    bbox: Mapped[list[float]] = mapped_column(JSON())
    raw_json: Mapped[dict | None] = mapped_column(JSON(), nullable=True)


class IndexJob(Base):
    __tablename__ = "index_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    stage: Mapped[str] = mapped_column(String(50), default="queued")
    error_message: Mapped[str | None] = mapped_column(Text(), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer(), default=0)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    original_query: Mapped[str] = mapped_column(Text())
    expanded_queries: Mapped[list[str]] = mapped_column(JSON())
    filters_json: Mapped[dict | None] = mapped_column(JSON(), nullable=True)
    weights_json: Mapped[dict | None] = mapped_column(JSON(), nullable=True)
    top_results_json: Mapped[list[dict[str, object]] | None] = mapped_column(JSON(), nullable=True)
