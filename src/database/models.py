"""SQLAlchemy models for the RAG system with pgvector support."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.config.constants import ChunkType, DifficultyLevel, DocumentType
from src.config.settings import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Workspace(Base):
    """Workspace/domain for organizing knowledge bases."""

    __tablename__ = "workspaces"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    color: Mapped[str] = mapped_column(String(7), default="#00F0FF")  # Hex color
    icon: Mapped[str] = mapped_column(String(50), default="folder")  # Icon name
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="workspace", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_workspaces_name", "name"),
        Index("idx_workspaces_is_active", "is_active"),
    )


class Document(Base):
    """Document metadata table."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    doc_type: Mapped[DocumentType] = mapped_column(
        Enum(DocumentType), nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(String(1000))
    source: Mapped[Optional[str]] = mapped_column(String(1000))  # URL or path
    author: Mapped[Optional[str]] = mapped_column(String(500))
    publication_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)  # SHA-256
    raw_text_length: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB)  # Additional metadata
    upload_date: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Multi-tenant support (RBAC)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)

    # Workspace association
    workspace_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )

    # Relationships
    workspace: Mapped[Optional["Workspace"]] = relationship("Workspace", back_populates="documents")
    sections: Mapped[list["Section"]] = relationship(
        "Section", back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_documents_filename", "filename"),
        Index("idx_documents_doc_type", "doc_type"),
        Index("idx_documents_upload_date", "upload_date"),
        Index("idx_documents_is_active", "is_active"),
        Index("idx_documents_tenant_id", "tenant_id"),
    )


class Section(Base):
    """Document section/hierarchy table."""

    __tablename__ = "sections"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_section_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sections.id", ondelete="CASCADE")
    )
    heading: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 = H1, 2 = H2, etc.
    ordering: Mapped[int] = mapped_column(Integer, nullable=False)  # Position in document
    path: Mapped[str] = mapped_column(Text)  # Materialized path: "1.2.3"

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="sections")
    parent_section: Mapped[Optional["Section"]] = relationship(
        "Section", remote_side=[id], backref="child_sections"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="section", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_sections_document_id", "document_id"),
        Index("idx_sections_parent_section_id", "parent_section_id"),
        Index("idx_sections_level", "level"),
        Index("idx_sections_ordering", "ordering"),
    )


class Chunk(Base):
    """Text chunk table with vector embedding."""

    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    section_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sections.id", ondelete="CASCADE")
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[ChunkType] = mapped_column(
        Enum(ChunkType), default=ChunkType.PARAGRAPH
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # Position in document
    start_char: Mapped[int] = mapped_column(Integer)  # Character offset in original doc
    end_char: Mapped[int] = mapped_column(Integer)

    # Dense embedding (BGE-M3: 1024 dimensions)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dim), nullable=True
    )

    # Full-text search vector (populated via trigger)
    # Note: tsvector column created via migration

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    section: Mapped[Optional["Section"]] = relationship("Section", back_populates="chunks")
    enriched_metadata: Mapped[Optional["ChunkMetadata"]] = relationship(
        "ChunkMetadata", back_populates="chunk", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_section_id", "section_id"),
        Index("idx_chunks_chunk_type", "chunk_type"),
        Index("idx_chunks_chunk_index", "chunk_index"),
        # Vector index created via migration (HNSW or IVFFlat)
    )


class ChunkMetadata(Base):
    """Enriched metadata for chunks (generated by LLM)."""

    __tablename__ = "chunk_metadata"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    chunk_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # LLM-generated metadata
    summary: Mapped[Optional[str]] = mapped_column(Text)  # 1-line summary
    keywords: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String(100)))
    hypothetical_questions: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))
    entity_mentions: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String(200)))
    topic_tags: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String(100)))
    difficulty_level: Mapped[Optional[DifficultyLevel]] = mapped_column(
        Enum(DifficultyLevel)
    )
    temporal_references: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String(100)))

    # Hypothetical question embeddings (for HyDE search)
    # Store as JSONB array of vectors since pgvector doesn't support array of vectors
    hq_embeddings: Mapped[Optional[list]] = mapped_column(JSONB)

    # Additional computed fields
    medical_concepts: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String(200)))
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)  # Quality score of enrichment

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="enriched_metadata")

    __table_args__ = (
        Index("idx_chunk_metadata_chunk_id", "chunk_id"),
        Index("idx_chunk_metadata_difficulty", "difficulty_level"),
        Index("idx_chunk_metadata_keywords", "keywords", postgresql_using="gin"),
        Index("idx_chunk_metadata_topic_tags", "topic_tags", postgresql_using="gin"),
        Index("idx_chunk_metadata_entities", "entity_mentions", postgresql_using="gin"),
    )


class QueryLog(Base):
    """Log of queries for analysis and improvement."""

    __tablename__ = "query_logs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(settings.embedding_dim)
    )

    # Retrieval results
    retrieved_chunk_ids: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String(36)))
    retrieval_scores: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Float))
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    confidence_band: Mapped[Optional[str]] = mapped_column(String(20))

    # Correction loop info
    failure_mode: Mapped[Optional[str]] = mapped_column(String(50))
    correction_attempts: Mapped[int] = mapped_column(Integer, default=0)
    final_confidence: Mapped[Optional[float]] = mapped_column(Float)

    # Generation
    response_text: Mapped[Optional[str]] = mapped_column(Text)
    claims_extracted: Mapped[Optional[int]] = mapped_column(Integer)
    claims_grounded: Mapped[Optional[int]] = mapped_column(Integer)
    claims_ungrounded: Mapped[Optional[int]] = mapped_column(Integer)

    # Performance
    retrieval_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    generation_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # User feedback
    user_rating: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5
    user_feedback: Mapped[Optional[str]] = mapped_column(Text)

    # Metadata
    session_id: Mapped[Optional[str]] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    # Relationships
    trace: Mapped[Optional["QueryTrace"]] = relationship(
        "QueryTrace", back_populates="query_log", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_query_logs_created_at", "created_at"),
        Index("idx_query_logs_confidence_band", "confidence_band"),
        Index("idx_query_logs_failure_mode", "failure_mode"),
        Index("idx_query_logs_session_id", "session_id"),
    )


class QueryTrace(Base):
    """Detailed reasoning trace for a query - for frontend trace panel."""

    __tablename__ = "query_traces"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    query_log_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_logs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Step 1: Retrieval
    retrieval_results: Mapped[Optional[list]] = mapped_column(JSONB)
    # Format: [{chunk_id, text_preview, score, keywords, document_title}, ...]

    # Step 2: Confidence estimation
    confidence_components: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Format: {top_score, dropoff, coherence, coverage, overall}
    coverage_report: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Format: {covered_aspects: [], missing_aspects: [], gaps: []}

    # Step 3: Correction loop (if triggered)
    correction_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    diagnosis: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Format: {failure_mode, confidence, reasoning, suggested_actions}
    correction_actions: Mapped[Optional[list]] = mapped_column(JSONB)
    # Format: [{attempt, strategy, details, success, new_confidence}, ...]
    pre_correction_confidence: Mapped[Optional[float]] = mapped_column(Float)
    post_correction_confidence: Mapped[Optional[float]] = mapped_column(Float)

    # Step 4: Generation
    generation_prompt_type: Mapped[Optional[str]] = mapped_column(String(20))
    # "high_confidence", "medium_hedged", "abstention"
    citations_generated: Mapped[Optional[list]] = mapped_column(JSONB)
    # Format: [{index, chunk_id, document_title, relevance_score}, ...]

    # Step 5: Claim validation
    claims_validation: Mapped[Optional[list]] = mapped_column(JSONB)
    # Format: [{claim, status, confidence, supporting_chunks}, ...]
    # Status: GROUNDED, RECOVERED, UNGROUNDED
    hallucination_risk: Mapped[Optional[float]] = mapped_column(Float)
    revision_notes: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))

    # Timing breakdown (milliseconds)
    timing_breakdown: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Format: {embedding, retrieval, reranking, confidence, correction, generation, validation}

    # Cost tracking
    llm_calls: Mapped[Optional[list]] = mapped_column(JSONB)
    # Format: [{model, input_tokens, output_tokens, purpose}, ...]
    total_input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    total_output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_cost_usd: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    # Relationships
    query_log: Mapped["QueryLog"] = relationship("QueryLog", back_populates="trace")

    __table_args__ = (
        Index("idx_query_traces_query_log_id", "query_log_id"),
        Index("idx_query_traces_correction_triggered", "correction_triggered"),
        Index("idx_query_traces_created_at", "created_at"),
    )
