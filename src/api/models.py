"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(..., min_length=1, max_length=2000, description="The user query")
    document_ids: list[UUID] = Field(
        default_factory=list,
        description="Optional filter to specific documents",
    )
    include_trace: bool = Field(
        default=False,
        description="Include reasoning trace in response",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        le=4096,
        description="Maximum tokens in response",
    )


class Citation(BaseModel):
    """A citation reference in the response."""

    index: int
    chunk_id: str
    document_id: str
    document_title: str
    text_snippet: str
    relevance_score: float


class TraceEntry(BaseModel):
    """A single entry in the reasoning trace."""

    timestamp: float
    state: str
    message: str
    data: dict


class QueryResponse(BaseModel):
    """Query response model."""

    query_id: str
    response: str
    citations: list[Citation]
    confidence_score: float = Field(ge=0, le=1)
    confidence_band: str
    correction_attempts: int = Field(ge=0)
    trace: Optional[list[TraceEntry]] = None
    latency_ms: int


class IngestRequest(BaseModel):
    """Document ingestion request (for URL-based ingestion)."""

    url: Optional[str] = Field(None, description="URL to fetch document from")
    skip_enrichment: bool = Field(
        default=False,
        description="Skip LLM metadata enrichment for faster processing",
    )


class IngestResponse(BaseModel):
    """Ingestion response model."""

    document_id: str
    filename: str
    total_chunks: int
    total_tokens: int
    processing_time_ms: int
    success: bool
    error: Optional[str] = None


class DocumentSummary(BaseModel):
    """Summary information about a document."""

    id: str
    filename: str
    title: Optional[str]
    doc_type: str
    total_chunks: int
    total_tokens: int
    upload_date: datetime


class DocumentListResponse(BaseModel):
    """Response for document list endpoint."""

    total: int
    skip: int
    limit: int
    documents: list[DocumentSummary]


class ChunkResponse(BaseModel):
    """Detailed chunk information."""

    chunk_id: str
    document_id: str
    text: str
    token_count: int
    chunk_type: str
    section_path: list[str]
    summary: Optional[str] = None
    keywords: list[str] = []
    hypothetical_questions: list[str] = []
    entity_mentions: list[str] = []
    topic_tags: list[str] = []
    difficulty_level: Optional[str] = None


class SearchResult(BaseModel):
    """A single search result."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    document_title: Optional[str]
    keywords: list[str] = []


class SearchResponse(BaseModel):
    """Search endpoint response."""

    query: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: dict
    embedding_model: str
    version: str


class MetricsResponse(BaseModel):
    """System metrics response."""

    total_documents: int
    total_chunks: int
    total_queries: int
    avg_confidence: float
    avg_latency_ms: float
    hallucination_rate: float
    queries_by_confidence: dict[str, int] = Field(
        default_factory=dict,
        description="Query counts by confidence band",
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
