"""FastAPI application with RAG endpoints."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.connection import (
    init_db,
    close_db,
    get_db,
    check_db_health,
    DatabaseSession,
)
from src.ingestion.pipeline import IngestionPipeline
from src.agents.graph import RAGAgentGraph, RAGState
from src.embeddings.bge_m3 import preload_model
from src.config.telemetry import setup_telemetry
from src.api.middleware import (
    PromptInjectionMiddleware,
    TenantMiddleware,
    get_tenant_id,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Job status tracking for async ingestion
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory job storage (use Redis in production)
_ingestion_jobs: dict[str, dict] = {}


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[UUID] = Field(default_factory=list)
    workspace_id: Optional[str] = None
    include_trace: bool = False


class QueryResponse(BaseModel):
    """Query response model."""

    query_id: str
    response: str
    citations: list[dict]
    confidence_score: float
    confidence_band: str
    correction_attempts: int
    trace: Optional[list[dict]] = None
    latency_ms: int


class IngestResponse(BaseModel):
    """Ingestion response model (synchronous)."""

    document_id: str
    filename: str
    total_chunks: int
    total_tokens: int
    processing_time_ms: int
    success: bool
    error: Optional[str] = None


class IngestAsyncResponse(BaseModel):
    """Async ingestion response model."""

    job_id: str
    status: JobStatus
    message: str


class IngestJobStatusResponse(BaseModel):
    """Ingestion job status response."""

    job_id: str
    status: JobStatus
    filename: Optional[str] = None
    document_id: Optional[str] = None
    total_chunks: Optional[int] = None
    total_tokens: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class ChunkResponse(BaseModel):
    """Chunk detail response model."""

    chunk_id: str
    document_id: str
    text: str
    token_count: int
    chunk_type: str
    section_path: list[str]
    summary: Optional[str] = None
    keywords: list[str] = []
    hypothetical_questions: list[str] = []


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    database: dict
    embedding_model: str
    version: str


class MetricsResponse(BaseModel):
    """System metrics response model."""

    total_documents: int
    total_chunks: int
    total_queries: int
    avg_confidence: float
    avg_latency_ms: float
    hallucination_rate: float


class WorkspaceCreate(BaseModel):
    """Workspace creation request."""

    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    color: str = Field(default="#00F0FF", pattern=r"^#[0-9A-Fa-f]{6}$")
    icon: str = Field(default="folder")


class WorkspaceResponse(BaseModel):
    """Workspace response model."""

    id: str
    name: str
    description: Optional[str]
    color: str
    icon: str
    document_count: int
    created_at: str


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Self-Healing RAG Engine")

    await init_db()
    logger.info("Database initialized")

    # Optionally preload embedding model
    if not settings.debug:
        await preload_model()
        logger.info("Embedding model preloaded")

    yield

    # Shutdown
    await close_db()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Self-Healing RAG Engine with Confidence-Calibrated Retrieval",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup OpenTelemetry (if OTEL endpoint configured)
import os
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    setup_telemetry(app=app, enable_console=settings.debug)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(TenantMiddleware)
app.add_middleware(PromptInjectionMiddleware, block_on_detection=True, log_only=False)


# Dependency injection
db_session = DatabaseSession()


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db_health = await check_db_health()

    return HealthResponse(
        status="healthy" if db_health["status"] == "healthy" else "degraded",
        database=db_health,
        embedding_model=settings.embedding_model,
        version=settings.app_version,
    )


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    skip_enrichment: bool = Query(False, description="Skip LLM metadata enrichment"),
    workspace_id: Optional[str] = Query(None, description="Workspace ID to add document to"),
):
    """Upload and process a document.

    Supported formats: PDF, DOCX, Markdown, TXT

    Note: Does NOT hold database connection during embedding generation.
    DB session is only opened for the final save step.
    """
    logger.info("Ingesting document", filename=file.filename, workspace_id=workspace_id)

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    allowed_extensions = [".pdf", ".docx", ".doc", ".md", ".txt"]
    file_ext = "." + file.filename.split(".")[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}",
        )

    # Read file content
    content = await file.read()

    # Process document - pipeline manages its own DB session
    pipeline = IngestionPipeline(skip_enrichment=skip_enrichment)
    result = await pipeline.ingest_bytes(
        content=content,
        filename=file.filename,
        session=None,  # Let pipeline manage DB connection
        workspace_id=workspace_id,
    )

    return IngestResponse(
        document_id=str(result.document_id),
        filename=result.filename,
        total_chunks=result.total_chunks,
        total_tokens=result.total_tokens,
        processing_time_ms=result.processing_time_ms,
        success=result.success,
        error=result.error,
    )


async def _process_ingestion_job(
    job_id: str,
    content: bytes,
    filename: str,
    skip_enrichment: bool,
) -> None:
    """Background task for document ingestion."""
    _ingestion_jobs[job_id]["status"] = JobStatus.PROCESSING

    try:
        pipeline = IngestionPipeline(skip_enrichment=skip_enrichment)
        result = await pipeline.ingest_bytes(
            content=content,
            filename=filename,
            session=None,
        )

        _ingestion_jobs[job_id].update({
            "status": JobStatus.COMPLETED if result.success else JobStatus.FAILED,
            "document_id": str(result.document_id) if result.document_id else None,
            "total_chunks": result.total_chunks,
            "total_tokens": result.total_tokens,
            "processing_time_ms": result.processing_time_ms,
            "error": result.error,
            "completed_at": datetime.now(),
        })

        logger.info(
            "Background ingestion completed",
            job_id=job_id,
            success=result.success,
        )

    except Exception as e:
        logger.error("Background ingestion failed", job_id=job_id, error=str(e))
        _ingestion_jobs[job_id].update({
            "status": JobStatus.FAILED,
            "error": str(e),
            "completed_at": datetime.now(),
        })


@app.post("/ingest/async", response_model=IngestAsyncResponse)
@limiter.limit("10/minute")
async def ingest_document_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    skip_enrichment: bool = Query(False, description="Skip LLM metadata enrichment"),
):
    """Upload and process a document asynchronously.

    Returns immediately with a job ID. Use /ingest/{job_id}/status to check progress.

    Supported formats: PDF, DOCX, Markdown, TXT
    """
    logger.info("Starting async ingestion", filename=file.filename)

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    allowed_extensions = [".pdf", ".docx", ".doc", ".md", ".txt"]
    file_ext = "." + file.filename.split(".")[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}",
        )

    # Read file content
    content = await file.read()

    # Create job
    job_id = str(uuid4())
    _ingestion_jobs[job_id] = {
        "status": JobStatus.PENDING,
        "filename": file.filename,
        "created_at": datetime.now(),
    }

    # Schedule background processing
    background_tasks.add_task(
        _process_ingestion_job,
        job_id,
        content,
        file.filename,
        skip_enrichment,
    )

    return IngestAsyncResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Document '{file.filename}' queued for processing. Check /ingest/{job_id}/status for progress.",
    )


@app.get("/ingest/{job_id}/status", response_model=IngestJobStatusResponse)
async def get_ingestion_status(job_id: str):
    """Get the status of an async ingestion job."""
    if job_id not in _ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _ingestion_jobs[job_id]
    return IngestJobStatusResponse(
        job_id=job_id,
        status=job["status"],
        filename=job.get("filename"),
        document_id=job.get("document_id"),
        total_chunks=job.get("total_chunks"),
        total_tokens=job.get("total_tokens"),
        processing_time_ms=job.get("processing_time_ms"),
        error=job.get("error"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def query(
    request: Request,
    body: QueryRequest,
    session: AsyncSession = Depends(db_session),
):
    """Main RAG query endpoint.

    Performs retrieval, confidence estimation, optional correction loops,
    generation, and validation.
    """
    start_time = datetime.now()
    query_id = str(UUID(int=0))  # Generate proper UUID in production

    # Get tenant ID from request state (set by TenantMiddleware)
    tenant_id = get_tenant_id(request)

    logger.info(
        "Processing query",
        query=body.query[:50],
        document_filter=len(body.document_ids),
        tenant_id=tenant_id,
    )

    try:
        # Run the RAG agent
        agent = RAGAgentGraph()
        final_state: RAGState = await agent.run(
            query=body.query,
            session=session,
            document_ids=body.document_ids if body.document_ids else None,
            tenant_id=tenant_id,
        )

        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return QueryResponse(
            query_id=query_id,
            response=final_state.response,
            citations=final_state.citations,
            confidence_score=final_state.confidence.score if final_state.confidence else 0.0,
            confidence_band=final_state.confidence.band.value if final_state.confidence else "low",
            correction_attempts=final_state.correction_attempts,
            trace=final_state.trace if body.include_trace else None,
            latency_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_rag_response(
    query: str,
    session: AsyncSession,
    document_ids: Optional[list[UUID]],
    tenant_id: Optional[str],
    workspace_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE events for streaming RAG response.

    Event types:
    - retrieval: Search results found
    - confidence: Confidence estimation complete
    - correction: Correction loop executed
    - generation: Response chunk
    - validation: Claim validation result
    - done: Stream complete
    - error: Error occurred
    """
    start_time = datetime.now()
    query_id = str(uuid4())

    def sse_event(event_type: str, data: dict) -> str:
        """Format data as SSE event."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    try:
        # Create a custom streaming RAG workflow
        from src.retrieval.search import HybridSearcher
        from src.retrieval.reranker import get_reranker
        from src.retrieval.confidence import ConfidenceEstimator
        from src.config.constants import ConfidenceBand
        from src.generation.generator import ResponseGenerator

        searcher = HybridSearcher()
        confidence_estimator = ConfidenceEstimator()

        # Step 1: Retrieval
        yield sse_event("status", {"step": "retrieval", "message": "Searching knowledge base..."})

        results = await searcher.search(
            query, session, document_ids=document_ids, tenant_id=tenant_id, workspace_id=workspace_id
        )

        reranker = get_reranker()
        reranked = await reranker.rerank(query, results, top_k=settings.rerank_top_k)

        yield sse_event("retrieval", {
            "results_count": len(reranked),
            "top_documents": [
                {"title": r.document_title, "score": round(r.score, 3)}
                for r in reranked[:3]
            ],
        })

        # Step 2: Confidence estimation
        yield sse_event("status", {"step": "confidence", "message": "Estimating confidence..."})

        confidence = await confidence_estimator.estimate(query, reranked)

        yield sse_event("confidence", {
            "score": round(confidence.score, 3),
            "band": confidence.band.value,
            "components": {k: round(v, 3) for k, v in confidence.components.items()},
        })

        # Step 3: Generation (simplified - no correction loop for streaming)
        yield sse_event("status", {"step": "generation", "message": "Generating response..."})

        generator = ResponseGenerator()

        if confidence.band == ConfidenceBand.LOW:
            # Abstain
            response = "I don't have enough confident information to answer this question reliably."
            if confidence.gaps:
                response += f" Identified gaps: {', '.join(confidence.gaps[:2])}."
            citations = []
            yield sse_event("generation", {"type": "abstention", "chunk": response})
        else:
            response, citations = await generator.generate(
                query=query,
                context_chunks=reranked,
                confidence_band=confidence.band,
            )

            # Stream the response in chunks for UI effect
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                yield sse_event("generation", {"type": "chunk", "chunk": chunk})
                await asyncio.sleep(0.02)  # Small delay for streaming effect

        # Step 4: Validation
        yield sse_event("status", {"step": "validation", "message": "Validating claims..."})

        try:
            from src.validation.claims import ClaimValidator
            validator = ClaimValidator()
            validation_result = await validator.validate_response(
                query=query,
                response=response,
                context_chunks=reranked,
            )

            yield sse_event("validation", {
                "total_claims": validation_result.get("total_claims", 0),
                "grounded_claims": validation_result.get("grounded_claims", 0),
                "hallucination_risk": round(validation_result.get("hallucination_risk", 0), 3),
            })
        except Exception as e:
            logger.warning("Validation skipped in streaming", error=str(e))
            yield sse_event("validation", {"skipped": True, "reason": str(e)})

        # Done
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        yield sse_event("done", {
            "query_id": query_id,
            "latency_ms": elapsed_ms,
            "citations": [
                {"index": i + 1, "document_title": c.get("document_title", "Unknown")}
                for i, c in enumerate(citations[:5])
            ] if citations else [],
        })

    except Exception as e:
        logger.error("Streaming query failed", error=str(e))
        yield sse_event("error", {"message": str(e)})


@app.post("/query/stream")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def query_stream(
    request: Request,
    body: QueryRequest,
    session: AsyncSession = Depends(db_session),
):
    """Streaming RAG query endpoint using Server-Sent Events.

    Returns SSE events for each step of the RAG pipeline:
    - retrieval: Search results
    - confidence: Confidence estimation
    - generation: Response chunks
    - validation: Claim validation
    - done: Complete with citations
    """
    tenant_id = get_tenant_id(request)

    logger.info(
        "Processing streaming query",
        query=body.query[:50],
        document_filter=len(body.document_ids),
        tenant_id=tenant_id,
    )

    return StreamingResponse(
        _stream_rag_response(
            query=body.query,
            session=session,
            document_ids=body.document_ids if body.document_ids else None,
            tenant_id=tenant_id,
            workspace_id=body.workspace_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/query/{query_id}/trace")
async def get_query_trace(
    query_id: UUID,
    session: AsyncSession = Depends(db_session),
):
    """Get the reasoning trace for a previous query."""
    from src.database.trace import get_trace

    trace = await get_trace(session, query_id)

    if not trace:
        raise HTTPException(
            status_code=404,
            detail="Trace not found for this query ID",
        )

    return trace


@app.get("/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    chunk_id: UUID,
    session: AsyncSession = Depends(db_session),
):
    """Get details for a specific chunk."""
    from sqlalchemy import select
    from src.database.models import Chunk, ChunkMetadata, Section

    result = await session.execute(
        select(Chunk, ChunkMetadata, Section)
        .outerjoin(ChunkMetadata, Chunk.id == ChunkMetadata.chunk_id)
        .outerjoin(Section, Chunk.section_id == Section.id)
        .where(Chunk.id == chunk_id)
    )
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail="Chunk not found")

    chunk, metadata, section = row

    return ChunkResponse(
        chunk_id=str(chunk.id),
        document_id=str(chunk.document_id),
        text=chunk.text,
        token_count=chunk.token_count,
        chunk_type=chunk.chunk_type.value if chunk.chunk_type else "paragraph",
        section_path=section.path.split("/") if section and section.path else [],
        summary=metadata.summary if metadata else None,
        keywords=metadata.keywords if metadata else [],
        hypothetical_questions=metadata.hypothetical_questions if metadata else [],
    )


@app.get("/documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    workspace_id: Optional[str] = Query(None, description="Filter by workspace ID"),
    session: AsyncSession = Depends(db_session),
):
    """List all documents, optionally filtered by workspace."""
    from sqlalchemy import select, func
    from src.database.models import Document

    # Build base query
    base_query = select(Document).where(Document.is_active == True)
    count_query = select(func.count()).select_from(Document).where(Document.is_active == True)

    # Filter by workspace if provided
    if workspace_id:
        base_query = base_query.where(Document.workspace_id == workspace_id)
        count_query = count_query.where(Document.workspace_id == workspace_id)

    # Get total count
    count_result = await session.execute(count_query)
    total = count_result.scalar()

    # Get documents
    result = await session.execute(
        base_query
        .order_by(Document.upload_date.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = result.scalars().all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "documents": [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "title": doc.title,
                "doc_type": doc.doc_type.value,
                "total_chunks": doc.total_chunks,
                "total_tokens": doc.total_tokens,
                "upload_date": doc.upload_date.isoformat(),
                "workspace_id": str(doc.workspace_id) if doc.workspace_id else None,
            }
            for doc in documents
        ],
    }


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: UUID,
    session: AsyncSession = Depends(db_session),
):
    """Delete a document and all its chunks."""
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    deleted = await pipeline.delete_document(document_id, session)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "deleted", "document_id": str(document_id)}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    session: AsyncSession = Depends(db_session),
):
    """Get aggregate system metrics."""
    from sqlalchemy import select, func
    from src.database.models import Document, Chunk, QueryLog

    # Document and chunk counts
    doc_count = await session.execute(
        select(func.count()).select_from(Document).where(Document.is_active == True)
    )
    chunk_count = await session.execute(select(func.count()).select_from(Chunk))

    # Query metrics
    query_count = await session.execute(select(func.count()).select_from(QueryLog))
    avg_confidence = await session.execute(
        select(func.avg(QueryLog.final_confidence)).where(
            QueryLog.final_confidence.isnot(None)
        )
    )
    avg_latency = await session.execute(
        select(func.avg(QueryLog.total_latency_ms)).where(
            QueryLog.total_latency_ms.isnot(None)
        )
    )

    # Hallucination rate
    hallucination_result = await session.execute(
        select(
            func.sum(QueryLog.claims_ungrounded).label("ungrounded"),
            func.sum(QueryLog.claims_extracted).label("total"),
        )
    )
    h_row = hallucination_result.first()
    hallucination_rate = 0.0
    if h_row and h_row.total and h_row.total > 0:
        hallucination_rate = (h_row.ungrounded or 0) / h_row.total

    return MetricsResponse(
        total_documents=doc_count.scalar() or 0,
        total_chunks=chunk_count.scalar() or 0,
        total_queries=query_count.scalar() or 0,
        avg_confidence=avg_confidence.scalar() or 0.0,
        avg_latency_ms=avg_latency.scalar() or 0.0,
        hallucination_rate=hallucination_rate,
    )


@app.get("/workspaces")
async def list_workspaces(
    session: AsyncSession = Depends(db_session),
):
    """List all workspaces."""
    from sqlalchemy import select, func
    from src.database.models import Workspace, Document

    # Get workspaces with document counts
    result = await session.execute(
        select(
            Workspace,
            func.count(Document.id).label("doc_count")
        )
        .outerjoin(Document, (Document.workspace_id == Workspace.id) & (Document.is_active == True))
        .where(Workspace.is_active == True)
        .group_by(Workspace.id)
        .order_by(Workspace.created_at.desc())
    )
    rows = result.all()

    return {
        "workspaces": [
            {
                "id": str(ws.id),
                "name": ws.name,
                "description": ws.description,
                "color": ws.color,
                "icon": ws.icon,
                "document_count": doc_count,
                "created_at": ws.created_at.isoformat(),
            }
            for ws, doc_count in rows
        ]
    }


@app.post("/workspaces", response_model=WorkspaceResponse)
async def create_workspace(
    body: WorkspaceCreate,
    session: AsyncSession = Depends(db_session),
):
    """Create a new workspace."""
    from src.database.models import Workspace

    workspace = Workspace(
        name=body.name,
        description=body.description,
        color=body.color,
        icon=body.icon,
    )
    session.add(workspace)
    await session.commit()
    await session.refresh(workspace)

    return WorkspaceResponse(
        id=str(workspace.id),
        name=workspace.name,
        description=workspace.description,
        color=workspace.color,
        icon=workspace.icon,
        document_count=0,
        created_at=workspace.created_at.isoformat(),
    )


@app.get("/workspaces/{workspace_id}")
async def get_workspace(
    workspace_id: UUID,
    session: AsyncSession = Depends(db_session),
):
    """Get a workspace by ID."""
    from sqlalchemy import select, func
    from src.database.models import Workspace, Document

    result = await session.execute(
        select(
            Workspace,
            func.count(Document.id).label("doc_count")
        )
        .outerjoin(Document, (Document.workspace_id == Workspace.id) & (Document.is_active == True))
        .where(Workspace.id == workspace_id)
        .where(Workspace.is_active == True)
        .group_by(Workspace.id)
    )
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")

    ws, doc_count = row
    return {
        "id": str(ws.id),
        "name": ws.name,
        "description": ws.description,
        "color": ws.color,
        "icon": ws.icon,
        "document_count": doc_count,
        "created_at": ws.created_at.isoformat(),
    }


@app.put("/workspaces/{workspace_id}")
async def update_workspace(
    workspace_id: UUID,
    body: WorkspaceCreate,
    session: AsyncSession = Depends(db_session),
):
    """Update a workspace."""
    from sqlalchemy import select
    from src.database.models import Workspace

    result = await session.execute(
        select(Workspace).where(Workspace.id == workspace_id).where(Workspace.is_active == True)
    )
    workspace = result.scalar_one_or_none()

    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    workspace.name = body.name
    workspace.description = body.description
    workspace.color = body.color
    workspace.icon = body.icon

    await session.commit()

    return {"status": "updated", "workspace_id": str(workspace_id)}


@app.delete("/workspaces/{workspace_id}")
async def delete_workspace(
    workspace_id: UUID,
    session: AsyncSession = Depends(db_session),
):
    """Delete a workspace (soft delete)."""
    from sqlalchemy import select
    from src.database.models import Workspace

    result = await session.execute(
        select(Workspace).where(Workspace.id == workspace_id).where(Workspace.is_active == True)
    )
    workspace = result.scalar_one_or_none()

    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    workspace.is_active = False
    await session.commit()

    return {"status": "deleted", "workspace_id": str(workspace_id)}


@app.get("/workspaces/{workspace_id}/documents")
async def list_workspace_documents(
    workspace_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(db_session),
):
    """List documents in a workspace."""
    from sqlalchemy import select, func
    from src.database.models import Document

    # Get total count
    count_result = await session.execute(
        select(func.count()).select_from(Document)
        .where(Document.workspace_id == workspace_id)
        .where(Document.is_active == True)
    )
    total = count_result.scalar()

    # Get documents
    result = await session.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .where(Document.is_active == True)
        .order_by(Document.upload_date.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = result.scalars().all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "documents": [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "title": doc.title,
                "doc_type": doc.doc_type.value,
                "total_chunks": doc.total_chunks,
                "total_tokens": doc.total_tokens,
                "upload_date": doc.upload_date.isoformat(),
            }
            for doc in documents
        ],
    }


@app.post("/search")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def search(
    request: Request,
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=50),
    rerank: bool = Query(True),
    session: AsyncSession = Depends(db_session),
):
    """Direct search endpoint (without full RAG pipeline)."""
    from src.retrieval.search import HybridSearcher
    from src.retrieval.reranker import get_reranker

    # Get tenant ID from request state (set by TenantMiddleware)
    tenant_id = get_tenant_id(request)

    searcher = HybridSearcher()
    results = await searcher.search(
        query, session, top_k=top_k * 2 if rerank else top_k, tenant_id=tenant_id
    )

    if rerank and results:
        reranker = get_reranker()
        results = await reranker.rerank(query, results, top_k=top_k)

    return {
        "query": query,
        "results": [
            {
                "chunk_id": str(r.chunk_id),
                "document_id": str(r.document_id),
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                "score": r.score,
                "rank": r.rank,
                "document_title": r.document_title,
                "keywords": r.keywords[:5] if r.keywords else [],
            }
            for r in results
        ],
    }


def run_server():
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run_server()
