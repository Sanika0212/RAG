"""Hybrid search combining vector, keyword, and HyDE search."""

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import numpy as np
import structlog
from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.models import Chunk, ChunkMetadata, Document, Section
from src.embeddings.bge_m3 import BGEEmbedder, get_embedder

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class SearchResult:
    """A single search result."""

    chunk_id: UUID
    document_id: UUID
    text: str
    score: float
    rank: int

    # Source-specific scores
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    hyde_score: Optional[float] = None

    # Metadata
    section_path: list[str] = field(default_factory=list)
    chunk_type: str = ""
    token_count: int = 0
    document_title: Optional[str] = None

    # Enriched metadata
    summary: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    hypothetical_questions: list[str] = field(default_factory=list)


class HybridSearcher:
    """Perform hybrid search across vector, keyword, and HyDE indices."""

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        top_k: int = settings.retrieval_top_k,
        weights: Optional[dict[str, float]] = None,
    ):
        """Initialize the hybrid searcher.

        Args:
            embedder: BGE embedder instance
            top_k: Number of results to retrieve from each source
            weights: Weights for hybrid fusion (vector, keyword, hyde)
        """
        self.embedder = embedder or get_embedder()
        self.top_k = top_k
        self.weights = weights or settings.hybrid_search_weights

    async def search(
        self,
        query: str,
        session: AsyncSession,
        top_k: Optional[int] = None,
        document_ids: Optional[list[UUID]] = None,
        filters: Optional[dict] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Perform hybrid search.

        Args:
            query: Search query
            session: Database session
            top_k: Override default top_k
            document_ids: Filter to specific documents
            filters: Additional filters (topic_tags, difficulty, etc.)
            tenant_id: Tenant ID for multi-tenant filtering (RBAC)
            workspace_id: Filter to specific workspace

        Returns:
            List of SearchResult objects, ranked by fused score
        """
        top_k = top_k or self.top_k
        logger.info(
            "Performing hybrid search",
            query=query[:100],
            top_k=top_k,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query, return_sparse=True)

        # Perform searches sequentially (asyncpg doesn't support parallel ops on same session)
        vector_results = await self._vector_search(
            query_embedding.dense,
            session,
            top_k,
            document_ids,
            filters,
            tenant_id,
            workspace_id,
        )
        keyword_results = await self._keyword_search(
            query, session, top_k, document_ids, filters, tenant_id, workspace_id
        )
        hyde_results = await self._hyde_search(
            query_embedding.dense, session, top_k, document_ids, tenant_id, workspace_id
        )

        # Fuse results using Reciprocal Rank Fusion
        fused_results = self._fuse_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            hyde_results=hyde_results,
            top_k=top_k,
        )

        logger.info(
            "Hybrid search complete",
            vector_hits=len(vector_results),
            keyword_hits=len(keyword_results),
            hyde_hits=len(hyde_results),
            fused_results=len(fused_results),
        )

        return fused_results

    async def _vector_search(
        self,
        query_embedding: np.ndarray,
        session: AsyncSession,
        top_k: int,
        document_ids: Optional[list[UUID]] = None,
        filters: Optional[dict] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Perform vector similarity search using pgvector."""
        # Build query with cosine distance
        embedding_list = query_embedding.tolist()

        # Build query string with conditional clauses
        query_str = """
            SELECT
                c.id as chunk_id,
                c.document_id,
                c.text,
                c.chunk_type,
                c.token_count,
                1 - (c.embedding <=> :embedding) as score,
                d.title as document_title,
                s.path as section_path,
                cm.summary,
                cm.keywords,
                cm.hypothetical_questions
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            LEFT JOIN sections s ON c.section_id = s.id
            LEFT JOIN chunk_metadata cm ON c.id = cm.chunk_id
            WHERE d.is_active = true
            AND c.embedding IS NOT NULL
        """

        # Add tenant filter (RBAC)
        if tenant_id:
            query_str += " AND d.tenant_id = :tenant_id"

        # Add workspace filter
        if workspace_id:
            query_str += " AND d.workspace_id = :workspace_id"

        # Add document filter
        if document_ids:
            query_str += " AND c.document_id = ANY(:doc_ids)"

        # Add metadata filters
        if filters:
            if filters.get("topic_tags"):
                query_str += " AND cm.topic_tags && :topic_tags"
            if filters.get("difficulty_level"):
                query_str += " AND cm.difficulty_level = :difficulty"

        query_str += " ORDER BY c.embedding <=> :embedding LIMIT :limit"

        params = {"embedding": str(embedding_list), "limit": top_k}
        if tenant_id:
            params["tenant_id"] = tenant_id
        if workspace_id:
            params["workspace_id"] = workspace_id
        if document_ids:
            params["doc_ids"] = [str(d) for d in document_ids]
        if filters:
            if filters.get("topic_tags"):
                params["topic_tags"] = filters["topic_tags"]
            if filters.get("difficulty_level"):
                params["difficulty"] = filters["difficulty_level"]

        query = text(query_str).bindparams(**params)
        result = await session.execute(query)
        rows = result.fetchall()

        return [
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                text=row.text,
                score=row.score,
                vector_score=row.score,
                rank=i + 1,
                section_path=row.section_path.split("/") if row.section_path else [],
                chunk_type=row.chunk_type,
                token_count=row.token_count,
                document_title=row.document_title,
                summary=row.summary,
                keywords=row.keywords or [],
                hypothetical_questions=row.hypothetical_questions or [],
            )
            for i, row in enumerate(rows)
        ]

    async def _keyword_search(
        self,
        query: str,
        session: AsyncSession,
        top_k: int,
        document_ids: Optional[list[UUID]] = None,
        filters: Optional[dict] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Perform full-text keyword search using PostgreSQL tsvector."""
        # Build tsquery from natural language query
        query_str = """
            SELECT
                c.id as chunk_id,
                c.document_id,
                c.text,
                c.chunk_type,
                c.token_count,
                ts_rank_cd(c.search_vector, plainto_tsquery('english', :query)) as score,
                d.title as document_title,
                s.path as section_path,
                cm.summary,
                cm.keywords
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            LEFT JOIN sections s ON c.section_id = s.id
            LEFT JOIN chunk_metadata cm ON c.id = cm.chunk_id
            WHERE d.is_active = true
            AND c.search_vector @@ plainto_tsquery('english', :query)
        """

        # Add tenant filter (RBAC)
        if tenant_id:
            query_str += " AND d.tenant_id = :tenant_id"

        # Add workspace filter
        if workspace_id:
            query_str += " AND d.workspace_id = :workspace_id"

        # Add document filter
        if document_ids:
            query_str += " AND c.document_id = ANY(:doc_ids)"

        query_str += " ORDER BY score DESC LIMIT :limit"

        params = {"query": query, "limit": top_k}
        if tenant_id:
            params["tenant_id"] = tenant_id
        if workspace_id:
            params["workspace_id"] = workspace_id
        if document_ids:
            params["doc_ids"] = [str(d) for d in document_ids]

        query_sql = text(query_str).bindparams(**params)
        result = await session.execute(query_sql)
        rows = result.fetchall()

        return [
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                text=row.text,
                score=row.score,
                keyword_score=row.score,
                rank=i + 1,
                section_path=row.section_path.split("/") if row.section_path else [],
                chunk_type=row.chunk_type,
                token_count=row.token_count,
                document_title=row.document_title,
                summary=row.summary,
                keywords=row.keywords or [],
            )
            for i, row in enumerate(rows)
        ]

    async def _hyde_search(
        self,
        query_embedding: np.ndarray,
        session: AsyncSession,
        top_k: int,
        document_ids: Optional[list[UUID]] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search against hypothetical question embeddings (HyDE-style)."""
        # This searches the hq_embeddings JSONB field
        # For simplicity, we'll use a stored procedure approach
        # In production, you might precompute and store HQ embeddings in a separate table

        embedding_list = query_embedding.tolist()

        # Query chunks that have hypothetical questions and compute similarity
        # Note: This is a simplified approach; for production, store HQ embeddings separately
        query_str = """
            WITH hq_similarities AS (
                SELECT
                    c.id as chunk_id,
                    c.document_id,
                    c.text,
                    c.chunk_type,
                    c.token_count,
                    d.title as document_title,
                    s.path as section_path,
                    cm.summary,
                    cm.hypothetical_questions,
                    -- Compute max similarity across HQ embeddings
                    (
                        SELECT MAX(1 - (hq_vec::vector <=> cast(:embedding as vector)))
                        FROM jsonb_array_elements(cm.hq_embeddings) as hq_vec
                        WHERE cm.hq_embeddings IS NOT NULL
                    ) as max_hq_score
                FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.id
                LEFT JOIN sections s ON c.section_id = s.id
                LEFT JOIN chunk_metadata cm ON c.id = cm.chunk_id
                WHERE d.is_active = true
                AND cm.hq_embeddings IS NOT NULL
                AND jsonb_array_length(cm.hq_embeddings) > 0
        """

        # Add tenant filter (RBAC)
        if tenant_id:
            query_str += " AND d.tenant_id = :tenant_id"

        # Add workspace filter
        if workspace_id:
            query_str += " AND d.workspace_id = :workspace_id"

        # Add document filter
        if document_ids:
            query_str += " AND c.document_id = ANY(:doc_ids)"

        query_str += """
            )
            SELECT *
            FROM hq_similarities
            WHERE max_hq_score IS NOT NULL
            ORDER BY max_hq_score DESC
            LIMIT :limit
        """

        params = {"embedding": str(embedding_list), "limit": top_k}
        if tenant_id:
            params["tenant_id"] = tenant_id
        if workspace_id:
            params["workspace_id"] = workspace_id
        if document_ids:
            params["doc_ids"] = [str(d) for d in document_ids]

        try:
            query = text(query_str).bindparams(**params)
            result = await session.execute(query)
            rows = result.fetchall()
        except Exception as e:
            # HyDE search might fail if no HQ embeddings exist
            # Rollback to clear the failed transaction state
            await session.rollback()
            logger.debug("HyDE search failed or no results", error=str(e))
            return []

        return [
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                text=row.text,
                score=row.max_hq_score or 0.0,
                hyde_score=row.max_hq_score,
                rank=i + 1,
                section_path=row.section_path.split("/") if row.section_path else [],
                chunk_type=row.chunk_type,
                token_count=row.token_count,
                document_title=row.document_title,
                summary=row.summary,
                hypothetical_questions=row.hypothetical_questions or [],
            )
            for i, row in enumerate(rows)
        ]

    def _fuse_results(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        hyde_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion."""
        from src.retrieval.fusion import ReciprocalRankFusion

        fusion = ReciprocalRankFusion(k=60)  # Standard RRF constant

        # Prepare ranked lists
        ranked_lists = []
        weights = []

        if vector_results:
            ranked_lists.append([r.chunk_id for r in vector_results])
            weights.append(self.weights.get("vector", 0.4))

        if keyword_results:
            ranked_lists.append([r.chunk_id for r in keyword_results])
            weights.append(self.weights.get("keyword", 0.2))

        if hyde_results:
            ranked_lists.append([r.chunk_id for r in hyde_results])
            weights.append(self.weights.get("hyde", 0.4))

        if not ranked_lists:
            return []

        # Fuse rankings
        fused_ranking = fusion.fuse(ranked_lists, weights)

        # Build result map
        result_map: dict[UUID, SearchResult] = {}
        for r in vector_results + keyword_results + hyde_results:
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r
            else:
                # Merge scores
                existing = result_map[r.chunk_id]
                if r.vector_score is not None:
                    existing.vector_score = r.vector_score
                if r.keyword_score is not None:
                    existing.keyword_score = r.keyword_score
                if r.hyde_score is not None:
                    existing.hyde_score = r.hyde_score

        # Build final results
        final_results = []
        for rank, (chunk_id, fused_score) in enumerate(fused_ranking[:top_k], 1):
            if chunk_id in result_map:
                result = result_map[chunk_id]
                result.score = fused_score
                result.rank = rank
                final_results.append(result)

        return final_results
