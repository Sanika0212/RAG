"""Correction strategies for retrieval failures."""

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import anthropic
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.constants import FailureMode
from src.config.settings import get_settings
from src.agents.diagnosis import DiagnosisResult
from src.retrieval.search import SearchResult, HybridSearcher
from src.embeddings.bge_m3 import get_embedder

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class CorrectionResult:
    """Result of applying a correction strategy."""

    success: bool
    new_results: list[SearchResult]
    strategy_used: str
    details: str
    additional_queries: list[str] = field(default_factory=list)


class CorrectionExecutor:
    """Execute correction strategies based on failure diagnosis.

    Strategies:
    - AMBIGUITY → Decompose into sub-queries
    - VOCAB_MISMATCH → Reformulate with synonyms
    - INFO_SCATTER → Multi-hop gap-filling
    - KNOWLEDGE_GAP → Abstain (no correction possible)
    - GRANULARITY_MISMATCH → Walk hierarchy
    """

    def __init__(
        self,
        searcher: Optional[HybridSearcher] = None,
        model: str = settings.agent_model,
    ):
        """Initialize the correction executor.

        Args:
            searcher: HybridSearcher instance
            model: Claude model for LLM-based corrections
        """
        self.searcher = searcher or HybridSearcher()
        self.model = model
        self._client: Optional[anthropic.AsyncAnthropic] = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    async def execute(
        self,
        original_query: str,
        diagnosis: DiagnosisResult,
        original_results: list[SearchResult],
        session: AsyncSession,
    ) -> CorrectionResult:
        """Execute the appropriate correction strategy.

        Args:
            original_query: Original search query
            diagnosis: Failure diagnosis
            original_results: Original retrieval results
            session: Database session

        Returns:
            CorrectionResult with new results
        """
        logger.info(
            "Executing correction",
            failure_mode=diagnosis.failure_mode.value,
            query=original_query[:50],
        )

        if diagnosis.failure_mode == FailureMode.AMBIGUITY:
            return await self._correct_ambiguity(
                original_query, diagnosis, original_results, session
            )

        elif diagnosis.failure_mode == FailureMode.VOCAB_MISMATCH:
            return await self._correct_vocab_mismatch(
                original_query, diagnosis, session
            )

        elif diagnosis.failure_mode == FailureMode.INFO_SCATTER:
            return await self._correct_info_scatter(
                original_query, diagnosis, original_results, session
            )

        elif diagnosis.failure_mode == FailureMode.GRANULARITY_MISMATCH:
            return await self._correct_granularity(
                original_query, diagnosis, original_results, session
            )

        else:  # KNOWLEDGE_GAP
            return CorrectionResult(
                success=False,
                new_results=[],
                strategy_used="abstain",
                details="Knowledge gap detected - information not in knowledge base",
            )

    async def _correct_ambiguity(
        self,
        original_query: str,
        diagnosis: DiagnosisResult,
        original_results: list[SearchResult],
        session: AsyncSession,
    ) -> CorrectionResult:
        """Handle ambiguous queries by decomposition."""
        # Get sub-queries from diagnosis or generate new ones
        sub_queries = diagnosis.sub_queries
        if not sub_queries:
            sub_queries = await self._generate_sub_queries(original_query)

        if not sub_queries:
            return CorrectionResult(
                success=False,
                new_results=original_results,
                strategy_used="decomposition",
                details="Failed to generate sub-queries",
            )

        # Search for each sub-query
        all_results = []
        for sub_query in sub_queries:
            results = await self.searcher.search(
                sub_query,
                session,
                top_k=settings.retrieval_top_k // len(sub_queries),
            )
            all_results.extend(results)

        # Deduplicate by chunk_id
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                unique_results.append(r)

        # Re-rank combined results
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()
        reranked = await reranker.rerank(
            original_query,
            unique_results,
            top_k=settings.rerank_top_k,
        )

        return CorrectionResult(
            success=len(reranked) > 0,
            new_results=reranked,
            strategy_used="decomposition",
            details=f"Decomposed into {len(sub_queries)} sub-queries",
            additional_queries=sub_queries,
        )

    async def _correct_vocab_mismatch(
        self,
        original_query: str,
        diagnosis: DiagnosisResult,
        session: AsyncSession,
    ) -> CorrectionResult:
        """Handle vocabulary mismatch by reformulation."""
        # Get reformulated query from diagnosis or generate
        reformulated = diagnosis.reformulated_query
        if not reformulated:
            reformulated = await self._generate_reformulation(original_query)

        if not reformulated or reformulated == original_query:
            return CorrectionResult(
                success=False,
                new_results=[],
                strategy_used="reformulation",
                details="Failed to generate alternative query",
            )

        # Search with reformulated query
        results = await self.searcher.search(reformulated, session)

        # Rerank against original query (we want results relevant to original intent)
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()
        reranked = await reranker.rerank(
            original_query,
            results,
            top_k=settings.rerank_top_k,
        )

        return CorrectionResult(
            success=len(reranked) > 0,
            new_results=reranked,
            strategy_used="reformulation",
            details=f"Reformulated: '{reformulated}'",
            additional_queries=[reformulated],
        )

    async def _correct_info_scatter(
        self,
        original_query: str,
        diagnosis: DiagnosisResult,
        original_results: list[SearchResult],
        session: AsyncSession,
    ) -> CorrectionResult:
        """Handle scattered information with multi-hop retrieval."""
        if not original_results:
            return CorrectionResult(
                success=False,
                new_results=[],
                strategy_used="multi_hop",
                details="No initial results to build on",
            )

        # Extract bridge concepts from top results
        bridge_concepts = await self._extract_bridge_concepts(
            original_query,
            original_results[:3],
        )

        if not bridge_concepts:
            return CorrectionResult(
                success=False,
                new_results=original_results,
                strategy_used="multi_hop",
                details="No bridge concepts found",
            )

        # Search for bridge concepts
        bridge_results = []
        for concept in bridge_concepts[:2]:  # Max 2 bridge searches
            results = await self.searcher.search(
                f"{original_query} {concept}",
                session,
                top_k=settings.retrieval_top_k // 2,
            )
            bridge_results.extend(results)

        # Combine original and bridge results
        combined = original_results + bridge_results

        # Deduplicate
        seen_ids = set()
        unique_results = []
        for r in combined:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                unique_results.append(r)

        # Rerank
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()
        reranked = await reranker.rerank(
            original_query,
            unique_results,
            top_k=settings.rerank_top_k,
        )

        return CorrectionResult(
            success=len(reranked) > len(original_results),
            new_results=reranked,
            strategy_used="multi_hop",
            details=f"Added bridge concepts: {', '.join(bridge_concepts[:2])}",
            additional_queries=[f"{original_query} {c}" for c in bridge_concepts[:2]],
        )

    async def _correct_granularity(
        self,
        original_query: str,
        diagnosis: DiagnosisResult,
        original_results: list[SearchResult],
        session: AsyncSession,
    ) -> CorrectionResult:
        """Handle granularity mismatch by walking hierarchy."""
        direction = diagnosis.hierarchy_direction or "down"

        if not original_results:
            return CorrectionResult(
                success=False,
                new_results=[],
                strategy_used="hierarchy_walk",
                details="No results to walk hierarchy from",
            )

        # Get parent or child chunks
        if direction == "up":
            new_results = await self._get_parent_chunks(original_results, session)
        else:
            new_results = await self._get_child_chunks(original_results, session)

        if not new_results:
            return CorrectionResult(
                success=False,
                new_results=original_results,
                strategy_used="hierarchy_walk",
                details=f"No {direction}ward chunks found",
            )

        # Combine and rerank
        combined = list(original_results) + new_results

        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()
        reranked = await reranker.rerank(
            original_query,
            combined,
            top_k=settings.rerank_top_k,
        )

        return CorrectionResult(
            success=len(reranked) > 0,
            new_results=reranked,
            strategy_used="hierarchy_walk",
            details=f"Walked hierarchy {direction}: found {len(new_results)} related chunks",
        )

    async def _generate_sub_queries(self, query: str) -> list[str]:
        """Generate sub-queries using LLM."""
        prompt = f"""Break this complex question into 2-3 simpler, specific sub-questions that together would answer the original.

Original question: "{query}"

Respond with a JSON array of sub-questions:
["sub-question 1", "sub-question 2", "sub-question 3"]

Only respond with valid JSON array."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            import json
            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())

        except Exception as e:
            logger.warning("Failed to generate sub-queries", error=str(e))
            return []

    async def _generate_reformulation(self, query: str) -> str:
        """Generate query reformulation with synonyms."""
        prompt = f"""Reformulate this medical/scientific query using alternative terminology and synonyms.

Original: "{query}"

Provide a single reformulated query that preserves the meaning but uses different terms.
Response format: just the reformulated query, no explanation."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=128,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text.strip().strip('"')

        except Exception as e:
            logger.warning("Failed to generate reformulation", error=str(e))
            return ""

    async def _extract_bridge_concepts(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[str]:
        """Extract bridge concepts that could connect scattered information."""
        # Collect all keywords from results
        all_keywords = []
        for r in results:
            all_keywords.extend(r.keywords)

        if not all_keywords:
            return []

        # Find keywords that appear in multiple results (potential bridges)
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        bridges = [kw for kw, count in keyword_counts.most_common(5) if count >= 2]

        if not bridges:
            # Use LLM to identify concepts
            prompt = f"""Identify 2-3 key concepts that might bridge these text snippets together in relation to the query.

Query: "{query}"

Snippets:
{chr(10).join(r.text[:150] + '...' for r in results[:3])}

Respond with JSON array of bridge concepts:
["concept1", "concept2"]"""

            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=128,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )

                import json
                content = response.content[0].text.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                bridges = json.loads(content.strip())
            except Exception:
                pass

        return bridges

    async def _get_parent_chunks(
        self,
        results: list[SearchResult],
        session: AsyncSession,
    ) -> list[SearchResult]:
        """Get parent chunks from the hierarchy."""
        from sqlalchemy import select
        from src.database.models import Chunk, Section, Document

        parent_chunks = []

        for result in results[:3]:  # Check top 3 results
            # Get the chunk's section
            chunk_query = select(Chunk).where(Chunk.id == result.chunk_id)
            chunk_result = await session.execute(chunk_query)
            chunk = chunk_result.scalar_one_or_none()

            if not chunk or not chunk.section_id:
                continue

            # Get parent section
            section_query = select(Section).where(Section.id == chunk.section_id)
            section_result = await session.execute(section_query)
            section = section_result.scalar_one_or_none()

            if not section or not section.parent_section_id:
                continue

            # Get chunks from parent section
            parent_chunks_query = (
                select(Chunk)
                .where(Chunk.section_id == section.parent_section_id)
                .limit(3)
            )
            parent_result = await session.execute(parent_chunks_query)
            parent_chunk_rows = parent_result.scalars().all()

            for pc in parent_chunk_rows:
                parent_chunks.append(SearchResult(
                    chunk_id=pc.id,
                    document_id=pc.document_id,
                    text=pc.text,
                    score=0.5,  # Neutral score, will be reranked
                    rank=0,
                    chunk_type=pc.chunk_type.value if pc.chunk_type else "",
                    token_count=pc.token_count,
                ))

        return parent_chunks

    async def _get_child_chunks(
        self,
        results: list[SearchResult],
        session: AsyncSession,
    ) -> list[SearchResult]:
        """Get child chunks from the hierarchy."""
        from sqlalchemy import select
        from src.database.models import Chunk, Section

        child_chunks = []

        for result in results[:3]:
            # Get the chunk's section
            chunk_query = select(Chunk).where(Chunk.id == result.chunk_id)
            chunk_result = await session.execute(chunk_query)
            chunk = chunk_result.scalar_one_or_none()

            if not chunk or not chunk.section_id:
                continue

            # Get child sections
            child_sections_query = (
                select(Section)
                .where(Section.parent_section_id == chunk.section_id)
            )
            child_result = await session.execute(child_sections_query)
            child_sections = child_result.scalars().all()

            for child_section in child_sections[:2]:  # Max 2 child sections
                child_chunks_query = (
                    select(Chunk)
                    .where(Chunk.section_id == child_section.id)
                    .limit(2)
                )
                child_chunk_result = await session.execute(child_chunks_query)
                child_chunk_rows = child_chunk_result.scalars().all()

                for cc in child_chunk_rows:
                    child_chunks.append(SearchResult(
                        chunk_id=cc.id,
                        document_id=cc.document_id,
                        text=cc.text,
                        score=0.5,
                        rank=0,
                        chunk_type=cc.chunk_type.value if cc.chunk_type else "",
                        token_count=cc.token_count,
                    ))

        return child_chunks
