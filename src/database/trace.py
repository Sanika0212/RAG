"""Trace service for storing and retrieving query reasoning traces."""

import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import QueryLog, QueryTrace

logger = structlog.get_logger(__name__)


@dataclass
class TraceBuilder:
    """Builder for constructing query traces incrementally."""

    query_text: str
    session_id: Optional[str] = None

    # Timing
    _start_time: float = field(default_factory=time.time)
    _stage_times: dict = field(default_factory=dict)

    # Retrieval
    retrieval_results: list = field(default_factory=list)

    # Confidence
    confidence_components: dict = field(default_factory=dict)
    coverage_report: dict = field(default_factory=dict)

    # Correction
    correction_triggered: bool = False
    diagnosis: Optional[dict] = None
    correction_actions: list = field(default_factory=list)
    pre_correction_confidence: Optional[float] = None
    post_correction_confidence: Optional[float] = None

    # Generation
    generation_prompt_type: Optional[str] = None
    citations_generated: list = field(default_factory=list)
    response_text: Optional[str] = None

    # Validation
    claims_validation: list = field(default_factory=list)
    hallucination_risk: Optional[float] = None
    revision_notes: list = field(default_factory=list)

    # LLM tracking
    llm_calls: list = field(default_factory=list)

    def start_stage(self, stage: str) -> None:
        """Mark the start of a processing stage."""
        self._stage_times[f"{stage}_start"] = time.time()

    def end_stage(self, stage: str) -> None:
        """Mark the end of a processing stage."""
        self._stage_times[f"{stage}_end"] = time.time()

    def add_retrieval_result(
        self,
        chunk_id: str,
        text_preview: str,
        score: float,
        keywords: list[str],
        document_title: str,
    ) -> None:
        """Add a retrieval result to the trace."""
        self.retrieval_results.append({
            "chunk_id": chunk_id,
            "text_preview": text_preview[:200] + "..." if len(text_preview) > 200 else text_preview,
            "score": round(score, 4),
            "keywords": keywords[:5],
            "document_title": document_title,
        })

    def set_confidence(
        self,
        score: float,
        components: dict,
        coverage: dict,
    ) -> None:
        """Set confidence estimation results."""
        self.confidence_components = {
            "overall": round(score, 4),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in components.items()}
        }
        self.coverage_report = coverage

    def add_correction_attempt(
        self,
        attempt: int,
        strategy: str,
        details: str,
        success: bool,
        new_confidence: float,
    ) -> None:
        """Add a correction attempt to the trace."""
        self.correction_actions.append({
            "attempt": attempt,
            "strategy": strategy,
            "details": details,
            "success": success,
            "new_confidence": round(new_confidence, 4),
        })

    def add_citation(
        self,
        index: int,
        chunk_id: str,
        document_title: str,
        relevance_score: float,
    ) -> None:
        """Add a citation to the trace."""
        self.citations_generated.append({
            "index": index,
            "chunk_id": chunk_id,
            "document_title": document_title,
            "relevance_score": round(relevance_score, 4),
        })

    def add_claim_validation(
        self,
        claim: str,
        status: str,
        confidence: float,
        supporting_chunks: list[str],
    ) -> None:
        """Add claim validation result to the trace."""
        self.claims_validation.append({
            "claim": claim,
            "status": status,
            "confidence": round(confidence, 4),
            "supporting_chunks": supporting_chunks,
        })

    def add_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str,
    ) -> None:
        """Track an LLM call for cost tracking."""
        self.llm_calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "purpose": purpose,
        })

    def _compute_timing_breakdown(self) -> dict:
        """Compute timing breakdown from stage times."""
        timing = {}
        stages = ["embedding", "retrieval", "reranking", "confidence", "correction", "generation", "validation"]

        for stage in stages:
            start = self._stage_times.get(f"{stage}_start")
            end = self._stage_times.get(f"{stage}_end")
            if start and end:
                timing[stage] = int((end - start) * 1000)  # ms

        timing["total"] = int((time.time() - self._start_time) * 1000)
        return timing

    def _compute_costs(self) -> tuple[int, int, float]:
        """Compute total tokens and estimated costs."""
        total_input = sum(call.get("input_tokens", 0) for call in self.llm_calls)
        total_output = sum(call.get("output_tokens", 0) for call in self.llm_calls)

        # Approximate costs (USD per 1M tokens)
        # Haiku: $0.25 input, $1.25 output
        # Sonnet: $3 input, $15 output
        cost = 0.0
        for call in self.llm_calls:
            model = call.get("model", "").lower()
            inp = call.get("input_tokens", 0)
            out = call.get("output_tokens", 0)

            if "haiku" in model:
                cost += (inp * 0.25 + out * 1.25) / 1_000_000
            elif "sonnet" in model:
                cost += (inp * 3 + out * 15) / 1_000_000
            elif "opus" in model:
                cost += (inp * 15 + out * 75) / 1_000_000

        return total_input, total_output, round(cost, 6)

    async def save(self, session: AsyncSession) -> tuple[UUID, UUID]:
        """Save the trace to the database.

        Returns:
            Tuple of (query_log_id, trace_id)
        """
        timing = self._compute_timing_breakdown()
        total_input, total_output, cost = self._compute_costs()

        # Create QueryLog
        query_log = QueryLog(
            query_text=self.query_text,
            retrieved_chunk_ids=[r["chunk_id"] for r in self.retrieval_results],
            retrieval_scores=[r["score"] for r in self.retrieval_results],
            confidence_score=self.confidence_components.get("overall"),
            confidence_band=self._get_confidence_band(),
            failure_mode=self.diagnosis.get("failure_mode") if self.diagnosis else None,
            correction_attempts=len(self.correction_actions),
            final_confidence=self.post_correction_confidence or self.confidence_components.get("overall"),
            response_text=self.response_text,
            claims_extracted=len(self.claims_validation) if self.claims_validation else None,
            claims_grounded=sum(1 for c in self.claims_validation if c["status"] == "GROUNDED"),
            claims_ungrounded=sum(1 for c in self.claims_validation if c["status"] == "UNGROUNDED"),
            retrieval_latency_ms=timing.get("retrieval"),
            generation_latency_ms=timing.get("generation"),
            total_latency_ms=timing.get("total"),
            session_id=self.session_id,
        )
        session.add(query_log)
        await session.flush()

        # Create QueryTrace
        trace = QueryTrace(
            query_log_id=query_log.id,
            retrieval_results=self.retrieval_results,
            confidence_components=self.confidence_components,
            coverage_report=self.coverage_report,
            correction_triggered=self.correction_triggered,
            diagnosis=self.diagnosis,
            correction_actions=self.correction_actions if self.correction_actions else None,
            pre_correction_confidence=self.pre_correction_confidence,
            post_correction_confidence=self.post_correction_confidence,
            generation_prompt_type=self.generation_prompt_type,
            citations_generated=self.citations_generated if self.citations_generated else None,
            claims_validation=self.claims_validation if self.claims_validation else None,
            hallucination_risk=self.hallucination_risk,
            revision_notes=self.revision_notes if self.revision_notes else None,
            timing_breakdown=timing,
            llm_calls=self.llm_calls if self.llm_calls else None,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            estimated_cost_usd=cost,
        )
        session.add(trace)
        await session.commit()

        logger.info(
            "Trace saved",
            query_log_id=str(query_log.id),
            trace_id=str(trace.id),
            total_latency_ms=timing.get("total"),
            cost_usd=cost,
        )

        return query_log.id, trace.id

    def _get_confidence_band(self) -> Optional[str]:
        """Get confidence band from score."""
        score = self.confidence_components.get("overall")
        if score is None:
            return None
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.45:
            return "MEDIUM"
        return "LOW"


async def get_trace(
    session: AsyncSession,
    query_log_id: UUID,
) -> Optional[dict]:
    """Get a trace by query log ID.

    Returns:
        Dictionary with full trace data or None if not found
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    result = await session.execute(
        select(QueryLog)
        .options(selectinload(QueryLog.trace))
        .where(QueryLog.id == query_log_id)
    )
    query_log = result.scalar_one_or_none()

    if not query_log or not query_log.trace:
        return None

    trace = query_log.trace
    return {
        "query_log_id": str(query_log.id),
        "query_text": query_log.query_text,
        "created_at": query_log.created_at.isoformat(),
        "retrieval": {
            "results": trace.retrieval_results,
            "count": len(trace.retrieval_results) if trace.retrieval_results else 0,
        },
        "confidence": {
            "components": trace.confidence_components,
            "coverage": trace.coverage_report,
            "band": query_log.confidence_band,
        },
        "correction": {
            "triggered": trace.correction_triggered,
            "diagnosis": trace.diagnosis,
            "actions": trace.correction_actions,
            "pre_confidence": trace.pre_correction_confidence,
            "post_confidence": trace.post_correction_confidence,
        },
        "generation": {
            "prompt_type": trace.generation_prompt_type,
            "citations": trace.citations_generated,
            "response": query_log.response_text,
        },
        "validation": {
            "claims": trace.claims_validation,
            "hallucination_risk": trace.hallucination_risk,
            "revision_notes": trace.revision_notes,
        },
        "performance": {
            "timing": trace.timing_breakdown,
            "llm_calls": trace.llm_calls,
            "total_input_tokens": trace.total_input_tokens,
            "total_output_tokens": trace.total_output_tokens,
            "estimated_cost_usd": trace.estimated_cost_usd,
        },
    }
