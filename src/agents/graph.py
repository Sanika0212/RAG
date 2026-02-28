"""LangGraph state machine for RAG agent workflow."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Optional, Literal
from uuid import UUID

import structlog
from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.constants import ConfidenceBand, FailureMode, AgentState
from src.config.settings import get_settings
from src.retrieval.search import SearchResult, HybridSearcher
from src.retrieval.reranker import get_reranker
from src.retrieval.confidence import ConfidenceEstimator, ConfidenceResult
from src.agents.diagnosis import FailureDiagnoser, DiagnosisResult
from src.agents.correction import CorrectionExecutor, CorrectionResult

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class RAGState:
    """State for the RAG agent workflow."""

    # Input
    query: str
    session: AsyncSession  # Database session
    document_ids: list[UUID] = field(default_factory=list)  # Optional filter
    tenant_id: Optional[str] = None  # Multi-tenant RBAC filtering

    # Retrieval state
    results: list[SearchResult] = field(default_factory=list)
    confidence: Optional[ConfidenceResult] = None

    # Correction state
    diagnosis: Optional[DiagnosisResult] = None
    correction_attempts: int = 0
    correction_history: list[CorrectionResult] = field(default_factory=list)

    # Generation state
    response: str = ""
    citations: list[dict] = field(default_factory=list)

    # Validation state
    claims: list[dict] = field(default_factory=list)
    validation_result: Optional[dict] = None

    # Tracing
    trace: list[dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    current_state: AgentState = AgentState.RETRIEVE


class RAGAgentGraph:
    """LangGraph-based RAG agent with self-correction loops.

    Flow:
    retrieve → estimate_confidence → [router]
    router → generate (HIGH) | generate_hedged (MEDIUM) | diagnose (LOW)
    diagnose → correct → estimate_confidence (loop, max 2)
    generate/generate_hedged → validate → final
    """

    def __init__(
        self,
        searcher: Optional[HybridSearcher] = None,
        confidence_estimator: Optional[ConfidenceEstimator] = None,
        diagnoser: Optional[FailureDiagnoser] = None,
        corrector: Optional[CorrectionExecutor] = None,
        max_corrections: int = settings.max_correction_loops,
    ):
        """Initialize the RAG agent graph.

        Args:
            searcher: HybridSearcher instance
            confidence_estimator: ConfidenceEstimator instance
            diagnoser: FailureDiagnoser instance
            corrector: CorrectionExecutor instance
            max_corrections: Maximum correction loop iterations
        """
        self.searcher = searcher or HybridSearcher()
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator()
        self.diagnoser = diagnoser or FailureDiagnoser()
        self.corrector = corrector or CorrectionExecutor(searcher=self.searcher)
        self.max_corrections = max_corrections

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create the graph with state schema
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("estimate_confidence", self._estimate_confidence_node)
        builder.add_node("diagnose", self._diagnose_node)
        builder.add_node("correct", self._correct_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("generate_hedged", self._generate_hedged_node)
        builder.add_node("validate", self._validate_node)
        builder.add_node("abstain", self._abstain_node)

        # Set entry point
        builder.set_entry_point("retrieve")

        # Add edges
        builder.add_edge("retrieve", "estimate_confidence")
        builder.add_conditional_edges(
            "estimate_confidence",
            self._route_by_confidence,
            {
                "generate": "generate",
                "generate_hedged": "generate_hedged",
                "diagnose": "diagnose",
                "abstain": "abstain",
            },
        )
        builder.add_edge("diagnose", "correct")
        builder.add_conditional_edges(
            "correct",
            self._route_after_correction,
            {
                "estimate_confidence": "estimate_confidence",
                "abstain": "abstain",
            },
        )
        builder.add_edge("generate", "validate")
        builder.add_edge("generate_hedged", "validate")
        builder.add_edge("validate", END)
        builder.add_edge("abstain", END)

        return builder.compile()

    async def _retrieve_node(self, state: RAGState) -> dict:
        """Retrieve relevant documents."""
        state.current_state = AgentState.RETRIEVE
        self._add_trace(state, "Starting retrieval", {"query": state.query, "tenant_id": state.tenant_id})

        try:
            results = await self.searcher.search(
                state.query,
                state.session,
                document_ids=state.document_ids if state.document_ids else None,
                tenant_id=state.tenant_id,
            )

            # Rerank results
            reranker = get_reranker()
            reranked = await reranker.rerank(
                state.query,
                results,
                top_k=settings.rerank_top_k,
            )

            self._add_trace(state, "Retrieval complete", {
                "initial_results": len(results),
                "reranked_results": len(reranked),
            })

            return {"results": reranked}

        except Exception as e:
            logger.error("Retrieval failed", error=str(e))
            self._add_trace(state, "Retrieval failed", {"error": str(e)})
            return {"results": []}

    async def _estimate_confidence_node(self, state: RAGState) -> dict:
        """Estimate confidence in retrieval results."""
        state.current_state = AgentState.ESTIMATE_CONFIDENCE

        confidence = await self.confidence_estimator.estimate(
            state.query,
            state.results,
        )

        self._add_trace(state, "Confidence estimated", {
            "score": confidence.score,
            "band": confidence.band.value,
            "components": confidence.components,
        })

        return {"confidence": confidence}

    def _route_by_confidence(self, state: RAGState) -> str:
        """Route based on confidence level."""
        state.current_state = AgentState.ROUTE

        if not state.confidence:
            return "abstain"

        band = state.confidence.band

        # Check if we've exhausted corrections
        if state.correction_attempts >= self.max_corrections:
            if band == ConfidenceBand.LOW:
                return "abstain"
            else:
                return "generate_hedged"

        if band == ConfidenceBand.HIGH:
            self._add_trace(state, "Routing to generate", {"reason": "high confidence"})
            return "generate"
        elif band == ConfidenceBand.MEDIUM:
            self._add_trace(state, "Routing to generate_hedged", {"reason": "medium confidence"})
            return "generate_hedged"
        else:
            self._add_trace(state, "Routing to diagnose", {"reason": "low confidence"})
            return "diagnose"

    async def _diagnose_node(self, state: RAGState) -> dict:
        """Diagnose retrieval failure."""
        state.current_state = AgentState.DIAGNOSE

        diagnosis = await self.diagnoser.diagnose(
            state.query,
            state.results,
            state.confidence,
        )

        self._add_trace(state, "Failure diagnosed", {
            "failure_mode": diagnosis.failure_mode.value,
            "confidence": diagnosis.confidence,
            "reasoning": diagnosis.reasoning,
        })

        return {"diagnosis": diagnosis}

    async def _correct_node(self, state: RAGState) -> dict:
        """Execute correction strategy."""
        state.current_state = AgentState.CORRECT

        correction = await self.corrector.execute(
            state.query,
            state.diagnosis,
            state.results,
            state.session,
        )

        self._add_trace(state, "Correction executed", {
            "strategy": correction.strategy_used,
            "success": correction.success,
            "new_results": len(correction.new_results),
        })

        # Update state - MUST include correction_attempts to prevent infinite loops
        new_state = {
            "results": correction.new_results if correction.success else state.results,
            "correction_history": state.correction_history + [correction],
            "correction_attempts": state.correction_attempts + 1,  # Increment counter
        }

        return new_state

    def _route_after_correction(self, state: RAGState) -> str:
        """Route after correction attempt."""
        # Check if correction was for KNOWLEDGE_GAP
        if state.diagnosis and state.diagnosis.failure_mode == FailureMode.KNOWLEDGE_GAP:
            return "abstain"

        # Check if we've exhausted corrections
        if state.correction_attempts >= self.max_corrections:
            return "abstain"

        # Re-estimate confidence
        return "estimate_confidence"

    async def _generate_node(self, state: RAGState) -> dict:
        """Generate confident response."""
        state.current_state = AgentState.GENERATE

        from src.generation.generator import ResponseGenerator
        generator = ResponseGenerator()

        response, citations = await generator.generate(
            query=state.query,
            context_chunks=state.results,
            confidence_band=ConfidenceBand.HIGH,
        )

        self._add_trace(state, "Response generated", {
            "mode": "confident",
            "response_length": len(response),
            "citations": len(citations),
        })

        return {"response": response, "citations": citations}

    async def _generate_hedged_node(self, state: RAGState) -> dict:
        """Generate hedged response for medium confidence."""
        state.current_state = AgentState.GENERATE_HEDGED

        from src.generation.generator import ResponseGenerator
        generator = ResponseGenerator()

        response, citations = await generator.generate(
            query=state.query,
            context_chunks=state.results,
            confidence_band=ConfidenceBand.MEDIUM,
        )

        self._add_trace(state, "Response generated", {
            "mode": "hedged",
            "response_length": len(response),
            "citations": len(citations),
        })

        return {"response": response, "citations": citations}

    async def _validate_node(self, state: RAGState) -> dict:
        """Validate response claims."""
        state.current_state = AgentState.VALIDATE

        try:
            from src.validation.claims import ClaimValidator
            validator = ClaimValidator()

            validation_result = await validator.validate_response(
                query=state.query,
                response=state.response,
                context_chunks=state.results,
            )

            self._add_trace(state, "Response validated", {
                "total_claims": validation_result.get("total_claims", 0),
                "grounded_claims": validation_result.get("grounded_claims", 0),
                "hallucination_risk": validation_result.get("hallucination_risk", 0),
            })

            # Update response if claims were removed
            updated_response = validation_result.get("revised_response", state.response)

            return {
                "response": updated_response,
                "claims": validation_result.get("claims", []),
                "validation_result": validation_result,
            }
        except Exception as e:
            # If validation fails (e.g., Gemini API issues), skip validation
            logger.warning("Validation skipped due to error", error=str(e))
            self._add_trace(state, "Validation skipped", {"error": str(e)})
            return {
                "response": state.response,
                "claims": [],
                "validation_result": {"skipped": True, "error": str(e)},
            }

    async def _abstain_node(self, state: RAGState) -> dict:
        """Handle abstention when we cannot provide a reliable answer."""
        state.current_state = AgentState.ABSTAIN

        from src.config.constants import ABSTENTION_PHRASES
        import random

        # Generate abstention response
        abstention_phrase = random.choice(ABSTENTION_PHRASES)

        # Add specific information if available
        if state.diagnosis:
            if state.diagnosis.failure_mode == FailureMode.KNOWLEDGE_GAP:
                abstention_phrase += " The specific information requested does not appear to be in the available documents."
            elif state.diagnosis.failure_mode == FailureMode.AMBIGUITY:
                abstention_phrase += " The query may be too broad or ambiguous. Consider asking a more specific question."

        if state.confidence and state.confidence.gaps:
            abstention_phrase += f" Identified gaps: {', '.join(state.confidence.gaps[:2])}."

        self._add_trace(state, "Abstaining from response", {
            "reason": state.diagnosis.failure_mode.value if state.diagnosis else "unknown",
            "confidence": state.confidence.score if state.confidence else 0,
        })

        return {
            "response": abstention_phrase,
            "citations": [],
        }

    def _add_trace(self, state: RAGState, message: str, data: dict) -> None:
        """Add trace entry to state."""
        elapsed = (datetime.now() - state.start_time).total_seconds()
        state.trace.append({
            "timestamp": elapsed,
            "state": state.current_state.value,
            "message": message,
            "data": data,
        })

    async def run(
        self,
        query: str,
        session: AsyncSession,
        document_ids: Optional[list[UUID]] = None,
        tenant_id: Optional[str] = None,
    ) -> RAGState:
        """Run the RAG agent workflow.

        Args:
            query: User query
            session: Database session
            document_ids: Optional filter to specific documents
            tenant_id: Tenant ID for multi-tenant RBAC filtering

        Returns:
            Final RAGState with response and trace
        """
        initial_state = RAGState(
            query=query,
            session=session,
            document_ids=document_ids or [],
            tenant_id=tenant_id,
        )

        logger.info("Starting RAG agent", query=query[:50])

        # Run the graph with recursion limit - LangGraph returns a dict
        result = await self.graph.ainvoke(
            initial_state,
            config={"recursion_limit": 10}  # Prevent infinite loops
        )

        # LangGraph may return a dict with state keys or the state object
        if isinstance(result, dict):
            # Convert dict back to RAGState
            final_state = RAGState(
                query=result.get("query", query),
                session=session,
                document_ids=result.get("document_ids", document_ids or []),
                tenant_id=result.get("tenant_id", tenant_id),
                results=result.get("results", []),
                confidence=result.get("confidence"),
                diagnosis=result.get("diagnosis"),
                correction_attempts=result.get("correction_attempts", 0),
                correction_history=result.get("correction_history", []),
                response=result.get("response", ""),
                citations=result.get("citations", []),
                claims=result.get("claims", []),
                validation_result=result.get("validation_result"),
                trace=result.get("trace", []),
            )
        else:
            final_state = result

        logger.info(
            "RAG agent complete",
            response_length=len(final_state.response) if final_state.response else 0,
            corrections=final_state.correction_attempts,
            elapsed=final_state.trace[-1]["timestamp"] if final_state.trace else 0,
        )

        return final_state
