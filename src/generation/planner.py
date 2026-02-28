"""Query planning and decomposition for complex queries."""

from dataclasses import dataclass, field
from typing import Optional

import anthropic
import structlog

from src.config.constants import QueryType
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class SubQuery:
    """A sub-query within a query plan."""

    query: str
    query_type: QueryType
    depends_on: list[int] = field(default_factory=list)  # Indices of dependent sub-queries
    priority: int = 1  # Execution priority (lower = first)
    confidence: Optional[float] = None  # Confidence after retrieval


@dataclass
class QueryPlan:
    """A plan for answering a complex query."""

    original_query: str
    query_type: QueryType
    sub_queries: list[SubQuery]
    reasoning: str
    requires_synthesis: bool = True


class QueryPlanner:
    """Plan query execution for complex queries.

    Handles:
    - SIMPLE: Single fact lookup, no planning needed
    - COMPOUND: Multiple related facts, parallel sub-queries
    - COMPARATIVE: Compare entities, structured sub-queries
    - TEMPORAL: Time-based, ordered sub-queries
    - CAUSAL: Cause-effect, dependency chain
    - PROCEDURAL: How-to, sequential steps
    """

    def __init__(
        self,
        model: str = settings.agent_model,
    ):
        """Initialize the query planner.

        Args:
            model: Claude model for planning
        """
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

    async def create_plan(self, query: str) -> QueryPlan:
        """Create a query plan.

        Args:
            query: User query

        Returns:
            QueryPlan with classified type and sub-queries
        """
        logger.info("Creating query plan", query=query[:50])

        # First, classify the query type
        query_type = await self._classify_query(query)

        # For simple queries, no decomposition needed
        if query_type == QueryType.SIMPLE:
            return QueryPlan(
                original_query=query,
                query_type=query_type,
                sub_queries=[SubQuery(query=query, query_type=QueryType.SIMPLE)],
                reasoning="Simple factual query, no decomposition needed.",
                requires_synthesis=False,
            )

        # For complex queries, decompose
        plan = await self._decompose_query(query, query_type)
        return plan

    async def _classify_query(self, query: str) -> QueryType:
        """Classify the query type."""
        prompt = f"""Classify this medical/scientific query into one of these types:

Query: "{query}"

Types:
- SIMPLE: Single fact lookup (e.g., "What is the half-life of aspirin?")
- COMPOUND: Multiple related facts needed (e.g., "What are the symptoms and treatment of diabetes?")
- COMPARATIVE: Comparing entities (e.g., "Compare metformin and insulin")
- TEMPORAL: Time-based question (e.g., "How has treatment for X evolved?")
- CAUSAL: Cause-effect relationship (e.g., "Why does X cause Y?")
- PROCEDURAL: How-to or process (e.g., "How do you diagnose X?")

Respond with just the type name (e.g., "SIMPLE")."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            type_str = response.content[0].text.strip().upper()

            # Map to enum
            type_map = {
                "SIMPLE": QueryType.SIMPLE,
                "COMPOUND": QueryType.COMPOUND,
                "COMPARATIVE": QueryType.COMPARATIVE,
                "TEMPORAL": QueryType.TEMPORAL,
                "CAUSAL": QueryType.CAUSAL,
                "PROCEDURAL": QueryType.PROCEDURAL,
            }

            return type_map.get(type_str, QueryType.SIMPLE)

        except Exception as e:
            logger.warning("Query classification failed", error=str(e))
            return QueryType.SIMPLE

    async def _decompose_query(
        self,
        query: str,
        query_type: QueryType,
    ) -> QueryPlan:
        """Decompose a complex query into sub-queries."""
        prompt = f"""Decompose this {query_type.value} query into sub-queries.

Query: "{query}"

Respond with JSON:
{{
    "reasoning": "Brief explanation of decomposition",
    "sub_queries": [
        {{
            "query": "sub-query text",
            "type": "SIMPLE|COMPOUND|COMPARATIVE|TEMPORAL|CAUSAL|PROCEDURAL",
            "depends_on": [0, 1],  // Indices of sub-queries this depends on (empty if independent)
            "priority": 1  // Execution order (1 = first)
        }}
    ]
}}

Guidelines:
- Create 2-4 focused sub-queries
- Mark dependencies correctly
- Prioritize foundational queries first
- Each sub-query should be answerable independently (except for dependencies)

Respond with valid JSON only."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            import json
            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            data = json.loads(content)

            sub_queries = []
            for sq in data.get("sub_queries", []):
                sub_type_str = sq.get("type", "SIMPLE").upper()
                sub_type = QueryType[sub_type_str] if sub_type_str in QueryType.__members__ else QueryType.SIMPLE

                sub_queries.append(SubQuery(
                    query=sq["query"],
                    query_type=sub_type,
                    depends_on=sq.get("depends_on", []),
                    priority=sq.get("priority", 1),
                ))

            return QueryPlan(
                original_query=query,
                query_type=query_type,
                sub_queries=sub_queries,
                reasoning=data.get("reasoning", ""),
                requires_synthesis=len(sub_queries) > 1,
            )

        except Exception as e:
            logger.warning("Query decomposition failed", error=str(e))
            # Fallback to single query
            return QueryPlan(
                original_query=query,
                query_type=query_type,
                sub_queries=[SubQuery(query=query, query_type=query_type)],
                reasoning=f"Decomposition failed: {str(e)}",
                requires_synthesis=False,
            )

    def get_execution_order(self, plan: QueryPlan) -> list[list[int]]:
        """Get the order of sub-query execution (for parallel execution).

        Returns:
            List of batches, where each batch contains indices of sub-queries
            that can be executed in parallel.
        """
        if not plan.sub_queries:
            return []

        # Group by priority
        priority_groups: dict[int, list[int]] = {}
        for i, sq in enumerate(plan.sub_queries):
            priority = sq.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(i)

        # Sort by priority
        batches = [
            indices
            for _, indices in sorted(priority_groups.items())
        ]

        return batches

    async def update_plan_with_confidence(
        self,
        plan: QueryPlan,
        sub_query_confidences: list[float],
    ) -> QueryPlan:
        """Update plan with confidence scores from retrieval.

        Args:
            plan: Original query plan
            sub_query_confidences: Confidence scores for each sub-query

        Returns:
            Updated plan with confidence information
        """
        for i, confidence in enumerate(sub_query_confidences):
            if i < len(plan.sub_queries):
                plan.sub_queries[i].confidence = confidence

        return plan

    def topological_sort(self, plan: QueryPlan) -> list[list[int]]:
        """Sort sub-queries by dependencies using Kahn's algorithm.

        Returns batches where each batch can be executed in parallel,
        and each batch depends on all previous batches completing.
        """
        n = len(plan.sub_queries)
        if n == 0:
            return []

        # Build dependency graph
        in_degree = [0] * n
        dependents: dict[int, list[int]] = {i: [] for i in range(n)}

        for i, sq in enumerate(plan.sub_queries):
            for dep in sq.depends_on:
                if 0 <= dep < n:
                    in_degree[i] += 1
                    dependents[dep].append(i)

        # Kahn's algorithm with batching
        batches = []
        ready = [i for i in range(n) if in_degree[i] == 0]

        while ready:
            # Current batch: all nodes with no remaining dependencies
            batch = sorted(ready, key=lambda x: plan.sub_queries[x].priority)
            batches.append(batch)

            next_ready = []
            for node in batch:
                for dependent in dependents[node]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_ready.append(dependent)

            ready = next_ready

        return batches


@dataclass
class SubQueryResult:
    """Result from executing a sub-query."""

    sub_query: SubQuery
    index: int
    answer: Optional[str] = None
    chunks: list = field(default_factory=list)
    confidence: float = 0.0
    status: str = "pending"  # pending, answered, uncertain, failed


@dataclass
class PlanExecutionResult:
    """Result from executing a full query plan."""

    plan: QueryPlan
    sub_results: list[SubQueryResult]
    synthesized_answer: str
    overall_confidence: float
    gaps: list[str] = field(default_factory=list)


class QueryPlanExecutor:
    """Execute query plans with per-sub-query confidence tracking."""

    def __init__(
        self,
        planner: Optional[QueryPlanner] = None,
        model: str = settings.agent_model,
    ):
        """Initialize the executor.

        Args:
            planner: QueryPlanner instance
            model: Claude model for synthesis
        """
        self.planner = planner or QueryPlanner()
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

    async def execute_plan(
        self,
        plan: QueryPlan,
        session,
    ) -> PlanExecutionResult:
        """Execute a query plan with dependency-aware ordering.

        Args:
            plan: The query plan to execute
            session: Database session

        Returns:
            PlanExecutionResult with all sub-results and synthesized answer
        """
        from src.retrieval.search import HybridSearcher
        from src.retrieval.confidence import ConfidenceEstimator
        from src.generation.generator import ResponseGenerator

        logger.info(
            "Executing query plan",
            query_type=plan.query_type.value,
            sub_queries=len(plan.sub_queries),
        )

        searcher = HybridSearcher()
        confidence_estimator = ConfidenceEstimator()
        generator = ResponseGenerator()

        # Get execution order
        batches = self.planner.topological_sort(plan)

        # Execute each batch
        sub_results: list[SubQueryResult] = [
            SubQueryResult(sub_query=sq, index=i)
            for i, sq in enumerate(plan.sub_queries)
        ]

        for batch in batches:
            # Execute sub-queries in this batch (could be parallelized)
            for idx in batch:
                sq = plan.sub_queries[idx]

                # Build context from dependencies
                dep_context = self._build_dependency_context(sub_results, sq.depends_on)

                # Search
                results = await searcher.search(
                    query=sq.query,
                    session=session,
                    top_k=settings.retrieval_top_k,
                )

                # Estimate confidence
                confidence_result = await confidence_estimator.estimate(
                    query=sq.query,
                    chunks=results,
                )

                sub_results[idx].chunks = results
                sub_results[idx].confidence = confidence_result.score

                # Generate answer if confidence is acceptable
                if confidence_result.score >= settings.confidence_low_threshold:
                    from src.config.constants import ConfidenceBand
                    band = (
                        ConfidenceBand.HIGH
                        if confidence_result.score >= settings.confidence_high_threshold
                        else ConfidenceBand.MEDIUM
                    )

                    answer, _ = await generator.generate(
                        query=sq.query,
                        context_chunks=results[:5],
                        confidence_band=band,
                        additional_context=dep_context,
                    )
                    sub_results[idx].answer = answer
                    sub_results[idx].status = "answered"
                else:
                    sub_results[idx].status = "uncertain"

        # Synthesize final answer
        synthesized, gaps = await self._synthesize_results(plan, sub_results)

        # Calculate overall confidence
        answered_results = [r for r in sub_results if r.status == "answered"]
        overall_confidence = (
            sum(r.confidence for r in answered_results) / len(answered_results)
            if answered_results
            else 0.0
        )

        return PlanExecutionResult(
            plan=plan,
            sub_results=sub_results,
            synthesized_answer=synthesized,
            overall_confidence=overall_confidence,
            gaps=gaps,
        )

    def _build_dependency_context(
        self,
        sub_results: list[SubQueryResult],
        depends_on: list[int],
    ) -> Optional[str]:
        """Build context from completed dependent sub-queries."""
        if not depends_on:
            return None

        context_parts = []
        for dep_idx in depends_on:
            if dep_idx < len(sub_results):
                result = sub_results[dep_idx]
                if result.answer:
                    context_parts.append(
                        f"Previous finding ({result.sub_query.query}): {result.answer}"
                    )

        return "\n\n".join(context_parts) if context_parts else None

    async def _synthesize_results(
        self,
        plan: QueryPlan,
        sub_results: list[SubQueryResult],
    ) -> tuple[str, list[str]]:
        """Synthesize sub-query results into a final answer.

        Returns:
            Tuple of (synthesized_answer, gaps_list)
        """
        if not plan.requires_synthesis or len(sub_results) == 1:
            # Single query, no synthesis needed
            if sub_results and sub_results[0].answer:
                return sub_results[0].answer, []
            return "Unable to find relevant information.", ["No results found"]

        # Build synthesis prompt
        answered = [r for r in sub_results if r.status == "answered"]
        uncertain = [r for r in sub_results if r.status == "uncertain"]

        if not answered:
            gaps = [r.sub_query.query for r in uncertain]
            return "Unable to find reliable information for this query.", gaps

        answered_parts = []
        for r in answered:
            conf_label = "high" if r.confidence >= 0.75 else "moderate"
            answered_parts.append(
                f"Sub-question: {r.sub_query.query}\n"
                f"Confidence: {conf_label}\n"
                f"Answer: {r.answer}"
            )

        uncertain_parts = [r.sub_query.query for r in uncertain]

        prompt = f"""Synthesize these sub-answers into a coherent response to the original question.

Original Question: "{plan.original_query}"

Sub-Answers:
{chr(10).join(answered_parts)}

{"Gaps (could not find reliable information):" + chr(10) + chr(10).join(uncertain_parts) if uncertain_parts else ""}

Guidelines:
1. Integrate the sub-answers naturally
2. If there are gaps, acknowledge them honestly
3. Maintain a coherent narrative flow
4. Preserve important details and nuances
5. Use hedging language for lower-confidence portions

Provide the synthesized response only."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text, uncertain_parts

        except Exception as e:
            logger.error("Synthesis failed", error=str(e))
            # Fallback: concatenate answers
            fallback = "\n\n".join(r.answer for r in answered if r.answer)
            return fallback, uncertain_parts
