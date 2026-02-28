"""Failure diagnosis for retrieval - Novel Contribution #2."""

from dataclasses import dataclass, field
from typing import Optional

import anthropic
import structlog

from src.config.constants import FailureMode
from src.config.settings import get_settings
from src.retrieval.search import SearchResult
from src.retrieval.confidence import ConfidenceResult

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class DiagnosisResult:
    """Result of failure diagnosis."""

    failure_mode: FailureMode
    confidence: float  # Confidence in diagnosis
    reasoning: str  # Explanation of diagnosis
    suggested_actions: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)  # For AMBIGUITY
    reformulated_query: Optional[str] = None  # For VOCAB_MISMATCH
    hierarchy_direction: Optional[str] = None  # "up" or "down" for GRANULARITY_MISMATCH


class FailureDiagnoser:
    """Diagnose retrieval failures to determine correction strategy.

    Novel contribution: Classify failures into actionable categories:
    - AMBIGUITY: Query is ambiguous, needs decomposition
    - VOCAB_MISMATCH: Query terms don't match document vocabulary
    - INFO_SCATTER: Information spread across multiple chunks
    - KNOWLEDGE_GAP: Information not in knowledge base
    - GRANULARITY_MISMATCH: Query at wrong abstraction level
    """

    def __init__(
        self,
        model: str = settings.agent_model,
    ):
        """Initialize the diagnoser.

        Args:
            model: Claude model to use for diagnosis
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

    async def diagnose(
        self,
        query: str,
        results: list[SearchResult],
        confidence: ConfidenceResult,
    ) -> DiagnosisResult:
        """Diagnose why retrieval failed.

        Args:
            query: Original search query
            results: Retrieved results (low confidence)
            confidence: Confidence estimation result

        Returns:
            DiagnosisResult with failure mode and suggested corrections
        """
        logger.info(
            "Diagnosing retrieval failure",
            query=query[:50],
            confidence=confidence.score,
            results=len(results),
        )

        # First, try rule-based diagnosis
        rule_diagnosis = self._rule_based_diagnosis(query, results, confidence)

        # If rule-based is confident, use it
        if rule_diagnosis.confidence > 0.7:
            return rule_diagnosis

        # Otherwise, use LLM for nuanced diagnosis
        llm_diagnosis = await self._llm_diagnosis(query, results, confidence)

        # Combine rule-based and LLM diagnoses
        if llm_diagnosis.confidence > rule_diagnosis.confidence:
            return llm_diagnosis
        else:
            return rule_diagnosis

    def _rule_based_diagnosis(
        self,
        query: str,
        results: list[SearchResult],
        confidence: ConfidenceResult,
    ) -> DiagnosisResult:
        """Apply rule-based heuristics for diagnosis."""
        components = confidence.components
        gaps = confidence.gaps

        # Check for AMBIGUITY: multiple interpretations possible
        if self._detect_ambiguity(query, results):
            return DiagnosisResult(
                failure_mode=FailureMode.AMBIGUITY,
                confidence=0.8,
                reasoning="Query appears to have multiple interpretations based on diverse result topics.",
                suggested_actions=["Decompose into specific sub-questions"],
                sub_queries=self._generate_sub_queries_heuristic(query),
            )

        # Check for VOCAB_MISMATCH: low keyword overlap
        if components.get("coherence", 1.0) > 0.6 and components.get("top_score", 1.0) < 0.4:
            return DiagnosisResult(
                failure_mode=FailureMode.VOCAB_MISMATCH,
                confidence=0.7,
                reasoning="Results are coherent but scores are low, suggesting terminology mismatch.",
                suggested_actions=["Reformulate with medical/scientific synonyms"],
                reformulated_query=self._generate_synonym_query_heuristic(query),
            )

        # Check for INFO_SCATTER: low coherence, moderate scores
        if components.get("coherence", 1.0) < 0.4 and components.get("top_score", 1.0) > 0.5:
            return DiagnosisResult(
                failure_mode=FailureMode.INFO_SCATTER,
                confidence=0.6,
                reasoning="Information appears scattered across unrelated chunks.",
                suggested_actions=["Use multi-hop retrieval to connect information"],
            )

        # Check for GRANULARITY_MISMATCH: specific query, general results (or vice versa)
        if self._detect_granularity_mismatch(query, results):
            direction = self._determine_hierarchy_direction(query, results)
            return DiagnosisResult(
                failure_mode=FailureMode.GRANULARITY_MISMATCH,
                confidence=0.6,
                reasoning=f"Query specificity doesn't match result granularity. Need to go {direction}.",
                suggested_actions=[f"Walk hierarchy {direction}"],
                hierarchy_direction=direction,
            )

        # Default to KNOWLEDGE_GAP if nothing else matches
        return DiagnosisResult(
            failure_mode=FailureMode.KNOWLEDGE_GAP,
            confidence=0.5,
            reasoning="Unable to find matching content. Information may not exist in knowledge base.",
            suggested_actions=["Verify knowledge base coverage", "Consider abstaining"],
        )

    async def _llm_diagnosis(
        self,
        query: str,
        results: list[SearchResult],
        confidence: ConfidenceResult,
    ) -> DiagnosisResult:
        """Use LLM for nuanced failure diagnosis."""
        # Prepare result summaries
        result_summaries = []
        for i, r in enumerate(results[:5], 1):
            summary = r.summary or r.text[:150]
            keywords = ", ".join(r.keywords[:5]) if r.keywords else "none"
            result_summaries.append(f"{i}. [{keywords}] {summary}")

        results_text = "\n".join(result_summaries)

        prompt = f"""Diagnose why this retrieval failed to provide confident results.

Query: "{query}"

Retrieved Results (low confidence):
{results_text}

Confidence Analysis:
- Overall score: {confidence.score:.2f}
- Coverage: {confidence.components.get('coverage', 'N/A')}
- Coherence: {confidence.components.get('coherence', 'N/A')}
- Gaps: {', '.join(confidence.gaps[:3]) if confidence.gaps else 'None identified'}

Failure Modes:
1. AMBIGUITY - Query has multiple interpretations
2. VOCAB_MISMATCH - Query terms don't match document terminology
3. INFO_SCATTER - Relevant info spread across many unrelated chunks
4. KNOWLEDGE_GAP - Information not in the knowledge base
5. GRANULARITY_MISMATCH - Query too specific/general for available content

Respond with JSON:
{{
    "failure_mode": "AMBIGUITY|VOCAB_MISMATCH|INFO_SCATTER|KNOWLEDGE_GAP|GRANULARITY_MISMATCH",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "suggested_actions": ["action1", "action2"],
    "sub_queries": ["query1", "query2"] // Only for AMBIGUITY
    "reformulated_query": "alternative query" // Only for VOCAB_MISMATCH
    "hierarchy_direction": "up|down" // Only for GRANULARITY_MISMATCH
}}

Respond with valid JSON only."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            return self._parse_diagnosis_response(content)

        except Exception as e:
            logger.error("LLM diagnosis failed", error=str(e))
            return DiagnosisResult(
                failure_mode=FailureMode.KNOWLEDGE_GAP,
                confidence=0.3,
                reasoning=f"Diagnosis failed: {str(e)}",
                suggested_actions=["Manual review required"],
            )

    def _parse_diagnosis_response(self, response: str) -> DiagnosisResult:
        """Parse LLM diagnosis response."""
        import json

        try:
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()

            data = json.loads(json_str)

            failure_mode = FailureMode(data["failure_mode"])
            confidence = float(data.get("confidence", 0.5))

            return DiagnosisResult(
                failure_mode=failure_mode,
                confidence=confidence,
                reasoning=data.get("reasoning", ""),
                suggested_actions=data.get("suggested_actions", []),
                sub_queries=data.get("sub_queries", []),
                reformulated_query=data.get("reformulated_query"),
                hierarchy_direction=data.get("hierarchy_direction"),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse diagnosis", error=str(e))
            return DiagnosisResult(
                failure_mode=FailureMode.KNOWLEDGE_GAP,
                confidence=0.3,
                reasoning="Failed to parse diagnosis response",
                suggested_actions=["Manual review required"],
            )

    def _detect_ambiguity(
        self,
        query: str,
        results: list[SearchResult],
    ) -> bool:
        """Detect if query is ambiguous based on result diversity."""
        if len(results) < 3:
            return False

        # Check topic diversity
        all_keywords = []
        for r in results[:5]:
            all_keywords.extend(r.keywords)

        if not all_keywords:
            return False

        unique_keywords = set(all_keywords)
        repetition_ratio = len(all_keywords) / len(unique_keywords)

        # Low repetition = high diversity = potential ambiguity
        return repetition_ratio < 1.5

    def _detect_granularity_mismatch(
        self,
        query: str,
        results: list[SearchResult],
    ) -> bool:
        """Detect if query granularity doesn't match results."""
        if not results:
            return False

        # Simple heuristic: query length vs result text length
        query_words = len(query.split())
        avg_result_words = sum(len(r.text.split()) for r in results[:3]) / 3

        # Very specific query (many words) but short general results
        if query_words > 15 and avg_result_words < 100:
            return True

        # Very short query but very long detailed results
        if query_words < 5 and avg_result_words > 300:
            return True

        return False

    def _determine_hierarchy_direction(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str:
        """Determine whether to go up or down in the hierarchy."""
        query_words = len(query.split())

        # Long specific query with short results → need more specific results
        if query_words > 10:
            return "down"

        # Short query with very detailed results → need higher-level overview
        return "up"

    def _generate_sub_queries_heuristic(self, query: str) -> list[str]:
        """Generate sub-queries using simple heuristics."""
        # Split by "and", "or", conjunctions
        import re

        parts = re.split(r'\band\b|\bor\b|,', query, flags=re.IGNORECASE)
        sub_queries = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

        if len(sub_queries) < 2:
            # Try splitting by question words
            if "?" in query:
                sub_queries = [query.replace("?", "").strip()]

        return sub_queries[:3]  # Max 3 sub-queries

    def _generate_synonym_query_heuristic(self, query: str) -> str:
        """Generate synonym-based reformulation heuristic."""
        # Simple approach: just return the original query
        # In practice, this would use a medical synonym dictionary
        return query
