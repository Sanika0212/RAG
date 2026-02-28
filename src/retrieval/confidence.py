"""Confidence estimation for retrieval quality - Novel Contribution #1."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog
import anthropic

from src.config.constants import ConfidenceBand
from src.config.settings import get_settings
from src.retrieval.search import SearchResult

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class ConfidenceResult:
    """Result of confidence estimation."""

    score: float  # Overall confidence score [0, 1]
    band: ConfidenceBand  # HIGH, MEDIUM, LOW
    components: dict[str, float] = field(default_factory=dict)  # Individual signal scores
    coverage_report: str = ""  # Natural language coverage assessment
    gaps: list[str] = field(default_factory=list)  # Identified information gaps
    recommendations: list[str] = field(default_factory=list)  # Suggested actions


class ConfidenceEstimator:
    """Estimate retrieval confidence using multiple signals.

    Novel contribution: Multi-signal confidence calibration combining:
    1. Top similarity score (baseline quality)
    2. Score dropoff (sharp dropoff = good match specificity)
    3. Inter-chunk coherence (semantic consistency of results)
    4. Query coverage (does retrieved content address the query?)
    """

    def __init__(
        self,
        high_threshold: Optional[float] = None,
        low_threshold: Optional[float] = None,
        use_llm_coverage: bool = True,
        model: Optional[str] = None,
    ):
        """Initialize the confidence estimator.

        Args:
            high_threshold: Threshold for HIGH confidence band (default from settings)
            low_threshold: Threshold for LOW confidence band (default from settings)
            use_llm_coverage: Use LLM for query coverage analysis
            model: Claude model for coverage analysis (default from settings)
        """
        # Read settings at runtime to avoid stale cached values
        current_settings = get_settings()
        self.high_threshold = high_threshold if high_threshold is not None else current_settings.confidence_high_threshold
        self.low_threshold = low_threshold if low_threshold is not None else current_settings.confidence_low_threshold
        self.use_llm_coverage = use_llm_coverage
        self.model = model if model is not None else current_settings.agent_model
        self._client: Optional[anthropic.AsyncAnthropic] = None

        # Signal weights (learned or tuned)
        self.weights = {
            "top_score": 0.25,
            "dropoff": 0.20,
            "coherence": 0.25,
            "coverage": 0.30,
        }

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    async def estimate(
        self,
        query: str,
        results: list[SearchResult],
        query_embedding: Optional[np.ndarray] = None,
    ) -> ConfidenceResult:
        """Estimate confidence in retrieval results.

        Args:
            query: Original search query
            results: Retrieved and reranked results
            query_embedding: Query embedding (for additional analysis)

        Returns:
            ConfidenceResult with score, band, and detailed analysis
        """
        if not results:
            return ConfidenceResult(
                score=0.0,
                band=ConfidenceBand.LOW,
                components={},
                coverage_report="No results retrieved.",
                gaps=["No relevant documents found"],
                recommendations=["Try reformulating the query", "Check if documents exist in the knowledge base"],
            )

        logger.info("Estimating confidence", query=query[:50], results=len(results))

        # Compute individual signals
        components = {}

        # Signal 1: Top score (highest relevance)
        components["top_score"] = self._compute_top_score(results)

        # Signal 2: Score dropoff (specificity)
        components["dropoff"] = self._compute_dropoff(results)

        # Signal 3: Inter-chunk coherence
        components["coherence"] = self._compute_coherence(results)

        # Signal 4: Query coverage (requires LLM)
        coverage_result = await self._compute_coverage(query, results)
        components["coverage"] = coverage_result["score"]
        coverage_report = coverage_result["report"]
        gaps = coverage_result["gaps"]

        # Compute weighted score
        score = sum(
            self.weights[signal] * components[signal]
            for signal in self.weights
        )

        # Determine band
        if score >= self.high_threshold:
            band = ConfidenceBand.HIGH
        elif score >= self.low_threshold:
            band = ConfidenceBand.MEDIUM
        else:
            band = ConfidenceBand.LOW

        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations(components, gaps)

        result = ConfidenceResult(
            score=score,
            band=band,
            components=components,
            coverage_report=coverage_report,
            gaps=gaps,
            recommendations=recommendations,
        )

        logger.info(
            "Confidence estimated",
            score=round(score, 3),
            band=band.value,
            components={k: round(v, 3) for k, v in components.items()},
        )

        return result

    def _compute_top_score(self, results: list[SearchResult]) -> float:
        """Compute confidence signal from top result score."""
        if not results:
            return 0.0

        top_score = results[0].score

        # Normalize to [0, 1] with a sigmoid-like curve
        # Scores above 0.7 map to high confidence
        if top_score >= 0.8:
            return 1.0
        elif top_score >= 0.6:
            return 0.8 + (top_score - 0.6) * 1.0  # 0.8-1.0
        elif top_score >= 0.4:
            return 0.5 + (top_score - 0.4) * 1.5  # 0.5-0.8
        else:
            return top_score * 1.25  # 0-0.5

    def _compute_dropoff(self, results: list[SearchResult]) -> float:
        """Compute confidence signal from score dropoff.

        Sharp dropoff indicates clear best match(es).
        Flat distribution suggests ambiguity or poor matches.
        """
        if len(results) < 2:
            return 0.5  # Neutral when insufficient data

        scores = [r.score for r in results]

        # Compute relative dropoff between top and 2nd, 5th, 10th results
        top_score = scores[0]
        if top_score == 0:
            return 0.0

        dropoffs = []

        # Top 1 to Top 2 dropoff
        if len(scores) >= 2:
            dropoff_1_2 = (top_score - scores[1]) / top_score
            dropoffs.append(dropoff_1_2 * 0.5)  # Weight: 50%

        # Top 1 to Top 5 dropoff
        if len(scores) >= 5:
            dropoff_1_5 = (top_score - scores[4]) / top_score
            dropoffs.append(dropoff_1_5 * 0.3)  # Weight: 30%

        # Score variance (lower = more uniform = less confident)
        variance = np.var(scores[:min(10, len(scores))])
        # Normalize variance (higher variance = better differentiation)
        variance_signal = min(1.0, variance * 10)
        dropoffs.append(variance_signal * 0.2)  # Weight: 20%

        return sum(dropoffs)

    def _compute_coherence(self, results: list[SearchResult]) -> float:
        """Compute inter-chunk coherence using keyword and topic overlap.

        High coherence suggests results are about the same topic.
        Low coherence suggests scattered, unrelated results.
        """
        if len(results) < 2:
            return 0.5

        # Compute pairwise keyword overlap
        all_keywords = [set(r.keywords) for r in results if r.keywords]
        if len(all_keywords) < 2:
            return 0.5

        overlaps = []
        for i in range(len(all_keywords)):
            for j in range(i + 1, min(i + 5, len(all_keywords))):  # Compare with next 4
                if all_keywords[i] and all_keywords[j]:
                    intersection = len(all_keywords[i] & all_keywords[j])
                    union = len(all_keywords[i] | all_keywords[j])
                    if union > 0:
                        overlaps.append(intersection / union)

        if not overlaps:
            return 0.5

        # Average overlap, with bonus for high consistency
        avg_overlap = np.mean(overlaps)
        consistency_bonus = 0.2 if np.std(overlaps) < 0.2 else 0

        return min(1.0, avg_overlap + consistency_bonus)

    async def _compute_coverage(
        self,
        query: str,
        results: list[SearchResult],
    ) -> dict:
        """Compute query coverage using LLM analysis.

        Determines how well the retrieved chunks address the query.
        """
        if not self.use_llm_coverage:
            return {
                "score": 0.5,  # Neutral without LLM
                "report": "LLM coverage analysis disabled.",
                "gaps": [],
            }

        # Prepare context summary
        context_summaries = []
        for i, r in enumerate(results[:5], 1):  # Top 5 results
            summary = r.summary or r.text[:200]
            context_summaries.append(f"{i}. {summary}")

        context_text = "\n".join(context_summaries)

        prompt = f"""Analyze how well these retrieved documents address the query.

Query: "{query}"

Retrieved Documents:
{context_text}

Respond with JSON:
{{
    "coverage_score": 0.0-1.0,  // How completely the documents address the query
    "coverage_description": "Brief explanation of coverage",
    "addressed_aspects": ["list", "of", "aspects", "covered"],
    "gaps": ["list", "of", "missing", "information"]
}}

Coverage score guide:
- 1.0: Query fully answered by retrieved documents
- 0.7-0.9: Most aspects covered, minor gaps
- 0.4-0.6: Partial coverage, significant gaps
- 0.1-0.3: Tangentially related, major gaps
- 0.0: No relevant information

Respond with valid JSON only."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            return self._parse_coverage_response(content)

        except Exception as e:
            logger.warning("Coverage analysis failed", error=str(e))
            return {
                "score": 0.5,
                "report": f"Coverage analysis failed: {str(e)}",
                "gaps": [],
            }

    def _parse_coverage_response(self, response: str) -> dict:
        """Parse LLM coverage response."""
        import json

        try:
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()

            data = json.loads(json_str)

            score = float(data.get("coverage_score", 0.5))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            addressed = data.get("addressed_aspects", [])
            gaps = data.get("gaps", [])

            report = data.get("coverage_description", "")
            if addressed:
                report += f" Addressed: {', '.join(addressed[:3])}."
            if gaps:
                report += f" Missing: {', '.join(gaps[:3])}."

            return {
                "score": score,
                "report": report,
                "gaps": gaps,
            }

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse coverage response", error=str(e))
            return {
                "score": 0.5,
                "report": "Unable to parse coverage analysis.",
                "gaps": [],
            }

    def _generate_recommendations(
        self,
        components: dict[str, float],
        gaps: list[str],
    ) -> list[str]:
        """Generate recommendations based on confidence analysis."""
        recommendations = []

        # Low top score → reformulate query
        if components.get("top_score", 1.0) < 0.5:
            recommendations.append("Consider reformulating the query with different terminology")

        # Low dropoff → query may be too broad
        if components.get("dropoff", 1.0) < 0.3:
            recommendations.append("Query may be too broad; consider adding specific constraints")

        # Low coherence → scattered information
        if components.get("coherence", 1.0) < 0.4:
            recommendations.append("Information appears scattered; may need multi-hop retrieval")

        # Low coverage → decompose query
        if components.get("coverage", 1.0) < 0.4:
            recommendations.append("Query coverage is low; consider breaking into sub-questions")

        # Add gap-specific recommendations
        for gap in gaps[:2]:  # Top 2 gaps
            recommendations.append(f"Missing information: {gap}")

        return recommendations


async def calibrate_thresholds(
    test_queries: list[dict],
    estimator: ConfidenceEstimator,
) -> dict[str, float]:
    """Calibrate confidence thresholds using labeled test data.

    Args:
        test_queries: List of {"query": str, "results": list, "label": "good"|"bad"}
        estimator: ConfidenceEstimator instance

    Returns:
        Recommended threshold values
    """
    scores_good = []
    scores_bad = []

    for item in test_queries:
        result = await estimator.estimate(item["query"], item["results"])
        if item["label"] == "good":
            scores_good.append(result.score)
        else:
            scores_bad.append(result.score)

    if not scores_good or not scores_bad:
        return {
            "high_threshold": settings.confidence_high_threshold,
            "low_threshold": settings.confidence_low_threshold,
        }

    # Find thresholds that separate good from bad
    # High threshold: 90th percentile of good scores
    high_threshold = np.percentile(scores_good, 90)

    # Low threshold: 10th percentile of good scores (or 90th of bad)
    low_threshold = max(
        np.percentile(scores_good, 10),
        np.percentile(scores_bad, 90),
    )

    return {
        "high_threshold": float(high_threshold),
        "low_threshold": float(low_threshold),
    }
