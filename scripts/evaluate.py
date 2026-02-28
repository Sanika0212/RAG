#!/usr/bin/env python3
"""Evaluation script for RAG system."""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from src.database.connection import init_db, close_db, get_db
from src.agents.graph import RAGAgentGraph

logger = structlog.get_logger(__name__)


@dataclass
class EvalResult:
    """Result for a single evaluation query."""

    query_id: str
    query: str
    expected_answer: Optional[str]
    predicted_answer: str
    confidence_score: float
    confidence_band: str
    correction_attempts: int
    latency_ms: int

    # Metrics (computed post-hoc)
    answer_correctness: Optional[float] = None
    faithfulness: Optional[float] = None
    retrieval_precision: Optional[float] = None
    abstention_correct: Optional[bool] = None


@dataclass
class EvalSummary:
    """Summary of evaluation results."""

    total_queries: int
    avg_answer_correctness: float
    avg_faithfulness: float
    avg_retrieval_precision: float
    abstention_precision: float
    hallucination_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    confidence_distribution: dict[str, int] = field(default_factory=dict)


class RAGEvaluator:
    """Evaluate RAG system against a test dataset."""

    def __init__(self):
        """Initialize the evaluator."""
        self.agent = RAGAgentGraph()
        self.results: list[EvalResult] = []

    async def evaluate_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
    ) -> EvalSummary:
        """Evaluate against a dataset.

        Expected dataset format (JSONL):
        {"query": "...", "expected_answer": "...", "should_abstain": false}

        Args:
            dataset_path: Path to evaluation dataset
            output_path: Optional path to save detailed results

        Returns:
            EvalSummary with aggregate metrics
        """
        logger.info("Loading evaluation dataset", path=dataset_path)

        # Load dataset
        with open(dataset_path) as f:
            queries = [json.loads(line) for line in f if line.strip()]

        logger.info("Starting evaluation", total_queries=len(queries))

        # Evaluate each query
        async with get_db() as session:
            for i, item in enumerate(queries):
                try:
                    result = await self._evaluate_single(
                        query_id=str(i),
                        query=item["query"],
                        expected_answer=item.get("expected_answer"),
                        should_abstain=item.get("should_abstain", False),
                        session=session,
                    )
                    self.results.append(result)

                    if (i + 1) % 10 == 0:
                        logger.info("Evaluation progress", completed=i + 1, total=len(queries))

                except Exception as e:
                    logger.error("Evaluation failed for query", query_id=i, error=str(e))

        # Compute summary
        summary = self._compute_summary()

        # Save results
        if output_path:
            self._save_results(output_path, summary)

        return summary

    async def _evaluate_single(
        self,
        query_id: str,
        query: str,
        expected_answer: Optional[str],
        should_abstain: bool,
        session,
    ) -> EvalResult:
        """Evaluate a single query."""
        start_time = datetime.now()

        # Run RAG agent
        state = await self.agent.run(query, session)

        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        result = EvalResult(
            query_id=query_id,
            query=query,
            expected_answer=expected_answer,
            predicted_answer=state.response,
            confidence_score=state.confidence.score if state.confidence else 0.0,
            confidence_band=state.confidence.band.value if state.confidence else "low",
            correction_attempts=state.correction_attempts,
            latency_ms=latency_ms,
        )

        # Compute metrics
        if expected_answer:
            result.answer_correctness = self._compute_answer_correctness(
                expected_answer, state.response
            )

        if state.validation_result:
            result.faithfulness = 1.0 - state.validation_result.get("hallucination_risk", 0.0)

        # Check abstention correctness
        is_abstention = "cannot provide" in state.response.lower() or "insufficient" in state.response.lower()
        result.abstention_correct = is_abstention == should_abstain

        return result

    def _compute_answer_correctness(
        self,
        expected: str,
        predicted: str,
    ) -> float:
        """Compute answer correctness using simple overlap."""
        # Simple word overlap for now
        # In production, use a proper metric like ROUGE or BERTScore
        expected_words = set(expected.lower().split())
        predicted_words = set(predicted.lower().split())

        if not expected_words:
            return 0.0

        overlap = len(expected_words & predicted_words)
        precision = overlap / len(predicted_words) if predicted_words else 0.0
        recall = overlap / len(expected_words)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _compute_summary(self) -> EvalSummary:
        """Compute evaluation summary."""
        if not self.results:
            return EvalSummary(
                total_queries=0,
                avg_answer_correctness=0.0,
                avg_faithfulness=0.0,
                avg_retrieval_precision=0.0,
                abstention_precision=0.0,
                hallucination_rate=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
            )

        # Compute averages
        correctness_scores = [r.answer_correctness for r in self.results if r.answer_correctness is not None]
        faithfulness_scores = [r.faithfulness for r in self.results if r.faithfulness is not None]
        latencies = [r.latency_ms for r in self.results]

        avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0

        # Abstention precision
        abstention_results = [r for r in self.results if r.abstention_correct is not None]
        abstention_precision = (
            sum(1 for r in abstention_results if r.abstention_correct) / len(abstention_results)
            if abstention_results else 0.0
        )

        # Hallucination rate
        hallucination_rate = 1.0 - avg_faithfulness if avg_faithfulness > 0 else 0.0

        # Latency percentiles
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0

        # Confidence distribution
        confidence_dist: dict[str, int] = {}
        for r in self.results:
            band = r.confidence_band
            confidence_dist[band] = confidence_dist.get(band, 0) + 1

        return EvalSummary(
            total_queries=len(self.results),
            avg_answer_correctness=avg_correctness,
            avg_faithfulness=avg_faithfulness,
            avg_retrieval_precision=0.0,  # Would need retrieval labels
            abstention_precision=abstention_precision,
            hallucination_rate=hallucination_rate,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            confidence_distribution=confidence_dist,
        )

    def _save_results(self, output_path: str, summary: EvalSummary) -> None:
        """Save evaluation results to file."""
        output = {
            "summary": {
                "total_queries": summary.total_queries,
                "avg_answer_correctness": summary.avg_answer_correctness,
                "avg_faithfulness": summary.avg_faithfulness,
                "abstention_precision": summary.abstention_precision,
                "hallucination_rate": summary.hallucination_rate,
                "avg_latency_ms": summary.avg_latency_ms,
                "p50_latency_ms": summary.p50_latency_ms,
                "p95_latency_ms": summary.p95_latency_ms,
                "confidence_distribution": summary.confidence_distribution,
            },
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "expected_answer": r.expected_answer,
                    "predicted_answer": r.predicted_answer,
                    "confidence_score": r.confidence_score,
                    "confidence_band": r.confidence_band,
                    "correction_attempts": r.correction_attempts,
                    "latency_ms": r.latency_ms,
                    "answer_correctness": r.answer_correctness,
                    "faithfulness": r.faithfulness,
                    "abstention_correct": r.abstention_correct,
                }
                for r in self.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info("Results saved", path=output_path)


async def run_evaluation(dataset_path: str, output_path: Optional[str]) -> None:
    """Run the evaluation."""
    await init_db()

    try:
        evaluator = RAGEvaluator()
        summary = await evaluator.evaluate_dataset(dataset_path, output_path)

        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Queries: {summary.total_queries}")
        print(f"Answer Correctness: {summary.avg_answer_correctness:.3f}")
        print(f"Faithfulness: {summary.avg_faithfulness:.3f}")
        print(f"Abstention Precision: {summary.abstention_precision:.3f}")
        print(f"Hallucination Rate: {summary.hallucination_rate:.3f}")
        print(f"Avg Latency: {summary.avg_latency_ms:.0f}ms")
        print(f"P50 Latency: {summary.p50_latency_ms:.0f}ms")
        print(f"P95 Latency: {summary.p95_latency_ms:.0f}ms")
        print("\nConfidence Distribution:")
        for band, count in summary.confidence_distribution.items():
            print(f"  {band}: {count}")

    finally:
        await close_db()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "dataset",
        help="Path to evaluation dataset (JSONL format)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save detailed results (JSON)",
    )

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return 1

    asyncio.run(run_evaluation(args.dataset, args.output))
    return 0


if __name__ == "__main__":
    exit(main())
