"""Cross-encoder reranking for search results."""

import asyncio
from functools import lru_cache
from typing import Optional

import numpy as np
import structlog
import torch
from sentence_transformers import CrossEncoder

from src.config.settings import get_settings
from src.retrieval.search import SearchResult

logger = structlog.get_logger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """Rerank search results using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize the reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run model on (auto-detected if not provided)
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.batch_size = batch_size
        self._model: Optional[CrossEncoder] = None
        self._lock = asyncio.Lock()

    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            logger.info(
                "Loading cross-encoder model",
                model=self.model_name,
                device=self.device,
            )
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
            )
            logger.info("Cross-encoder model loaded")
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[SearchResult]:
        """Rerank search results using cross-encoder.

        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of results to return (default: rerank_top_k from settings)
            min_score: Minimum reranker score to include

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        # Get settings for top_k
        current_settings = get_settings()
        top_k = top_k or current_settings.rerank_top_k
        # Don't filter by min_score - let top_k handle result limiting
        min_score = min_score if min_score is not None else 0.0

        logger.info("Reranking results", input_count=len(results), top_k=top_k)

        # Run reranking in thread pool
        async with self._lock:
            loop = asyncio.get_event_loop()
            reranked = await loop.run_in_executor(
                None,
                lambda: self._rerank_sync(query, results, top_k, min_score),
            )

        return reranked

    def _rerank_sync(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Synchronous reranking."""
        # Prepare query-document pairs
        pairs = [(query, r.text) for r in results]

        # Score pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Convert to probabilities (sigmoid for pointwise scoring)
        # Some models output logits, others output scores
        if scores.min() < 0 or scores.max() > 1:
            scores = 1 / (1 + np.exp(-scores))  # Sigmoid

        # Pair results with scores
        scored_results = list(zip(results, scores))

        # Sort by reranker score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Filter and update
        reranked = []
        for rank, (result, score) in enumerate(scored_results, 1):
            if score < min_score:
                continue
            if len(reranked) >= top_k:
                break

            # Update result with new rank and preserve original score
            result.rank = rank
            # Store reranker score as the primary score
            result.score = float(score)
            reranked.append(result)

        logger.debug(
            "Reranking complete",
            input=len(results),
            output=len(reranked),
            top_score=reranked[0].score if reranked else 0,
        )

        return reranked

    def compute_pairwise_scores(
        self,
        query: str,
        texts: list[str],
    ) -> np.ndarray:
        """Compute relevance scores for query-text pairs.

        Args:
            query: Query text
            texts: List of document texts

        Returns:
            Array of relevance scores
        """
        if not texts:
            return np.array([])

        pairs = [(query, text) for text in texts]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Normalize to [0, 1]
        if scores.min() < 0 or scores.max() > 1:
            scores = 1 / (1 + np.exp(-scores))

        return scores


# Global reranker instance
_reranker: Optional[CrossEncoderReranker] = None


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
