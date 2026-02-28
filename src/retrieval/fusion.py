"""Reciprocal Rank Fusion for combining multiple ranked lists."""

from collections import defaultdict
from typing import TypeVar
from uuid import UUID

T = TypeVar("T")


class ReciprocalRankFusion:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum over lists: weight * 1 / (k + rank)
    where k is a constant (typically 60) to prevent high scores for top ranks.
    """

    def __init__(self, k: int = 60):
        """Initialize RRF.

        Args:
            k: Smoothing constant (higher = more equal weighting across ranks)
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: list[list[UUID]],
        weights: list[float] | None = None,
    ) -> list[tuple[UUID, float]]:
        """Fuse multiple ranked lists into a single ranking.

        Args:
            ranked_lists: List of ranked lists (each list is ordered by rank)
            weights: Optional weights for each list (default: equal weights)

        Returns:
            List of (item, fused_score) tuples, sorted by score descending
        """
        if not ranked_lists:
            return []

        # Default to equal weights
        if weights is None:
            weights = [1.0] * len(ranked_lists)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Compute RRF scores
        scores: dict[UUID, float] = defaultdict(float)

        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, item in enumerate(ranked_list, 1):
                # RRF formula: weight * 1 / (k + rank)
                scores[item] += weight / (self.k + rank)

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def fuse_with_scores(
        self,
        ranked_lists_with_scores: list[list[tuple[UUID, float]]],
        weights: list[float] | None = None,
        use_original_scores: bool = False,
    ) -> list[tuple[UUID, float]]:
        """Fuse ranked lists that include original scores.

        Args:
            ranked_lists_with_scores: List of [(item, score), ...] lists
            weights: Optional weights for each list
            use_original_scores: If True, use original scores instead of ranks

        Returns:
            List of (item, fused_score) tuples, sorted by score descending
        """
        if not ranked_lists_with_scores:
            return []

        if use_original_scores:
            # Score-based fusion: weighted sum of normalized scores
            return self._score_fusion(ranked_lists_with_scores, weights)
        else:
            # Standard RRF: convert to ranked lists
            ranked_lists = [
                [item for item, _ in scored_list]
                for scored_list in ranked_lists_with_scores
            ]
            return self.fuse(ranked_lists, weights)

    def _score_fusion(
        self,
        ranked_lists_with_scores: list[list[tuple[UUID, float]]],
        weights: list[float] | None,
    ) -> list[tuple[UUID, float]]:
        """Fuse using normalized scores instead of ranks."""
        if weights is None:
            weights = [1.0] * len(ranked_lists_with_scores)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Compute min/max for normalization
        all_scores = []
        for scored_list in ranked_lists_with_scores:
            all_scores.extend(score for _, score in scored_list)

        if not all_scores:
            return []

        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        # Compute fused scores
        scores: dict[UUID, float] = defaultdict(float)

        for scored_list, weight in zip(ranked_lists_with_scores, weights):
            for item, score in scored_list:
                # Normalize to [0, 1] and apply weight
                normalized_score = (score - min_score) / score_range
                scores[item] += weight * normalized_score

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results


class CombSUM:
    """Simple score combination by summing normalized scores."""

    def fuse(
        self,
        scored_lists: list[list[tuple[UUID, float]]],
        weights: list[float] | None = None,
    ) -> list[tuple[UUID, float]]:
        """Combine scores by weighted sum."""
        if not scored_lists:
            return []

        if weights is None:
            weights = [1.0] * len(scored_lists)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Sum scores
        scores: dict[UUID, float] = defaultdict(float)

        for scored_list, weight in zip(scored_lists, weights):
            for item, score in scored_list:
                scores[item] += weight * score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class CombMNZ:
    """CombMNZ: multiply sum by number of lists containing the item."""

    def fuse(
        self,
        scored_lists: list[list[tuple[UUID, float]]],
        weights: list[float] | None = None,
    ) -> list[tuple[UUID, float]]:
        """Combine using CombMNZ formula."""
        if not scored_lists:
            return []

        if weights is None:
            weights = [1.0] * len(scored_lists)

        # Count occurrences and sum scores
        scores: dict[UUID, float] = defaultdict(float)
        counts: dict[UUID, int] = defaultdict(int)

        for scored_list, weight in zip(scored_lists, weights):
            for item, score in scored_list:
                scores[item] += weight * score
                counts[item] += 1

        # Apply MNZ multiplier
        final_scores = {
            item: score * counts[item]
            for item, score in scores.items()
        }

        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
