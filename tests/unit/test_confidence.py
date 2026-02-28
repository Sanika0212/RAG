"""Tests for confidence estimation."""

import pytest
from uuid import uuid4

from src.config.constants import ConfidenceBand
from src.retrieval.confidence import ConfidenceEstimator, ConfidenceResult
from src.retrieval.search import SearchResult


@pytest.fixture
def high_confidence_results():
    """Create search results that should yield high confidence."""
    return [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Aspirin works by inhibiting cyclooxygenase enzymes (COX-1 and COX-2), which reduces prostaglandin synthesis.",
            score=0.92,
            rank=1,
            keywords=["aspirin", "COX", "prostaglandin", "enzyme"],
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="The mechanism of aspirin involves irreversible inhibition of COX enzymes, leading to decreased thromboxane production.",
            score=0.88,
            rank=2,
            keywords=["aspirin", "COX", "thromboxane", "inhibition"],
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Aspirin's anti-inflammatory effects are mediated through COX-2 inhibition and reduced prostaglandin synthesis.",
            score=0.85,
            rank=3,
            keywords=["aspirin", "anti-inflammatory", "COX-2", "prostaglandin"],
        ),
    ]


@pytest.fixture
def low_confidence_results():
    """Create search results that should yield low confidence."""
    return [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="The weather forecast for tomorrow shows partly cloudy skies.",
            score=0.35,
            rank=1,
            keywords=["weather", "forecast", "cloudy"],
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Python is a popular programming language used for data science.",
            score=0.32,
            rank=2,
            keywords=["python", "programming", "data science"],
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="The stock market showed mixed results today.",
            score=0.30,
            rank=3,
            keywords=["stock", "market", "trading"],
        ),
    ]


def test_confidence_estimator_initialization():
    """Test confidence estimator initialization."""
    estimator = ConfidenceEstimator(
        high_threshold=0.8,
        low_threshold=0.5,
        use_llm_coverage=False,
    )

    assert estimator.high_threshold == 0.8
    assert estimator.low_threshold == 0.5
    assert estimator.use_llm_coverage is False


def test_top_score_signal_high():
    """Test top score signal with high-scoring results."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)
    signal = estimator._compute_top_score([
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="test",
            score=0.9,
            rank=1,
        )
    ])

    assert signal >= 0.8


def test_top_score_signal_low():
    """Test top score signal with low-scoring results."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)
    signal = estimator._compute_top_score([
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="test",
            score=0.3,
            rank=1,
        )
    ])

    assert signal < 0.5


def test_dropoff_signal_sharp():
    """Test dropoff signal with sharp score dropoff."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)

    results = [
        SearchResult(chunk_id=uuid4(), document_id=uuid4(), text=f"text{i}", score=1.0 - i * 0.2, rank=i)
        for i in range(5)
    ]

    signal = estimator._compute_dropoff(results)
    # Sharp dropoff should give reasonable signal
    assert signal > 0.2


def test_dropoff_signal_flat():
    """Test dropoff signal with flat scores."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)

    results = [
        SearchResult(chunk_id=uuid4(), document_id=uuid4(), text=f"text{i}", score=0.5, rank=i)
        for i in range(5)
    ]

    signal = estimator._compute_dropoff(results)
    # Flat distribution should give low signal
    assert signal < 0.3


def test_coherence_signal_high():
    """Test coherence signal with coherent results."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)

    results = [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="text",
            score=0.9,
            rank=1,
            keywords=["aspirin", "COX", "enzyme"],
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="text",
            score=0.8,
            rank=2,
            keywords=["aspirin", "COX", "inhibition"],
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="text",
            score=0.7,
            rank=3,
            keywords=["aspirin", "enzyme", "mechanism"],
        ),
    ]

    signal = estimator._compute_coherence(results)
    assert signal > 0.4


def test_coherence_signal_low():
    """Test coherence signal with incoherent results."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)

    results = [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="text",
            score=0.9,
            rank=1,
            keywords=["weather", "forecast"],
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="text",
            score=0.8,
            rank=2,
            keywords=["python", "programming"],
        ),
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="text",
            score=0.7,
            rank=3,
            keywords=["stock", "market"],
        ),
    ]

    signal = estimator._compute_coherence(results)
    # Different topics should give low coherence
    assert signal < 0.4


def test_empty_results():
    """Test confidence with no results."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)

    # Should handle empty results gracefully
    top_score = estimator._compute_top_score([])
    assert top_score == 0.0

    dropoff = estimator._compute_dropoff([])
    assert dropoff == 0.5  # Neutral

    coherence = estimator._compute_coherence([])
    assert coherence == 0.5  # Neutral


@pytest.mark.asyncio
async def test_full_estimation_high_confidence(high_confidence_results):
    """Test full confidence estimation with high-quality results."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)

    result = await estimator.estimate(
        query="How does aspirin work?",
        results=high_confidence_results,
    )

    assert isinstance(result, ConfidenceResult)
    assert result.score > 0.5
    # With good results, should be at least MEDIUM confidence
    assert result.band in [ConfidenceBand.HIGH, ConfidenceBand.MEDIUM]


@pytest.mark.asyncio
async def test_full_estimation_low_confidence(low_confidence_results):
    """Test full confidence estimation with poor results."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)

    result = await estimator.estimate(
        query="How does aspirin work?",
        results=low_confidence_results,
    )

    assert isinstance(result, ConfidenceResult)
    # Irrelevant results should give low confidence
    assert result.band == ConfidenceBand.LOW


def test_recommendation_generation():
    """Test that recommendations are generated based on signals."""
    estimator = ConfidenceEstimator(use_llm_coverage=False)

    # Low top score
    recs = estimator._generate_recommendations(
        {"top_score": 0.3, "dropoff": 0.5, "coherence": 0.5, "coverage": 0.5},
        [],
    )
    assert any("reformulat" in r.lower() for r in recs)

    # Low dropoff (broad query)
    recs = estimator._generate_recommendations(
        {"top_score": 0.7, "dropoff": 0.2, "coherence": 0.5, "coverage": 0.5},
        [],
    )
    assert any("broad" in r.lower() for r in recs)
