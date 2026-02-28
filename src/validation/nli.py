"""Natural Language Inference scoring for claim grounding."""

import asyncio
from functools import lru_cache
from typing import Optional

import numpy as np
import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class NLIScorer:
    """Score claim-evidence pairs using NLI models.

    Uses DeBERTa-v3-large-mnli or similar for:
    - Entailment detection (evidence supports claim)
    - Contradiction detection (evidence contradicts claim)
    - Neutral detection (no clear relationship)
    """

    def __init__(
        self,
        model_name: str = settings.nli_model,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        """Initialize the NLI scorer.

        Args:
            model_name: HuggingFace model name for NLI
            device: Device to run model on (auto-detected if not provided)
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.batch_size = batch_size
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._lock = asyncio.Lock()

    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self) -> AutoModelForSequenceClassification:
        """Lazy load the NLI model."""
        if self._model is None:
            logger.info("Loading NLI model", model=self.model_name, device=self.device)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info("NLI model loaded")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    async def score_grounding(
        self,
        claim: str,
        evidence_texts: list[str],
    ) -> list[float]:
        """Score how well each evidence text supports the claim.

        Args:
            claim: The claim to verify
            evidence_texts: List of potential evidence texts

        Returns:
            List of entailment scores (0-1) for each evidence
        """
        if not evidence_texts:
            return []

        async with self._lock:
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self._score_grounding_sync(claim, evidence_texts),
            )
        return scores

    def _score_grounding_sync(
        self,
        claim: str,
        evidence_texts: list[str],
    ) -> list[float]:
        """Synchronous scoring of claim-evidence pairs."""
        scores = []

        for i in range(0, len(evidence_texts), self.batch_size):
            batch_evidence = evidence_texts[i : i + self.batch_size]
            batch_scores = self._score_batch(claim, batch_evidence)
            scores.extend(batch_scores)

        return scores

    def _score_batch(
        self,
        claim: str,
        evidence_batch: list[str],
    ) -> list[float]:
        """Score a batch of evidence texts against a claim."""
        # Create premise-hypothesis pairs
        # NLI format: premise (evidence) → hypothesis (claim)
        pairs = [(evidence, claim) for evidence in evidence_batch]

        # Tokenize
        inputs = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get entailment probability
        # Typical label ordering: contradiction=0, neutral=1, entailment=2
        # But check model config for actual ordering
        entailment_idx = self._get_entailment_index()
        entailment_probs = probs[:, entailment_idx].cpu().numpy()

        return entailment_probs.tolist()

    def _get_entailment_index(self) -> int:
        """Get the index of the entailment label."""
        # Common label orderings
        # DeBERTa MNLI: contradiction=0, neutral=1, entailment=2
        # Some models: entailment=0, contradiction=1, neutral=2

        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
            id2label = self.model.config.id2label
            for idx, label in id2label.items():
                if "entail" in label.lower():
                    return int(idx)

        # Default assumption
        return 2

    async def score_contradiction(
        self,
        claim: str,
        evidence_texts: list[str],
    ) -> list[float]:
        """Score contradiction between claim and evidence.

        Args:
            claim: The claim to check
            evidence_texts: List of evidence texts

        Returns:
            List of contradiction scores (0-1) for each evidence
        """
        if not evidence_texts:
            return []

        async with self._lock:
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self._score_contradiction_sync(claim, evidence_texts),
            )
        return scores

    def _score_contradiction_sync(
        self,
        claim: str,
        evidence_texts: list[str],
    ) -> list[float]:
        """Synchronous contradiction scoring."""
        scores = []

        for i in range(0, len(evidence_texts), self.batch_size):
            batch_evidence = evidence_texts[i : i + self.batch_size]
            batch_scores = self._score_contradiction_batch(claim, batch_evidence)
            scores.extend(batch_scores)

        return scores

    def _score_contradiction_batch(
        self,
        claim: str,
        evidence_batch: list[str],
    ) -> list[float]:
        """Score contradiction for a batch."""
        pairs = [(evidence, claim) for evidence in evidence_batch]

        inputs = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)

        # Contradiction is typically index 0
        contradiction_idx = self._get_contradiction_index()
        contradiction_probs = probs[:, contradiction_idx].cpu().numpy()

        return contradiction_probs.tolist()

    def _get_contradiction_index(self) -> int:
        """Get the index of the contradiction label."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
            id2label = self.model.config.id2label
            for idx, label in id2label.items():
                if "contra" in label.lower():
                    return int(idx)

        return 0  # Default assumption

    async def classify_relationship(
        self,
        claim: str,
        evidence: str,
    ) -> dict:
        """Get full NLI classification for a claim-evidence pair.

        Args:
            claim: The claim
            evidence: The evidence text

        Returns:
            Dict with entailment, contradiction, and neutral probabilities
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._classify_relationship_sync(claim, evidence),
            )
        return result

    def _classify_relationship_sync(
        self,
        claim: str,
        evidence: str,
    ) -> dict:
        """Synchronous relationship classification."""
        inputs = self.tokenizer(
            evidence,
            claim,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Map probabilities to labels
        id2label = getattr(self.model.config, "id2label", {0: "contradiction", 1: "neutral", 2: "entailment"})

        result = {}
        for idx, label in id2label.items():
            result[label.lower()] = float(probs[int(idx)])

        return result


# Global NLI scorer instance
_nli_scorer: Optional[NLIScorer] = None


@lru_cache(maxsize=1)
def get_nli_scorer() -> NLIScorer:
    """Get or create the global NLI scorer instance."""
    global _nli_scorer
    if _nli_scorer is None:
        _nli_scorer = NLIScorer()
    return _nli_scorer
