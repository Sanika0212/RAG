"""Claim-level validation for response grounding - Novel Contribution #3."""

import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import structlog

from src.config.constants import ClaimStatus
from src.config.settings import get_settings
from src.retrieval.search import SearchResult

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class Claim:
    """An atomic claim extracted from a response."""

    text: str
    status: ClaimStatus
    supporting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    grounding_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ClaimValidationResult:
    """Result of claim-level validation."""

    claims: list[Claim]
    total_claims: int
    grounded_claims: int
    recovered_claims: int
    ungrounded_claims: int
    hallucination_risk: float  # Proportion of ungrounded claims
    revised_response: str
    revision_notes: list[str] = field(default_factory=list)


class ClaimValidator:
    """Validate response claims against retrieved context.

    Novel contribution: Claim-level causal validation:
    1. Extract atomic claims from response
    2. Generate no-context response (counterfactual)
    3. Classify each claim:
       - GROUNDED: Claim supported by retrieved context
       - RECOVERED: Claim from model knowledge, verified via targeted retrieval
       - UNGROUNDED: Claim not supported (potential hallucination)
    4. Rewrite response removing ungrounded claims
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        grounding_threshold: float = settings.claim_grounding_threshold,
    ):
        """Initialize the claim validator.

        Args:
            model: Claude model for claim extraction and validation
            grounding_threshold: Threshold for considering a claim grounded
        """
        self.model_name = model
        self.grounding_threshold = grounding_threshold
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self._searcher = None  # Lazy loaded for targeted retrieval

    async def _get_searcher(self):
        """Lazy load the hybrid searcher for targeted retrieval."""
        if self._searcher is None:
            from src.retrieval.search import HybridSearcher
            self._searcher = HybridSearcher()
        return self._searcher

    async def validate_response(
        self,
        query: str,
        response: str,
        context_chunks: list[SearchResult],
        session=None,
    ) -> dict:
        """Validate all claims in a response.

        Args:
            query: Original user query
            response: Generated response to validate
            context_chunks: Retrieved context chunks
            session: Database session for targeted retrieval (optional)

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating response claims", response_length=len(response))

        # Step 1: Extract atomic claims
        claims = await self._extract_claims(response)

        if not claims:
            return {
                "claims": [],
                "total_claims": 0,
                "grounded_claims": 0,
                "recovered_claims": 0,
                "ungrounded_claims": 0,
                "hallucination_risk": 0.0,
                "revised_response": response,
            }

        # Step 2: Generate counterfactual (no-context) response
        counterfactual_claims = await self._generate_counterfactual(query)

        # Step 3: Validate each claim (with targeted retrieval for recovery)
        context_text = "\n\n".join(c.text for c in context_chunks)
        validated_claims = []

        for claim_text in claims:
            claim = await self._validate_claim(
                claim_text=claim_text,
                context=context_text,
                counterfactual_claims=counterfactual_claims,
                context_chunks=context_chunks,
                session=session,
            )
            validated_claims.append(claim)

        # Step 4: Calculate metrics
        grounded = sum(1 for c in validated_claims if c.status == ClaimStatus.GROUNDED)
        recovered = sum(1 for c in validated_claims if c.status == ClaimStatus.RECOVERED)
        ungrounded = sum(1 for c in validated_claims if c.status == ClaimStatus.UNGROUNDED)
        total = len(validated_claims)

        hallucination_risk = ungrounded / total if total > 0 else 0.0

        # Step 5: Revise response if needed
        revised_response, revision_notes = await self._revise_response(
            original_response=response,
            validated_claims=validated_claims,
        )

        logger.info(
            "Validation complete",
            total=total,
            grounded=grounded,
            recovered=recovered,
            ungrounded=ungrounded,
            hallucination_risk=round(hallucination_risk, 3),
        )

        return {
            "claims": [
                {
                    "text": c.text,
                    "status": c.status.value,
                    "confidence": c.confidence,
                    "supporting_evidence": c.supporting_evidence,
                    "grounding_chunk_ids": c.grounding_chunk_ids,
                }
                for c in validated_claims
            ],
            "total_claims": total,
            "grounded_claims": grounded,
            "recovered_claims": recovered,
            "ungrounded_claims": ungrounded,
            "hallucination_risk": hallucination_risk,
            "revised_response": revised_response,
            "revision_notes": revision_notes,
        }

    async def _extract_claims(self, response: str) -> list[str]:
        """Extract atomic claims from response using Claude."""
        prompt = f"""Extract all atomic factual claims from this response. Each claim should be:
- A single, verifiable statement
- Self-contained (understandable without other claims)
- Not an opinion or subjective statement

Response:
\"\"\"
{response}
\"\"\"

Respond with ONLY a JSON array of claims, no other text:
["claim 1", "claim 2", "claim 3"]

Only include factual claims, not citations or hedging phrases."""

        try:
            api_response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            content = api_response.content[0].text.strip()
            # Handle code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            claims = json.loads(content)
            return claims if isinstance(claims, list) else []

        except Exception as e:
            logger.warning("Claim extraction failed", error=str(e))
            return []

    async def _generate_counterfactual(self, query: str) -> list[str]:
        """Generate response without context (counterfactual baseline) using Claude."""
        prompt = f"""Answer this question briefly using only your general knowledge. Do not make up specific facts.

Question: {query}

Provide a concise answer with only well-established facts."""

        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract claims from counterfactual
            counterfactual_text = response.content[0].text
            claims = await self._extract_claims(counterfactual_text)
            return claims

        except Exception as e:
            logger.warning("Counterfactual generation failed", error=str(e))
            return []

    async def _validate_claim(
        self,
        claim_text: str,
        context: str,
        counterfactual_claims: list[str],
        context_chunks: list[SearchResult],
        session=None,
    ) -> Claim:
        """Validate a single claim against context with targeted retrieval for recovery."""
        # Check if claim is in counterfactual (model's prior knowledge)
        in_counterfactual = any(
            self._claims_match(claim_text, cf_claim)
            for cf_claim in counterfactual_claims
        )

        # Check grounding in context using NLI
        from src.validation.nli import get_nli_scorer
        nli_scorer = get_nli_scorer()

        grounding_results = await nli_scorer.score_grounding(
            claim=claim_text,
            evidence_texts=[c.text for c in context_chunks],
        )

        # Find best grounding score
        best_score = max(grounding_results) if grounding_results else 0.0
        is_grounded = best_score >= self.grounding_threshold

        # Find supporting chunks
        supporting_chunks = []
        supporting_evidence = []
        for i, score in enumerate(grounding_results):
            if score >= self.grounding_threshold and i < len(context_chunks):
                supporting_chunks.append(str(context_chunks[i].chunk_id))
                supporting_evidence.append(context_chunks[i].text[:200])

        # Determine status with targeted retrieval for recovery
        if is_grounded:
            status = ClaimStatus.GROUNDED
            confidence = best_score
        elif in_counterfactual or best_score >= 0.4:
            # Attempt targeted retrieval to find supporting evidence
            recovered, recovery_evidence, recovery_chunk_id = await self._attempt_targeted_retrieval(
                claim_text, session
            )
            if recovered:
                status = ClaimStatus.RECOVERED
                confidence = 0.75  # Moderate confidence for recovered claims
                supporting_chunks.append(recovery_chunk_id)
                supporting_evidence.append(recovery_evidence[:200])
            else:
                status = ClaimStatus.UNGROUNDED
                confidence = best_score
        else:
            status = ClaimStatus.UNGROUNDED
            confidence = best_score

        return Claim(
            text=claim_text,
            status=status,
            supporting_evidence=supporting_evidence,
            confidence=confidence,
            grounding_chunk_ids=supporting_chunks,
        )

    async def _attempt_targeted_retrieval(
        self,
        claim: str,
        session=None,
    ) -> tuple[bool, str, str]:
        """Attempt targeted retrieval to find evidence for an ungrounded claim.

        Args:
            claim: The claim to find evidence for
            session: Database session (optional, will create if needed)

        Returns:
            Tuple of (recovered: bool, evidence_text: str, chunk_id: str)
        """
        try:
            from src.database.connection import get_db
            from src.validation.nli import get_nli_scorer

            searcher = await self._get_searcher()
            nli_scorer = get_nli_scorer()

            # Use the claim itself as a search query
            if session is None:
                async with get_db() as db_session:
                    results = await searcher.search(
                        query=claim,
                        session=db_session,
                        top_k=3,
                    )
            else:
                results = await searcher.search(
                    query=claim,
                    session=session,
                    top_k=3,
                )

            if not results:
                return False, "", ""

            # Check NLI scores for the new results
            evidence_texts = [r.text for r in results]
            scores = await nli_scorer.score_grounding(claim, evidence_texts)

            # Find best scoring result
            if scores:
                best_idx = scores.index(max(scores))
                best_score = scores[best_idx]

                if best_score >= self.grounding_threshold:
                    return True, results[best_idx].text, str(results[best_idx].chunk_id)

            return False, "", ""

        except Exception as e:
            logger.warning("Targeted retrieval failed", error=str(e))
            return False, "", ""

    def _claims_match(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are semantically similar."""
        # Simple word overlap heuristic
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        max_len = max(len(words1), len(words2))

        return overlap / max_len > 0.5

    async def _revise_response(
        self,
        original_response: str,
        validated_claims: list[Claim],
    ) -> tuple[str, list[str]]:
        """Revise response to remove ungrounded claims using Claude."""
        ungrounded = [c for c in validated_claims if c.status == ClaimStatus.UNGROUNDED]

        if not ungrounded:
            return original_response, []

        # Build revision prompt
        ungrounded_list = "\n".join(f"- {c.text}" for c in ungrounded)

        prompt = f"""Revise this response to remove or hedge the following ungrounded claims.

Original Response:
\"\"\"
{original_response}
\"\"\"

Ungrounded Claims to Remove/Hedge:
{ungrounded_list}

Guidelines:
1. Remove statements that cannot be verified
2. Add hedging language for uncertain claims (e.g., "It's possible that...", "Some sources suggest...")
3. Maintain the overall coherence and helpfulness
4. Keep citations that are still relevant
5. Don't add new information

Respond with ONLY the revised response, no preamble or explanation."""

        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            revised = response.content[0].text

            revision_notes = [f"Removed/hedged ungrounded claim: {c.text}" for c in ungrounded]

            return revised, revision_notes

        except Exception as e:
            logger.warning("Response revision failed", error=str(e))
            return original_response, [f"Revision failed: {str(e)}"]
