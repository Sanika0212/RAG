"""Confidence-conditioned response generation using Claude with resilience."""

from dataclasses import dataclass
from typing import Optional

import anthropic
import structlog

from src.config.constants import ConfidenceBand, HEDGING_PHRASES
from src.config.settings import get_settings
from src.config.circuit_breaker import get_resilient_llm_client, CircuitOpenError
from src.retrieval.search import SearchResult

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class Citation:
    """A citation reference."""

    chunk_id: str
    document_title: str
    text_snippet: str
    relevance_score: float


class ResponseGenerator:
    """Generate responses conditioned on confidence level using Claude.

    - HIGH confidence: Assertive responses with inline citations
    - MEDIUM confidence: Hedged language with appropriate caveats
    - LOW confidence: Handled by abstention in the agent graph
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = settings.max_tokens,
        temperature: float = settings.temperature,
    ):
        """Initialize the response generator.

        Args:
            model: Claude model for generation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._resilient_client = get_resilient_llm_client()

    async def generate(
        self,
        query: str,
        context_chunks: list[SearchResult],
        confidence_band: ConfidenceBand,
        additional_context: Optional[str] = None,
    ) -> tuple[str, list[dict]]:
        """Generate a response based on query and context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            confidence_band: Confidence level (HIGH or MEDIUM)
            additional_context: Optional additional context

        Returns:
            Tuple of (response_text, citations)
        """
        logger.info(
            "Generating response with Claude",
            query=query[:50],
            chunks=len(context_chunks),
            confidence=confidence_band.value,
        )

        # Build context string with numbered citations
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            doc_title = chunk.document_title or "Unknown Document"
            context_parts.append(f"[{i}] Source: {doc_title}\n{chunk.text}")

        context_string = "\n\n".join(context_parts)

        # Select prompt based on confidence
        if confidence_band == ConfidenceBand.HIGH:
            system_prompt = self._get_high_confidence_prompt()
        else:
            system_prompt = self._get_medium_confidence_prompt()

        # Add any additional context
        if additional_context:
            context_string = f"{additional_context}\n\n{context_string}"

        user_message = f"""Context:
{context_string}

Question: {query}

Provide a well-structured answer using the context above. Include citation numbers [1], [2], etc. when referencing sources."""

        try:
            response = await self._resilient_client.create_message(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                fallback_response="I'm temporarily unable to generate a response. Please try again in a moment.",
            )

            response_text = response.content[0].text

            # Build citations list
            citations = []
            for i, chunk in enumerate(context_chunks, 1):
                citations.append({
                    "index": i,
                    "chunk_id": str(chunk.chunk_id),
                    "document_title": chunk.document_title or "Unknown",
                    "text_snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "relevance_score": chunk.score,
                })

            logger.info(
                "Response generated",
                response_length=len(response_text),
                citations=len(citations),
            )

            return response_text, citations

        except Exception as e:
            logger.error("Generation failed", error=str(e))
            raise

    def _get_high_confidence_prompt(self) -> str:
        """Get system prompt for high-confidence responses."""
        return """You are a knowledgeable assistant providing accurate, well-sourced answers.

Guidelines:
1. Provide direct, confident answers based on the given context
2. Use inline citations [1], [2], etc. to reference sources
3. Be comprehensive but concise
4. If the context contains conflicting information, note the discrepancy
5. Structure your response clearly with paragraphs or bullet points as appropriate"""

    def _get_medium_confidence_prompt(self) -> str:
        """Get system prompt for medium-confidence (hedged) responses."""
        hedging_examples = ", ".join(f'"{p}"' for p in list(HEDGING_PHRASES)[:5])

        return f"""You are a careful assistant providing balanced answers with appropriate epistemic caution.

Guidelines:
1. Use hedging language to indicate uncertainty: {hedging_examples}
2. Distinguish between what the sources explicitly state vs. what you're inferring
3. Include inline citations [1], [2], etc.
4. If information seems incomplete, acknowledge the limitation
5. Present multiple perspectives if the sources suggest different views
6. Be helpful while being honest about the limits of the available information"""

    async def generate_abstention(
        self,
        query: str,
        reason: str,
    ) -> str:
        """Generate an abstention response when confidence is too low.

        Args:
            query: Original user query
            reason: Reason for abstaining

        Returns:
            Polite abstention message
        """
        logger.info("Generating abstention", query=query[:50], reason=reason)

        prompt = f"""The user asked: "{query}"

However, I need to abstain from answering because: {reason}

Generate a polite, helpful response that:
1. Acknowledges the user's question
2. Explains why a reliable answer cannot be provided
3. Suggests what additional information or context might help
4. Offers to help in other ways if possible

Keep the response concise and empathetic."""

        try:
            response = await self._resilient_client.create_message(
                model=self.model_name,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                fallback_response=f"I apologize, but I cannot provide a reliable answer to your question about '{query}' at this time. {reason}",
            )

            return response.content[0].text

        except Exception as e:
            logger.error("Abstention generation failed", error=str(e))
            return f"I apologize, but I cannot provide a reliable answer to your question about '{query}' at this time. {reason}"
