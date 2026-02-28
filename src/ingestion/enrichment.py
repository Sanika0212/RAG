"""Metadata enrichment using Claude Haiku."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import structlog

from src.config.constants import DifficultyLevel, MEDICAL_TOPIC_VOCABULARY
from src.config.settings import get_settings
from src.ingestion.chunker import TextChunk

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class EnrichedMetadata:
    """Enriched metadata for a chunk."""

    chunk_id: str
    summary: str
    keywords: list[str]
    hypothetical_questions: list[str]
    entity_mentions: list[str]
    topic_tags: list[str]
    difficulty_level: DifficultyLevel
    temporal_references: list[str]
    medical_concepts: list[str] = field(default_factory=list)
    confidence_score: float = 0.0


class MetadataEnricher:
    """Enrich chunk metadata using Claude Haiku."""

    def __init__(
        self,
        model: str = settings.agent_model,
        max_concurrent: int = 5,
        batch_size: int = 10,
    ):
        """Initialize the enricher.

        Args:
            model: Claude model to use for enrichment
            max_concurrent: Maximum concurrent API calls
            batch_size: Number of chunks to process in a batch
        """
        self.model = model
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    async def enrich_chunks(
        self,
        chunks: list[TextChunk],
        document_context: Optional[str] = None,
    ) -> list[EnrichedMetadata]:
        """Enrich multiple chunks with metadata.

        Args:
            chunks: List of chunks to enrich
            document_context: Optional context about the document

        Returns:
            List of EnrichedMetadata objects
        """
        logger.info("Enriching chunks", count=len(chunks))

        # Process in batches with concurrency control
        results = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_results = await asyncio.gather(
                *[
                    self._enrich_chunk(chunk, document_context)
                    for chunk in batch
                ],
                return_exceptions=True,
            )

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Failed to enrich chunk",
                        chunk_id=batch[j].id,
                        error=str(result),
                    )
                    # Create minimal metadata on failure
                    results.append(self._create_fallback_metadata(batch[j]))
                else:
                    results.append(result)

        logger.info("Chunks enriched", count=len(results))
        return results

    async def _enrich_chunk(
        self,
        chunk: TextChunk,
        document_context: Optional[str] = None,
    ) -> EnrichedMetadata:
        """Enrich a single chunk."""
        async with self._semaphore:
            prompt = self._build_enrichment_prompt(chunk, document_context)

            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )

                content = response.content[0].text
                return self._parse_enrichment_response(chunk.id, content)

            except anthropic.APIError as e:
                logger.error("Anthropic API error", error=str(e))
                raise

    def _build_enrichment_prompt(
        self,
        chunk: TextChunk,
        document_context: Optional[str] = None,
    ) -> str:
        """Build the enrichment prompt."""
        context_section = ""
        if document_context:
            context_section = f"\nDocument Context: {document_context}\n"

        section_info = ""
        if chunk.section_path:
            section_info = f"\nSection: {' > '.join(chunk.section_path)}\n"

        topic_vocab = ", ".join(MEDICAL_TOPIC_VOCABULARY[:20])

        return f"""Analyze this medical/scientific text chunk and extract structured metadata.

{context_section}{section_info}
Text:
\"\"\"
{chunk.text}
\"\"\"

Respond with a JSON object containing:
1. "summary": A single sentence summary (max 100 characters)
2. "keywords": 3-7 key terms/concepts
3. "hypothetical_questions": 2-3 questions this chunk could answer
4. "entity_mentions": Named entities (drugs, diseases, genes, organizations, etc.)
5. "topic_tags": 1-3 tags from this vocabulary: {topic_vocab}
6. "difficulty_level": One of "basic", "intermediate", "technical", "expert"
7. "temporal_references": Any dates, time periods, or temporal markers
8. "medical_concepts": Specific medical/scientific concepts (MeSH-like terms)

Respond ONLY with valid JSON, no other text."""

    def _parse_enrichment_response(
        self,
        chunk_id: str,
        response: str,
    ) -> EnrichedMetadata:
        """Parse the LLM response into EnrichedMetadata."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()

            data = json.loads(json_str)

            # Map difficulty level
            difficulty_map = {
                "basic": DifficultyLevel.BASIC,
                "intermediate": DifficultyLevel.INTERMEDIATE,
                "technical": DifficultyLevel.TECHNICAL,
                "expert": DifficultyLevel.EXPERT,
            }
            difficulty = difficulty_map.get(
                data.get("difficulty_level", "").lower(),
                DifficultyLevel.INTERMEDIATE,
            )

            return EnrichedMetadata(
                chunk_id=chunk_id,
                summary=data.get("summary", "")[:500],
                keywords=data.get("keywords", [])[:10],
                hypothetical_questions=data.get("hypothetical_questions", [])[:5],
                entity_mentions=data.get("entity_mentions", [])[:20],
                topic_tags=self._validate_topic_tags(data.get("topic_tags", [])),
                difficulty_level=difficulty,
                temporal_references=data.get("temporal_references", [])[:10],
                medical_concepts=data.get("medical_concepts", [])[:20],
                confidence_score=0.9,  # High confidence for successful parse
            )

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse enrichment response",
                chunk_id=chunk_id,
                error=str(e),
            )
            return self._create_fallback_metadata_from_text(chunk_id, response)

    def _validate_topic_tags(self, tags: list[str]) -> list[str]:
        """Validate topic tags against controlled vocabulary."""
        valid_tags = []
        for tag in tags:
            tag_lower = tag.lower().replace(" ", "_")
            if tag_lower in MEDICAL_TOPIC_VOCABULARY:
                valid_tags.append(tag_lower)
            else:
                # Try to find closest match
                for vocab_tag in MEDICAL_TOPIC_VOCABULARY:
                    if tag_lower in vocab_tag or vocab_tag in tag_lower:
                        valid_tags.append(vocab_tag)
                        break
        return list(set(valid_tags))[:5]

    def _create_fallback_metadata(self, chunk: TextChunk) -> EnrichedMetadata:
        """Create fallback metadata when enrichment fails."""
        # Extract basic keywords from text
        words = chunk.text.lower().split()
        keywords = list(set(w for w in words if len(w) > 5))[:5]

        return EnrichedMetadata(
            chunk_id=chunk.id,
            summary=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
            keywords=keywords,
            hypothetical_questions=[],
            entity_mentions=[],
            topic_tags=[],
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            temporal_references=[],
            medical_concepts=[],
            confidence_score=0.3,
        )

    def _create_fallback_metadata_from_text(
        self,
        chunk_id: str,
        response: str,
    ) -> EnrichedMetadata:
        """Create metadata from partially parsed response."""
        # Try to extract any useful information from the response
        return EnrichedMetadata(
            chunk_id=chunk_id,
            summary=response[:100] if response else "",
            keywords=[],
            hypothetical_questions=[],
            entity_mentions=[],
            topic_tags=[],
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            temporal_references=[],
            medical_concepts=[],
            confidence_score=0.2,
        )


async def generate_hypothetical_questions_embeddings(
    enriched_metadata: list[EnrichedMetadata],
    embedder,
) -> list[EnrichedMetadata]:
    """Generate embeddings for hypothetical questions.

    Args:
        enriched_metadata: List of enriched metadata
        embedder: BGE embedder instance

    Returns:
        Metadata with HQ embeddings added
    """
    for metadata in enriched_metadata:
        if metadata.hypothetical_questions:
            embeddings = await embedder.embed_texts(metadata.hypothetical_questions)
            # Store as list of lists for JSON serialization
            metadata.hq_embeddings = [
                emb.dense.tolist() for emb in embeddings
            ]
    return enriched_metadata
