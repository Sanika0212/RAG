"""Query filter extraction using Claude Haiku."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import anthropic
import structlog

from src.config.constants import MEDICAL_TOPIC_VOCABULARY, DifficultyLevel
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class QueryFilters:
    """Extracted filters from a natural language query."""

    topic_tags: list[str] = field(default_factory=list)
    difficulty_level: Optional[DifficultyLevel] = None
    entity_mentions: list[str] = field(default_factory=list)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    document_types: list[str] = field(default_factory=list)
    keywords_must_include: list[str] = field(default_factory=list)
    keywords_must_exclude: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return (
            not self.topic_tags
            and self.difficulty_level is None
            and not self.entity_mentions
            and self.date_from is None
            and self.date_to is None
            and not self.document_types
            and not self.keywords_must_include
            and not self.keywords_must_exclude
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for database queries."""
        result = {}
        if self.topic_tags:
            result["topic_tags"] = self.topic_tags
        if self.difficulty_level:
            result["difficulty_level"] = self.difficulty_level.value
        if self.entity_mentions:
            result["entity_mentions"] = self.entity_mentions
        if self.date_from:
            result["date_from"] = self.date_from
        if self.date_to:
            result["date_to"] = self.date_to
        if self.document_types:
            result["document_types"] = self.document_types
        if self.keywords_must_include:
            result["keywords_must_include"] = self.keywords_must_include
        if self.keywords_must_exclude:
            result["keywords_must_exclude"] = self.keywords_must_exclude
        return result


class QueryFilterExtractor:
    """Extract structured filters from natural language queries."""

    def __init__(
        self,
        model: str = settings.agent_model,
        use_llm: bool = True,
    ):
        """Initialize the filter extractor.

        Args:
            model: Claude model to use
            use_llm: Whether to use LLM for extraction (vs rule-based only)
        """
        self.model = model
        self.use_llm = use_llm
        self._client: Optional[anthropic.AsyncAnthropic] = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    async def extract_filters(self, query: str) -> QueryFilters:
        """Extract filters from a query.

        Args:
            query: Natural language query

        Returns:
            QueryFilters with extracted structured filters
        """
        # First, try rule-based extraction
        filters = self._rule_based_extraction(query)

        # If LLM is enabled and query seems complex, use LLM
        if self.use_llm and self._needs_llm_extraction(query):
            try:
                llm_filters = await self._llm_extraction(query)
                filters = self._merge_filters(filters, llm_filters)
            except Exception as e:
                logger.warning("LLM filter extraction failed", error=str(e))

        return filters

    def _rule_based_extraction(self, query: str) -> QueryFilters:
        """Extract filters using rule-based patterns."""
        query_lower = query.lower()
        filters = QueryFilters()

        # Extract topic tags
        for topic in MEDICAL_TOPIC_VOCABULARY:
            # Check for topic mention (with word boundaries)
            topic_pattern = topic.replace("_", " ")
            if topic_pattern in query_lower or topic in query_lower:
                filters.topic_tags.append(topic)

        # Extract difficulty indicators
        if any(word in query_lower for word in ["basic", "beginner", "introduction", "simple"]):
            filters.difficulty_level = DifficultyLevel.BASIC
        elif any(word in query_lower for word in ["advanced", "expert", "specialist"]):
            filters.difficulty_level = DifficultyLevel.EXPERT
        elif any(word in query_lower for word in ["technical", "detailed", "mechanism"]):
            filters.difficulty_level = DifficultyLevel.TECHNICAL

        # Extract date patterns (simple year extraction)
        import re
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        if years:
            # If single year, assume it's a range around that year
            year = int(years[0])
            filters.date_from = datetime(year, 1, 1)
            filters.date_to = datetime(year, 12, 31)

        # Extract document type preferences
        if any(word in query_lower for word in ["study", "research", "trial"]):
            filters.document_types.append("research")
        if any(word in query_lower for word in ["guideline", "protocol", "recommendation"]):
            filters.document_types.append("guideline")
        if any(word in query_lower for word in ["review", "meta-analysis", "systematic"]):
            filters.document_types.append("review")

        # Extract exclusion keywords
        if "not" in query_lower or "except" in query_lower or "exclude" in query_lower:
            # Simple pattern: "not X" or "except X"
            not_pattern = r'(?:not|except|exclude)\s+(\w+)'
            matches = re.findall(not_pattern, query_lower)
            filters.keywords_must_exclude.extend(matches)

        return filters

    def _needs_llm_extraction(self, query: str) -> bool:
        """Determine if query needs LLM for filter extraction."""
        # Use LLM for complex queries
        indicators = [
            len(query.split()) > 10,  # Long queries
            "?" in query,  # Questions
            any(word in query.lower() for word in ["compare", "versus", "vs", "difference"]),
            any(word in query.lower() for word in ["recent", "latest", "new"]),
        ]
        return any(indicators)

    async def _llm_extraction(self, query: str) -> QueryFilters:
        """Extract filters using Claude Haiku."""
        topic_vocab = ", ".join(MEDICAL_TOPIC_VOCABULARY[:30])

        prompt = f"""Extract search filters from this medical/scientific query.

Query: "{query}"

Valid topic tags: {topic_vocab}
Valid difficulty levels: basic, intermediate, technical, expert

Respond with JSON only:
{{
    "topic_tags": ["tag1", "tag2"],
    "difficulty_level": "basic|intermediate|technical|expert" or null,
    "entity_mentions": ["drug names", "disease names", "gene names"],
    "date_range": {{"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}} or null,
    "keywords_must_include": ["required", "terms"],
    "keywords_must_exclude": ["excluded", "terms"]
}}

Only include fields that are clearly indicated by the query. Respond with valid JSON only."""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        return self._parse_llm_response(content)

    def _parse_llm_response(self, response: str) -> QueryFilters:
        """Parse LLM response into QueryFilters."""
        try:
            # Extract JSON from response
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()

            data = json.loads(json_str)

            filters = QueryFilters()

            if data.get("topic_tags"):
                filters.topic_tags = [
                    t for t in data["topic_tags"]
                    if t.lower().replace(" ", "_") in MEDICAL_TOPIC_VOCABULARY
                ]

            if data.get("difficulty_level"):
                try:
                    filters.difficulty_level = DifficultyLevel(data["difficulty_level"])
                except ValueError:
                    pass

            if data.get("entity_mentions"):
                filters.entity_mentions = data["entity_mentions"][:10]

            if data.get("date_range"):
                dr = data["date_range"]
                if dr.get("from"):
                    try:
                        filters.date_from = datetime.fromisoformat(dr["from"])
                    except ValueError:
                        pass
                if dr.get("to"):
                    try:
                        filters.date_to = datetime.fromisoformat(dr["to"])
                    except ValueError:
                        pass

            if data.get("keywords_must_include"):
                filters.keywords_must_include = data["keywords_must_include"][:5]

            if data.get("keywords_must_exclude"):
                filters.keywords_must_exclude = data["keywords_must_exclude"][:5]

            return filters

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM filter response")
            return QueryFilters()

    def _merge_filters(
        self,
        rule_based: QueryFilters,
        llm_based: QueryFilters,
    ) -> QueryFilters:
        """Merge rule-based and LLM-based filters."""
        # Combine lists, removing duplicates
        topic_tags = list(set(rule_based.topic_tags + llm_based.topic_tags))
        entity_mentions = list(set(rule_based.entity_mentions + llm_based.entity_mentions))
        keywords_must_include = list(set(
            rule_based.keywords_must_include + llm_based.keywords_must_include
        ))
        keywords_must_exclude = list(set(
            rule_based.keywords_must_exclude + llm_based.keywords_must_exclude
        ))

        # Prefer LLM for difficulty (more nuanced)
        difficulty_level = llm_based.difficulty_level or rule_based.difficulty_level

        # Use LLM dates if available, otherwise rule-based
        date_from = llm_based.date_from or rule_based.date_from
        date_to = llm_based.date_to or rule_based.date_to

        return QueryFilters(
            topic_tags=topic_tags,
            difficulty_level=difficulty_level,
            entity_mentions=entity_mentions,
            date_from=date_from,
            date_to=date_to,
            document_types=rule_based.document_types,  # Rule-based only
            keywords_must_include=keywords_must_include,
            keywords_must_exclude=keywords_must_exclude,
        )
