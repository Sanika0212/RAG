"""Constants and enumerations for the RAG system."""

from enum import Enum, auto


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"
    TEXT = "txt"
    HTML = "html"


class ChunkType(str, Enum):
    """Types of content chunks."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    FIGURE_CAPTION = "figure_caption"


class DifficultyLevel(str, Enum):
    """Content difficulty levels for medical content."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    TECHNICAL = "technical"
    EXPERT = "expert"


class ConfidenceBand(str, Enum):
    """Confidence bands for retrieval quality."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureMode(str, Enum):
    """Retrieval failure modes for diagnosis."""

    AMBIGUITY = "ambiguity"  # Query is ambiguous, needs decomposition
    VOCAB_MISMATCH = "vocab_mismatch"  # Query terms don't match document vocabulary
    INFO_SCATTER = "info_scatter"  # Information spread across multiple chunks
    KNOWLEDGE_GAP = "knowledge_gap"  # Information not in knowledge base
    GRANULARITY_MISMATCH = "granularity_mismatch"  # Query at wrong abstraction level


class QueryType(str, Enum):
    """Types of queries for planning."""

    SIMPLE = "simple"  # Single fact lookup
    COMPOUND = "compound"  # Multiple related facts
    COMPARATIVE = "comparative"  # Compare two or more entities
    TEMPORAL = "temporal"  # Time-based queries
    CAUSAL = "causal"  # Cause-effect relationships
    PROCEDURAL = "procedural"  # How-to queries


class ClaimStatus(str, Enum):
    """Claim validation status."""

    GROUNDED = "grounded"  # Claim supported by retrieved context
    RECOVERED = "recovered"  # Claim from model knowledge, verified
    UNGROUNDED = "ungrounded"  # Claim not supported, potential hallucination


class AgentState(str, Enum):
    """States in the RAG agent state machine."""

    RETRIEVE = "retrieve"
    ESTIMATE_CONFIDENCE = "estimate_confidence"
    ROUTE = "route"
    GENERATE = "generate"
    GENERATE_HEDGED = "generate_hedged"
    DIAGNOSE = "diagnose"
    CORRECT = "correct"
    VALIDATE = "validate"
    FINAL = "final"
    ABSTAIN = "abstain"


# Medical domain specific constants
MEDICAL_TOPIC_VOCABULARY = [
    "anatomy",
    "physiology",
    "pathology",
    "pharmacology",
    "microbiology",
    "immunology",
    "genetics",
    "biochemistry",
    "cardiology",
    "neurology",
    "oncology",
    "pediatrics",
    "surgery",
    "radiology",
    "psychiatry",
    "dermatology",
    "endocrinology",
    "gastroenterology",
    "nephrology",
    "pulmonology",
    "rheumatology",
    "hematology",
    "infectious_disease",
    "emergency_medicine",
    "internal_medicine",
    "family_medicine",
    "obstetrics_gynecology",
    "orthopedics",
    "ophthalmology",
    "otolaryngology",
    "urology",
    "anesthesiology",
    "clinical_trials",
    "epidemiology",
    "public_health",
    "diagnostics",
    "therapeutics",
    "prevention",
    "treatment",
    "symptoms",
    "diagnosis",
    "prognosis",
]

# Hedging phrases for uncertain responses
HEDGING_PHRASES = [
    "Based on the available information",
    "The retrieved documents suggest",
    "According to the sources found",
    "It appears that",
    "The evidence indicates",
    "While not definitive",
    "The information available suggests",
]

# Abstention phrases
ABSTENTION_PHRASES = [
    "I cannot provide a reliable answer to this question based on the available documents.",
    "The retrieved information is insufficient to answer this question accurately.",
    "I don't have enough confident information to answer this question.",
    "This question appears to be outside the scope of the available knowledge base.",
]

# Token limits
MAX_CONTEXT_TOKENS = 8000  # Max tokens for context in generation
MAX_RESPONSE_TOKENS = 2000  # Max tokens in generated response
MAX_QUERY_TOKENS = 500  # Max tokens in user query

# Similarity thresholds
DUPLICATE_CHUNK_THRESHOLD = 0.90  # Threshold for considering chunks as duplicates
MIN_RERANK_SCORE = 0.3  # Minimum reranker score to include in results
