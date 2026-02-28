"""Application settings and configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Self-Healing RAG Engine"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Database (Neon PostgreSQL)
    database_url: SecretStr = Field(
        default=SecretStr("postgresql+asyncpg://user:password@localhost:5432/rag_db"),
        description="PostgreSQL connection string with asyncpg driver",
    )
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    # Vector dimensions
    embedding_dim: int = 1024  # BGE-M3 dense embedding dimension
    sparse_embedding_dim: int = 250002  # BGE-M3 sparse vocabulary size

    # Embeddings (BGE-M3)
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # or "cuda", "mps"
    embedding_normalize: bool = True

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 10

    # LLM - Claude (for agents: diagnosis, correction, confidence)
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key for Claude",
    )
    agent_model: str = "claude-haiku-4-5-20251001"

    # LLM - Gemini (for generation: responses, claim validation)
    google_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google API key for Gemini",
    )
    generation_model: str = "gemini-2.0-flash"  # Fast and capable
    validation_model: str = "gemini-2.0-flash"  # For claim extraction

    max_tokens: int = 4096
    temperature: float = 0.1

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    similarity_threshold: float = 0.0  # Set to 0 to let reranker decide
    hybrid_search_weights: dict[str, float] = Field(
        default={"vector": 0.4, "keyword": 0.2, "hyde": 0.4},
        description="Weights for hybrid search fusion",
    )

    # Chunking
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens
    min_chunk_size: int = 100  # tokens
    max_chunk_size: int = 1024  # tokens

    # Confidence thresholds
    confidence_high_threshold: float = 0.75
    confidence_low_threshold: float = 0.45
    max_correction_loops: int = 2

    # Validation
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    claim_grounding_threshold: float = 0.7

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL for Alembic migrations."""
        url = self.database_url.get_secret_value()
        return url.replace("postgresql+asyncpg://", "postgresql://")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
