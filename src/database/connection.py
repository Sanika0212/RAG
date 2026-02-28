"""Database connection management for Neon PostgreSQL with retry logic."""

import contextlib
from typing import AsyncGenerator

import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.config.settings import get_settings
from src.database.models import Base

logger = structlog.get_logger(__name__)

settings = get_settings()

# Retry configuration for database operations
DB_RETRY_ATTEMPTS = 3
DB_RETRY_MIN_WAIT = 1  # seconds
DB_RETRY_MAX_WAIT = 10  # seconds

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    OperationalError,
    InterfaceError,
    DBAPIError,
    asyncpg.InterfaceError,
    asyncpg.PostgresConnectionError,
    asyncpg.TooManyConnectionsError,
    ConnectionRefusedError,
    TimeoutError,
)

# Global engine and session factory
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url.get_secret_value(),
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_pre_ping=True,  # Enable connection health checks
            echo=settings.debug,  # Log SQL statements in debug mode
        )
        logger.info("Database engine created", pool_size=settings.db_pool_size)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


async def init_db() -> None:
    """Initialize the database, creating tables and extensions."""
    engine = get_engine()

    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension enabled")

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

        # Create HNSW index for vector similarity search (if not exists)
        # Using cosine distance for normalized embeddings
        await conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                ON chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        )
        logger.info("HNSW index created for chunk embeddings")

        # Create full-text search configuration for chunks
        await conn.execute(
            text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_attribute
                        WHERE attrelid = 'chunks'::regclass
                        AND attname = 'search_vector'
                    ) THEN
                        ALTER TABLE chunks ADD COLUMN search_vector tsvector;
                    END IF;
                END $$;
            """)
        )

        # Create GIN index for full-text search
        await conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
                ON chunks
                USING gin (search_vector)
            """)
        )

        # Create trigger to auto-update search_vector
        await conn.execute(
            text("""
                CREATE OR REPLACE FUNCTION chunks_search_vector_update() RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector := to_tsvector('english', COALESCE(NEW.text, ''));
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql;
            """)
        )

        await conn.execute(
            text("DROP TRIGGER IF EXISTS chunks_search_vector_trigger ON chunks")
        )
        await conn.execute(
            text("""
                CREATE TRIGGER chunks_search_vector_trigger
                BEFORE INSERT OR UPDATE OF text ON chunks
                FOR EACH ROW EXECUTE FUNCTION chunks_search_vector_update()
            """)
        )
        logger.info("Full-text search configuration created")


async def close_db() -> None:
    """Close database connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


def db_retry():
    """Create a retry decorator for database operations."""
    return retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(DB_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=DB_RETRY_MIN_WAIT, max=DB_RETRY_MAX_WAIT),
        before_sleep=before_sleep_log(logger, log_level=20),  # INFO level
        reraise=True,
    )


@contextlib.asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session as a context manager with retry logic."""
    session_factory = get_session_factory()

    @db_retry()
    async def _get_session():
        return session_factory()

    session = await _get_session()
    try:
        yield session
        await session.commit()
    except RETRYABLE_EXCEPTIONS as e:
        await session.rollback()
        logger.warning("Database operation failed, may retry", error=str(e))
        raise
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def execute_with_retry(session: AsyncSession, statement):
    """Execute a database statement with retry logic.

    Use this for critical operations that should be retried on transient failures.
    """
    @db_retry()
    async def _execute():
        return await session.execute(statement)

    return await _execute()


class DatabaseSession:
    """Database session dependency for FastAPI."""

    async def __call__(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a database session."""
        async with get_db() as session:
            yield session


async def check_db_health() -> dict:
    """Check database health and return status."""
    try:
        async with get_db() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()

            # Check pgvector extension
            result = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            vector_version = result.scalar()

            # Get table counts
            result = await session.execute(
                text("SELECT COUNT(*) FROM documents WHERE is_active = true")
            )
            doc_count = result.scalar()

            result = await session.execute(text("SELECT COUNT(*) FROM chunks"))
            chunk_count = result.scalar()

            return {
                "status": "healthy",
                "pgvector_version": vector_version,
                "documents": doc_count,
                "chunks": chunk_count,
            }
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
        }
