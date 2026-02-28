"""BGE-M3 embedding service with support for dense and sparse embeddings."""

import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import structlog
import torch
from FlagEmbedding import BGEM3FlagModel

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class EmbeddingCache:
    """LRU cache for embeddings with text hash keys."""

    def __init__(self, max_size: int = 10000):
        """Initialize the embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, "EmbeddingResult"] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Create a hash key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> Optional["EmbeddingResult"]:
        """Get embedding from cache."""
        key = self._hash_text(text)
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: "EmbeddingResult") -> None:
        """Put embedding in cache."""
        key = self._hash_text(text)

        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                # Remove oldest item
                self._cache.popitem(last=False)
            self._cache[key] = embedding

    def get_many(self, texts: list[str]) -> tuple[list[Optional["EmbeddingResult"]], list[int]]:
        """Get multiple embeddings from cache.

        Returns:
            Tuple of (results, miss_indices) where results has None for misses
            and miss_indices contains indices that need to be computed.
        """
        results = []
        miss_indices = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            results.append(cached)
            if cached is None:
                miss_indices.append(i)

        return results, miss_indices

    def put_many(self, texts: list[str], embeddings: list["EmbeddingResult"]) -> None:
        """Put multiple embeddings in cache."""
        for text, emb in zip(texts, embeddings):
            self.put(text, emb)

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    dense: np.ndarray  # Dense embedding vector
    sparse: Optional[dict[int, float]] = None  # Sparse embedding (token_id -> weight)
    colbert: Optional[np.ndarray] = None  # ColBERT token embeddings


class BGEEmbedder:
    """BGE-M3 embedding service with batch processing support and caching."""

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str = settings.embedding_device,
        normalize: bool = settings.embedding_normalize,
        batch_size: int = settings.embedding_batch_size,
        cache_size: int = 10000,
    ):
        """Initialize the BGE-M3 embedder.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-m3)
            device: Device to run model on (cpu, cuda, mps)
            normalize: Whether to normalize embeddings
            batch_size: Batch size for encoding
            cache_size: Maximum number of embeddings to cache
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.normalize = normalize
        self.batch_size = batch_size
        self._model: Optional[BGEM3FlagModel] = None
        self._lock = asyncio.Lock()
        self._cache = EmbeddingCache(max_size=cache_size)

    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self) -> BGEM3FlagModel:
        """Lazy load the model."""
        if self._model is None:
            logger.info(
                "Loading BGE-M3 model",
                model=self.model_name,
                device=self.device,
            )
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=(self.device == "cuda"),
                device=self.device,
            )
            logger.info("BGE-M3 model loaded successfully")
        return self._model

    async def embed_texts(
        self,
        texts: list[str],
        return_sparse: bool = False,
        return_colbert: bool = False,
        use_cache: bool = True,
    ) -> list[EmbeddingResult]:
        """Embed a list of texts asynchronously with caching.

        Args:
            texts: List of texts to embed
            return_sparse: Whether to return sparse embeddings
            return_colbert: Whether to return ColBERT embeddings
            use_cache: Whether to use embedding cache (default True)

        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []

        # Check cache first (only for dense-only embeddings)
        if use_cache and not return_sparse and not return_colbert:
            cached_results, miss_indices = self._cache.get_many(texts)

            if not miss_indices:
                # All texts were cached
                logger.debug("All embeddings from cache", count=len(texts))
                return cached_results

            # Only compute missing embeddings
            texts_to_embed = [texts[i] for i in miss_indices]

            async with self._lock:
                loop = asyncio.get_event_loop()
                new_results = await loop.run_in_executor(
                    None,
                    lambda: self._embed_texts_sync(texts_to_embed, return_sparse, return_colbert),
                )

            # Cache new results
            self._cache.put_many(texts_to_embed, new_results)

            # Merge cached and new results
            final_results = list(cached_results)
            for i, idx in enumerate(miss_indices):
                final_results[idx] = new_results[i]

            logger.debug(
                "Embeddings computed",
                total=len(texts),
                cached=len(texts) - len(miss_indices),
                computed=len(miss_indices),
            )
            return final_results

        # No caching for sparse/colbert embeddings
        async with self._lock:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._embed_texts_sync(texts, return_sparse, return_colbert),
            )
        return results

    @property
    def cache_stats(self) -> dict:
        """Get embedding cache statistics."""
        return self._cache.stats

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def _embed_texts_sync(
        self,
        texts: list[str],
        return_sparse: bool = False,
        return_colbert: bool = False,
    ) -> list[EmbeddingResult]:
        """Synchronous embedding for batch processing."""
        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Get embeddings from model
            output = self.model.encode(
                batch,
                batch_size=self.batch_size,
                max_length=8192,  # BGE-M3 supports up to 8192 tokens
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert,
            )

            # Process each item in batch
            for j in range(len(batch)):
                dense = output["dense_vecs"][j]
                if self.normalize:
                    dense = dense / np.linalg.norm(dense)

                sparse = None
                if return_sparse and "lexical_weights" in output:
                    sparse = output["lexical_weights"][j]

                colbert = None
                if return_colbert and "colbert_vecs" in output:
                    colbert = output["colbert_vecs"][j]

                results.append(EmbeddingResult(
                    dense=dense,
                    sparse=sparse,
                    colbert=colbert,
                ))

        logger.debug("Embedded texts", count=len(texts))
        return results

    async def embed_query(
        self,
        query: str,
        return_sparse: bool = True,
    ) -> EmbeddingResult:
        """Embed a single query with sparse embedding for hybrid search.

        Args:
            query: Query text to embed
            return_sparse: Whether to return sparse embedding (default True for queries)

        Returns:
            EmbeddingResult with dense and optionally sparse embedding
        """
        results = await self.embed_texts([query], return_sparse=return_sparse)
        return results[0]

    async def embed_documents(
        self,
        documents: list[str],
    ) -> list[EmbeddingResult]:
        """Embed documents (chunks) with dense embeddings only.

        Args:
            documents: List of document texts to embed

        Returns:
            List of EmbeddingResult objects with dense embeddings
        """
        return await self.embed_texts(documents, return_sparse=False, return_colbert=False)

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: list[np.ndarray],
    ) -> list[float]:
        """Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors

        Returns:
            List of similarity scores
        """
        if not document_embeddings:
            return []

        # Stack document embeddings
        doc_matrix = np.stack(document_embeddings)

        # Normalize if not already
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities.tolist()

    def compute_sparse_similarity(
        self,
        query_sparse: dict[int, float],
        document_sparse: dict[int, float],
    ) -> float:
        """Compute sparse embedding similarity using lexical matching.

        Args:
            query_sparse: Query sparse embedding (token_id -> weight)
            document_sparse: Document sparse embedding (token_id -> weight)

        Returns:
            Similarity score
        """
        if not query_sparse or not document_sparse:
            return 0.0

        # Find common tokens and compute weighted overlap
        common_tokens = set(query_sparse.keys()) & set(document_sparse.keys())

        if not common_tokens:
            return 0.0

        score = sum(
            query_sparse[token] * document_sparse[token]
            for token in common_tokens
        )
        return score


# Global embedder instance
_embedder: Optional[BGEEmbedder] = None


@lru_cache(maxsize=1)
def get_embedder() -> BGEEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder


async def preload_model() -> None:
    """Preload the embedding model at startup."""
    embedder = get_embedder()
    # Trigger model loading by embedding a test text
    await embedder.embed_texts(["warmup"])
    logger.info("Embedding model preloaded")
