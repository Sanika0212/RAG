"""Tests for hierarchical chunking."""

import pytest

from src.config.constants import ChunkType
from src.ingestion.parser import ParsedDocument, ParsedElement
from src.ingestion.chunker import HierarchicalChunker, TextChunk


@pytest.fixture
def sample_elements():
    """Create sample parsed elements."""
    return [
        ParsedElement(
            text="Introduction to Medical RAG",
            element_type=ChunkType.HEADING,
            heading_level=1,
        ),
        ParsedElement(
            text="This document covers the fundamentals of retrieval-augmented generation for medical applications. RAG systems combine information retrieval with language models to provide accurate, grounded responses.",
            element_type=ChunkType.PARAGRAPH,
            parent_heading="Introduction to Medical RAG",
        ),
        ParsedElement(
            text="Key Concepts",
            element_type=ChunkType.HEADING,
            heading_level=2,
        ),
        ParsedElement(
            text="Embedding models convert text into dense vector representations that capture semantic meaning. These vectors enable similarity search to find relevant documents.",
            element_type=ChunkType.PARAGRAPH,
            parent_heading="Key Concepts",
        ),
        ParsedElement(
            text="Retrieval strategies include dense retrieval, sparse retrieval, and hybrid approaches that combine both methods for optimal performance.",
            element_type=ChunkType.PARAGRAPH,
            parent_heading="Key Concepts",
        ),
    ]


@pytest.fixture
def sample_document(sample_elements):
    """Create a sample parsed document."""
    from src.config.constants import DocumentType

    return ParsedDocument(
        filename="test_doc.md",
        doc_type=DocumentType.MARKDOWN,
        elements=sample_elements,
        title="Introduction to Medical RAG",
        raw_text="\n\n".join(e.text for e in sample_elements),
        file_hash="abc123",
    )


def test_chunker_initialization():
    """Test chunker initialization with custom parameters."""
    chunker = HierarchicalChunker(
        chunk_size=256,
        chunk_overlap=25,
        min_chunk_size=50,
    )

    assert chunker.chunk_size == 256
    assert chunker.chunk_overlap == 25
    assert chunker.min_chunk_size == 50


def test_chunk_document(sample_document):
    """Test basic document chunking."""
    chunker = HierarchicalChunker(chunk_size=100, min_chunk_size=20)
    chunks = chunker.chunk_document(sample_document)

    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_chunk_indices_are_sequential(sample_document):
    """Test that chunk indices are sequential."""
    chunker = HierarchicalChunker(chunk_size=100, min_chunk_size=20)
    chunks = chunker.chunk_document(sample_document)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_section_path_preserved(sample_document):
    """Test that section hierarchy is preserved."""
    chunker = HierarchicalChunker(chunk_size=200, min_chunk_size=20)
    chunks = chunker.chunk_document(sample_document)

    # Should have chunks with section paths
    chunks_with_paths = [c for c in chunks if c.section_path]
    assert len(chunks_with_paths) > 0


def test_deduplication():
    """Test that near-duplicate chunks are merged."""
    chunker = HierarchicalChunker(similarity_threshold=0.9, chunk_size=15, min_chunk_size=1)

    # Create document with duplicate content
    elements = [
        ParsedElement(
            text="This is the exact same content repeated verbatim. And here is more content.",
            element_type=ChunkType.PARAGRAPH,
        ),
        ParsedElement(
            text="This is the exact same content repeated verbatim. And here is more content.",
            element_type=ChunkType.PARAGRAPH,
        ),
    ]

    from src.config.constants import DocumentType
    doc = ParsedDocument(
        filename="dupe.txt",
        doc_type=DocumentType.TEXT,
        elements=elements,
        raw_text="",
        file_hash="dupe123",
    )

    chunks = chunker.chunk_document(doc)

    # Should deduplicate to 1 chunk
    assert len(chunks) == 1


def test_large_element_splitting():
    """Test that large elements are split correctly."""
    chunker = HierarchicalChunker(
        chunk_size=50,  # Very small for testing
        max_chunk_size=100,
        min_chunk_size=10,
    )

    # Create a large paragraph with distinct words and punctuation so it can be split into sentences and won't be deduplicated
    large_text = " ".join([f"word_{i}." for i in range(200)])  # ~200 tokens

    elements = [
        ParsedElement(
            text=large_text,
            element_type=ChunkType.PARAGRAPH,
        ),
    ]

    from src.config.constants import DocumentType
    doc = ParsedDocument(
        filename="large.txt",
        doc_type=DocumentType.TEXT,
        elements=elements,
        raw_text=large_text,
        file_hash="large123",
    )

    chunks = chunker.chunk_document(doc)

    # Should be split into multiple chunks
    assert len(chunks) > 1


def test_token_count_accuracy(sample_document):
    """Test that token counts are accurate."""
    chunker = HierarchicalChunker()
    chunks = chunker.chunk_document(sample_document)

    for chunk in chunks:
        # Verify token count is reasonable
        assert chunk.token_count > 0
        # Token count should be roughly proportional to text length
        # (not exact due to tokenization)
        assert chunk.token_count < len(chunk.text)  # Tokens < characters
