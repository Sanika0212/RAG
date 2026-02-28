"""Hierarchical text chunker with overlap-aware deduplication."""

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import structlog
import tiktoken

from src.config.constants import ChunkType
from src.config.settings import get_settings
from src.ingestion.parser import ParsedDocument, ParsedElement

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class TextChunk:
    """A text chunk with hierarchical context."""

    id: str
    text: str
    token_count: int
    chunk_type: ChunkType
    chunk_index: int
    start_char: int
    end_char: int

    # Hierarchical context
    section_path: list[str] = field(default_factory=list)  # ["H1", "H2", "H3"]
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: list[str] = field(default_factory=list)

    # Source info
    document_filename: str = ""
    page_numbers: list[int] = field(default_factory=list)

    # Overlap tracking
    overlap_with_previous: int = 0  # Characters overlapping with previous chunk
    overlap_with_next: int = 0  # Characters overlapping with next chunk

    metadata: dict = field(default_factory=dict)


class HierarchicalChunker:
    """Chunk documents while preserving hierarchical structure."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        min_chunk_size: int = settings.min_chunk_size,
        max_chunk_size: int = settings.max_chunk_size,
        similarity_threshold: float = 0.90,
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            similarity_threshold: Threshold for duplicate detection
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        """Chunk a parsed document into hierarchical chunks.

        Args:
            document: ParsedDocument from parser

        Returns:
            List of TextChunk objects
        """
        logger.info("Chunking document", filename=document.filename)

        # First pass: group elements by section
        sections = self._group_by_section(document.elements)

        # Second pass: chunk each section
        chunks = []
        char_offset = 0

        for section_path, elements in sections:
            section_chunks = self._chunk_section(
                elements=elements,
                section_path=section_path,
                document_filename=document.filename,
                start_char_offset=char_offset,
            )

            # Update character offset
            if section_chunks:
                char_offset = section_chunks[-1].end_char

            chunks.extend(section_chunks)

        # Third pass: deduplicate similar chunks
        chunks = self._deduplicate_chunks(chunks)

        # Fourth pass: assign indices and link parent-child relationships
        chunks = self._assign_indices_and_links(chunks)

        logger.info(
            "Document chunked",
            filename=document.filename,
            chunks=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
        )

        return chunks

    def _group_by_section(
        self, elements: list[ParsedElement]
    ) -> list[tuple[list[str], list[ParsedElement]]]:
        """Group elements by their section hierarchy.

        Returns list of (section_path, elements) tuples.
        """
        sections: list[tuple[list[str], list[ParsedElement]]] = []
        current_path: list[str] = []
        current_elements: list[ParsedElement] = []

        for elem in elements:
            if elem.element_type == ChunkType.HEADING:
                # Save current section if it has content
                if current_elements:
                    sections.append((current_path.copy(), current_elements))
                    current_elements = []

                # Update path based on heading level
                level = elem.heading_level or 1
                # Truncate path to parent level and add new heading
                current_path = current_path[: level - 1]
                current_path.append(elem.text)

                # Add heading as first element of new section
                current_elements.append(elem)
            else:
                current_elements.append(elem)

        # Don't forget last section
        if current_elements:
            sections.append((current_path.copy(), current_elements))

        return sections

    def _chunk_section(
        self,
        elements: list[ParsedElement],
        section_path: list[str],
        document_filename: str,
        start_char_offset: int,
    ) -> list[TextChunk]:
        """Chunk a section's elements.

        Args:
            elements: Elements in this section
            section_path: Hierarchical path to this section
            document_filename: Source document filename
            start_char_offset: Character offset from document start

        Returns:
            List of chunks for this section
        """
        chunks = []
        current_text = ""
        current_types: list[ChunkType] = []
        current_pages: list[int] = []
        char_offset = start_char_offset

        for elem in elements:
            elem_tokens = self._count_tokens(elem.text)

            # If single element exceeds max size, split it
            if elem_tokens > self.max_chunk_size:
                # First, save any accumulated text
                if current_text:
                    chunk = self._create_chunk(
                        text=current_text.strip(),
                        chunk_types=current_types,
                        section_path=section_path,
                        document_filename=document_filename,
                        start_char=char_offset,
                        page_numbers=current_pages,
                    )
                    if chunk:
                        chunks.append(chunk)
                        char_offset = chunk.end_char
                    current_text = ""
                    current_types = []
                    current_pages = []

                # Split the large element
                split_chunks = self._split_large_text(
                    text=elem.text,
                    chunk_type=elem.element_type,
                    section_path=section_path,
                    document_filename=document_filename,
                    start_char=char_offset,
                    page_number=elem.page_number,
                )
                chunks.extend(split_chunks)
                if split_chunks:
                    char_offset = split_chunks[-1].end_char
                continue

            # Check if adding this element would exceed chunk size
            combined_text = f"{current_text}\n\n{elem.text}" if current_text else elem.text
            combined_tokens = self._count_tokens(combined_text)

            if combined_tokens > self.chunk_size and current_text:
                # Create chunk from accumulated text
                chunk = self._create_chunk(
                    text=current_text.strip(),
                    chunk_types=current_types,
                    section_path=section_path,
                    document_filename=document_filename,
                    start_char=char_offset,
                    page_numbers=current_pages,
                )
                if chunk:
                    chunks.append(chunk)
                    char_offset = chunk.end_char

                # Start new accumulation with overlap
                overlap_text = self._get_overlap_text(current_text)
                current_text = f"{overlap_text}\n\n{elem.text}" if overlap_text else elem.text
                current_types = [elem.element_type]
                current_pages = [elem.page_number] if elem.page_number else []
            else:
                # Add to current accumulation
                current_text = combined_text
                current_types.append(elem.element_type)
                if elem.page_number and elem.page_number not in current_pages:
                    current_pages.append(elem.page_number)

        # Create final chunk from remaining text
        if current_text:
            chunk = self._create_chunk(
                text=current_text.strip(),
                chunk_types=current_types,
                section_path=section_path,
                document_filename=document_filename,
                start_char=char_offset,
                page_numbers=current_pages,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        text: str,
        chunk_types: list[ChunkType],
        section_path: list[str],
        document_filename: str,
        start_char: int,
        page_numbers: list[int],
    ) -> Optional[TextChunk]:
        """Create a TextChunk from accumulated text."""
        token_count = self._count_tokens(text)

        # Skip if too small
        if token_count < self.min_chunk_size:
            return None

        # Determine primary chunk type
        if ChunkType.TABLE in chunk_types:
            chunk_type = ChunkType.TABLE
        elif ChunkType.HEADING in chunk_types and len(chunk_types) == 1:
            chunk_type = ChunkType.HEADING
        elif ChunkType.LIST in chunk_types:
            chunk_type = ChunkType.LIST
        else:
            chunk_type = ChunkType.PARAGRAPH

        return TextChunk(
            id=str(uuid4()),
            text=text,
            token_count=token_count,
            chunk_type=chunk_type,
            chunk_index=0,  # Will be assigned later
            start_char=start_char,
            end_char=start_char + len(text),
            section_path=section_path,
            document_filename=document_filename,
            page_numbers=page_numbers,
        )

    def _split_large_text(
        self,
        text: str,
        chunk_type: ChunkType,
        section_path: list[str],
        document_filename: str,
        start_char: int,
        page_number: Optional[int],
    ) -> list[TextChunk]:
        """Split a large text into multiple chunks with overlap."""
        chunks = []
        sentences = self._split_into_sentences(text)

        current_text = ""
        char_offset = start_char

        for sentence in sentences:
            combined = f"{current_text} {sentence}" if current_text else sentence
            combined_tokens = self._count_tokens(combined)

            if combined_tokens > self.chunk_size and current_text:
                # Create chunk
                chunk = TextChunk(
                    id=str(uuid4()),
                    text=current_text.strip(),
                    token_count=self._count_tokens(current_text),
                    chunk_type=chunk_type,
                    chunk_index=0,
                    start_char=char_offset,
                    end_char=char_offset + len(current_text),
                    section_path=section_path,
                    document_filename=document_filename,
                    page_numbers=[page_number] if page_number else [],
                )
                chunks.append(chunk)
                char_offset = chunk.end_char

                # Start new with overlap
                overlap = self._get_overlap_text(current_text)
                current_text = f"{overlap} {sentence}" if overlap else sentence
            else:
                current_text = combined

        # Final chunk
        if current_text:
            chunk = TextChunk(
                id=str(uuid4()),
                text=current_text.strip(),
                token_count=self._count_tokens(current_text),
                chunk_type=chunk_type,
                chunk_index=0,
                start_char=char_offset,
                end_char=char_offset + len(current_text),
                section_path=section_path,
                document_filename=document_filename,
                page_numbers=[page_number] if page_number else [],
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap portion from the end of text."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return ""

        overlap_text = ""
        overlap_tokens = 0

        # Take sentences from end until we hit overlap target
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence)
            if overlap_tokens + sentence_tokens > self.chunk_overlap:
                break
            overlap_text = f"{sentence} {overlap_text}".strip()
            overlap_tokens += sentence_tokens

        return overlap_text

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Allow all special tokens to be encoded as regular text
        # This handles texts containing LLM-related content like <|endoftext|>
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def _deduplicate_chunks(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """Remove near-duplicate chunks based on text similarity."""
        if len(chunks) <= 1:
            return chunks

        deduplicated = [chunks[0]]

        for chunk in chunks[1:]:
            is_duplicate = False
            for existing in deduplicated:
                similarity = self._compute_jaccard_similarity(
                    chunk.text, existing.text
                )
                if similarity > self.similarity_threshold:
                    # Merge into existing chunk if it adds content
                    if len(chunk.text) > len(existing.text):
                        existing.text = chunk.text
                        existing.token_count = chunk.token_count
                        existing.end_char = chunk.end_char
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(chunk)

        if len(deduplicated) < len(chunks):
            logger.debug(
                "Deduplicated chunks",
                original=len(chunks),
                after=len(deduplicated),
            )

        return deduplicated

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _assign_indices_and_links(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """Assign sequential indices and parent-child relationships."""
        # Group chunks by section path depth
        section_chunks: dict[str, list[TextChunk]] = {}

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

            # Create section key
            section_key = "/".join(chunk.section_path) if chunk.section_path else ""

            if section_key not in section_chunks:
                section_chunks[section_key] = []
            section_chunks[section_key].append(chunk)

        # Link parent-child based on section hierarchy
        for section_key, section_chunk_list in section_chunks.items():
            if not section_key:
                continue

            # Find parent section
            parts = section_key.split("/")
            for depth in range(len(parts) - 1, 0, -1):
                parent_key = "/".join(parts[:depth])
                if parent_key in section_chunks:
                    parent_chunks = section_chunks[parent_key]
                    if parent_chunks:
                        parent = parent_chunks[0]
                        for child in section_chunk_list:
                            child.parent_chunk_id = parent.id
                            if child.id not in parent.child_chunk_ids:
                                parent.child_chunk_ids.append(child.id)
                    break

        return chunks
