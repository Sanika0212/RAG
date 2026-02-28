"""Document ingestion pipeline orchestrator."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, BinaryIO
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.constants import DocumentType
from src.database.connection import get_db
from src.database.models import Document, Section, Chunk, ChunkMetadata
from src.embeddings.bge_m3 import BGEEmbedder, get_embedder
from src.ingestion.parser import DocumentParser, ParsedDocument
from src.ingestion.chunker import HierarchicalChunker, TextChunk
from src.ingestion.enrichment import (
    MetadataEnricher,
    EnrichedMetadata,
    generate_hypothetical_questions_embeddings,
)

logger = structlog.get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    document_id: UUID
    filename: str
    total_chunks: int
    total_tokens: int
    processing_time_ms: int
    success: bool
    error: Optional[str] = None


class IngestionPipeline:
    """Orchestrate the complete document ingestion process."""

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        chunker: Optional[HierarchicalChunker] = None,
        enricher: Optional[MetadataEnricher] = None,
        embedder: Optional[BGEEmbedder] = None,
        skip_enrichment: bool = False,
    ):
        """Initialize the ingestion pipeline.

        Args:
            parser: Document parser instance
            chunker: Hierarchical chunker instance
            enricher: Metadata enricher instance
            embedder: BGE embedder instance
            skip_enrichment: Skip LLM enrichment (for faster testing)
        """
        self.parser = parser or DocumentParser()
        self.chunker = chunker or HierarchicalChunker()
        self.enricher = enricher or MetadataEnricher()
        self.embedder = embedder or get_embedder()
        self.skip_enrichment = skip_enrichment

    async def ingest_file(
        self,
        file_path: str | Path,
        session: Optional[AsyncSession] = None,
    ) -> IngestionResult:
        """Ingest a document from file path.

        Args:
            file_path: Path to the document file
            session: Optional database session (creates new if not provided)

        Returns:
            IngestionResult with processing details
        """
        start_time = datetime.now()
        path = Path(file_path)

        try:
            # Parse document
            parsed_doc = self.parser.parse_file(path)

            # Check for duplicate
            if session:
                existing = await self._check_duplicate(session, parsed_doc.file_hash)
                if existing:
                    return IngestionResult(
                        document_id=existing.id,
                        filename=parsed_doc.filename,
                        total_chunks=existing.total_chunks,
                        total_tokens=existing.total_tokens,
                        processing_time_ms=0,
                        success=True,
                        error="Document already exists (duplicate hash)",
                    )

            # Process document
            return await self._process_document(parsed_doc, session, start_time)

        except Exception as e:
            logger.error("Ingestion failed", file=str(file_path), error=str(e))
            elapsed = int((datetime.now() - start_time).total_seconds() * 1000)
            return IngestionResult(
                document_id=UUID(int=0),
                filename=path.name,
                total_chunks=0,
                total_tokens=0,
                processing_time_ms=elapsed,
                success=False,
                error=str(e),
            )

    async def ingest_bytes(
        self,
        content: bytes | BinaryIO,
        filename: str,
        doc_type: Optional[DocumentType] = None,
        session: Optional[AsyncSession] = None,
        workspace_id: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest a document from bytes.

        Args:
            content: Document content as bytes or file-like object
            filename: Original filename
            doc_type: Document type (auto-detected if not provided)
            session: Optional database session
            workspace_id: Optional workspace ID to associate document with

        Returns:
            IngestionResult with processing details
        """
        start_time = datetime.now()

        try:
            # Parse document from bytes
            parsed_doc = self.parser.parse_bytes(content, filename, doc_type)

            # Check for duplicate
            if session:
                existing = await self._check_duplicate(session, parsed_doc.file_hash)
                if existing:
                    return IngestionResult(
                        document_id=existing.id,
                        filename=filename,
                        total_chunks=existing.total_chunks,
                        total_tokens=existing.total_tokens,
                        processing_time_ms=0,
                        success=True,
                        error="Document already exists (duplicate hash)",
                    )

            return await self._process_document(parsed_doc, session, start_time, workspace_id)

        except Exception as e:
            logger.error("Ingestion failed", filename=filename, error=str(e))
            elapsed = int((datetime.now() - start_time).total_seconds() * 1000)
            return IngestionResult(
                document_id=UUID(int=0),
                filename=filename,
                total_chunks=0,
                total_tokens=0,
                processing_time_ms=elapsed,
                success=False,
                error=str(e),
            )

    async def _process_document(
        self,
        parsed_doc: ParsedDocument,
        session: Optional[AsyncSession],
        start_time: datetime,
        workspace_id: Optional[str] = None,
    ) -> IngestionResult:
        """Process a parsed document through the pipeline.

        IMPORTANT: All heavy processing (chunking, embedding) happens BEFORE
        opening any database connection to avoid connection timeouts.
        """
        # Chunk document
        logger.info("Chunking document", filename=parsed_doc.filename)
        chunks = self.chunker.chunk_document(parsed_doc)

        if not chunks:
            raise ValueError("No chunks generated from document")

        # Generate embeddings (this is slow - no DB connection yet!)
        logger.info("Generating embeddings", chunks=len(chunks))
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed_documents(chunk_texts)
        logger.info("Embeddings generated", chunks=len(chunks))

        # Enrich metadata (optional - also slow, no DB connection)
        enriched_metadata: list[EnrichedMetadata] = []
        if not self.skip_enrichment:
            logger.info("Enriching metadata", chunks=len(chunks))
            enriched_metadata = await self.enricher.enrich_chunks(
                chunks,
                document_context=parsed_doc.title,
            )

            # Generate HQ embeddings
            enriched_metadata = await generate_hypothetical_questions_embeddings(
                enriched_metadata,
                self.embedder,
            )

        # Calculate totals
        total_tokens = sum(c.token_count for c in chunks)

        # NOW save to database - open fresh connection for quick save
        document_id: UUID
        if session:
            # Use provided session
            document_id = await self._save_to_database(
                session=session,
                parsed_doc=parsed_doc,
                chunks=chunks,
                embeddings=embeddings,
                enriched_metadata=enriched_metadata,
                total_tokens=total_tokens,
                workspace_id=workspace_id,
            )
        else:
            # Open a fresh session just for saving (avoids timeout during embedding)
            logger.info("Opening database session for save", chunks=len(chunks))
            async with get_db() as save_session:
                # Quick duplicate check
                existing = await self._check_duplicate(save_session, parsed_doc.file_hash)
                if existing:
                    elapsed = int((datetime.now() - start_time).total_seconds() * 1000)
                    return IngestionResult(
                        document_id=existing.id,
                        filename=parsed_doc.filename,
                        total_chunks=existing.total_chunks,
                        total_tokens=existing.total_tokens,
                        processing_time_ms=elapsed,
                        success=True,
                        error="Document already exists (duplicate hash)",
                    )

                document_id = await self._save_to_database(
                    session=save_session,
                    parsed_doc=parsed_doc,
                    chunks=chunks,
                    embeddings=embeddings,
                    enriched_metadata=enriched_metadata,
                    total_tokens=total_tokens,
                    workspace_id=workspace_id,
                )
            logger.info("Document saved to database", document_id=str(document_id))

        elapsed = int((datetime.now() - start_time).total_seconds() * 1000)

        logger.info(
            "Document ingested",
            document_id=str(document_id),
            filename=parsed_doc.filename,
            chunks=len(chunks),
            tokens=total_tokens,
            time_ms=elapsed,
        )

        return IngestionResult(
            document_id=document_id,
            filename=parsed_doc.filename,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            processing_time_ms=elapsed,
            success=True,
        )

    async def _check_duplicate(
        self,
        session: AsyncSession,
        file_hash: str,
    ) -> Optional[Document]:
        """Check if document already exists by hash."""
        result = await session.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        return result.scalar_one_or_none()

    async def _save_to_database(
        self,
        session: AsyncSession,
        parsed_doc: ParsedDocument,
        chunks: list[TextChunk],
        embeddings: list,
        enriched_metadata: list[EnrichedMetadata],
        total_tokens: int,
        workspace_id: Optional[str] = None,
    ) -> UUID:
        """Save document, chunks, and metadata to database."""
        from uuid import UUID as UUIDType

        # Create document record
        document = Document(
            filename=parsed_doc.filename,
            doc_type=parsed_doc.doc_type,
            title=parsed_doc.title,
            author=parsed_doc.author,
            publication_date=parsed_doc.publication_date,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            file_hash=parsed_doc.file_hash,
            raw_text_length=len(parsed_doc.raw_text),
            metadata_json=parsed_doc.metadata,
            workspace_id=UUIDType(workspace_id) if workspace_id else None,
        )
        session.add(document)
        await session.flush()  # Get document ID

        # Create sections from chunk hierarchy
        section_map: dict[str, Section] = {}
        for chunk in chunks:
            if chunk.section_path:
                for i, heading in enumerate(chunk.section_path):
                    path_key = "/".join(chunk.section_path[: i + 1])
                    if path_key not in section_map:
                        parent_key = "/".join(chunk.section_path[:i]) if i > 0 else None
                        parent_section = section_map.get(parent_key) if parent_key else None

                        section = Section(
                            document_id=document.id,
                            parent_section_id=parent_section.id if parent_section else None,
                            heading=heading,
                            level=i + 1,
                            ordering=len(section_map),
                            path=path_key,
                        )
                        session.add(section)
                        await session.flush()
                        section_map[path_key] = section

        # Create metadata lookup
        metadata_map = {m.chunk_id: m for m in enriched_metadata}

        # Create chunks
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Find section for this chunk
            section_id = None
            if chunk.section_path:
                path_key = "/".join(chunk.section_path)
                if path_key in section_map:
                    section_id = section_map[path_key].id

            db_chunk = Chunk(
                document_id=document.id,
                section_id=section_id,
                text=chunk.text,
                token_count=chunk.token_count,
                chunk_type=chunk.chunk_type,
                chunk_index=i,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                embedding=embedding.dense.tolist(),
            )
            session.add(db_chunk)
            await session.flush()

            # Add metadata if available
            if chunk.id in metadata_map:
                meta = metadata_map[chunk.id]
                db_metadata = ChunkMetadata(
                    chunk_id=db_chunk.id,
                    summary=meta.summary,
                    keywords=meta.keywords,
                    hypothetical_questions=meta.hypothetical_questions,
                    entity_mentions=meta.entity_mentions,
                    topic_tags=meta.topic_tags,
                    difficulty_level=meta.difficulty_level,
                    temporal_references=meta.temporal_references,
                    medical_concepts=meta.medical_concepts,
                    confidence_score=meta.confidence_score,
                    hq_embeddings=getattr(meta, 'hq_embeddings', None),
                )
                session.add(db_metadata)

        await session.commit()
        return document.id

    async def delete_document(
        self,
        document_id: UUID,
        session: AsyncSession,
    ) -> bool:
        """Delete a document and all associated data.

        Args:
            document_id: Document ID to delete
            session: Database session

        Returns:
            True if deleted, False if not found
        """
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            return False

        await session.delete(document)
        await session.commit()

        logger.info("Document deleted", document_id=str(document_id))
        return True


# Convenience function for CLI usage
async def ingest_directory(
    directory: str | Path,
    extensions: list[str] | None = None,
    skip_enrichment: bool = False,
) -> list[IngestionResult]:
    """Ingest all documents in a directory.

    Args:
        directory: Path to directory
        extensions: File extensions to process (default: pdf, docx, md, txt)
        skip_enrichment: Skip LLM enrichment

    Returns:
        List of IngestionResult objects
    """
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    extensions = extensions or [".pdf", ".docx", ".md", ".txt"]
    files = [f for f in path.rglob("*") if f.suffix.lower() in extensions]

    logger.info("Found files to ingest", count=len(files), directory=str(directory))

    pipeline = IngestionPipeline(skip_enrichment=skip_enrichment)
    results = []

    async with get_db() as session:
        for file in files:
            result = await pipeline.ingest_file(file, session)
            results.append(result)

    return results
