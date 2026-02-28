"""Document parser for PDF, DOCX, and Markdown files."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, BinaryIO

import structlog
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text
from unstructured.documents.elements import (
    Element,
    Title,
    NarrativeText,
    ListItem,
    Table,
    FigureCaption,
    Header,
    Footer,
)

from src.config.constants import ChunkType, DocumentType

logger = structlog.get_logger(__name__)


@dataclass
class ParsedElement:
    """A parsed element from a document."""

    text: str
    element_type: ChunkType
    metadata: dict = field(default_factory=dict)
    heading_level: Optional[int] = None
    parent_heading: Optional[str] = None
    page_number: Optional[int] = None
    coordinates: Optional[dict] = None


@dataclass
class ParsedDocument:
    """Result of parsing a document."""

    filename: str
    doc_type: DocumentType
    elements: list[ParsedElement]
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    raw_text: str = ""
    file_hash: str = ""
    metadata: dict = field(default_factory=dict)


class DocumentParser:
    """Parse documents into structured elements."""

    def __init__(
        self,
        extract_images: bool = False,
        extract_tables: bool = True,
        ocr_languages: list[str] | None = None,
    ):
        """Initialize the document parser.

        Args:
            extract_images: Whether to extract images (OCR)
            extract_tables: Whether to extract tables
            ocr_languages: Languages for OCR (default: English)
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.ocr_languages = ocr_languages or ["eng"]

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """Parse a document file.

        Args:
            file_path: Path to the document file

        Returns:
            ParsedDocument with extracted elements
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_type = self._detect_doc_type(path)
        file_hash = self._compute_file_hash(path)

        logger.info("Parsing document", filename=path.name, doc_type=doc_type)

        # Parse based on document type
        if doc_type == DocumentType.PDF:
            elements = self._parse_pdf(path)
        elif doc_type == DocumentType.DOCX:
            elements = self._parse_docx(path)
        elif doc_type == DocumentType.MARKDOWN:
            elements = self._parse_markdown(path)
        elif doc_type == DocumentType.TEXT:
            elements = self._parse_text(path)
        else:
            # Try auto-detection
            elements = self._parse_auto(path)

        # Extract title from first heading if available
        title = None
        for elem in elements:
            if elem.element_type == ChunkType.HEADING:
                title = elem.text
                break
                
        # If no heading found, default to the filename stem (without extension)
        if not title:
            title = path.stem

        # Build raw text
        raw_text = "\n\n".join(elem.text for elem in elements)

        parsed_doc = ParsedDocument(
            filename=path.name,
            doc_type=doc_type,
            elements=elements,
            title=title,
            raw_text=raw_text,
            file_hash=file_hash,
        )

        logger.info(
            "Document parsed",
            filename=path.name,
            elements=len(elements),
            raw_text_length=len(raw_text),
        )

        return parsed_doc

    def parse_bytes(
        self,
        content: bytes | BinaryIO,
        filename: str,
        doc_type: Optional[DocumentType] = None,
    ) -> ParsedDocument:
        """Parse document from bytes or file-like object.

        Args:
            content: Document content as bytes or file-like object
            filename: Original filename
            doc_type: Document type (auto-detected if not provided)

        Returns:
            ParsedDocument with extracted elements
        """
        import tempfile
        import os

        # Determine doc type from filename if not provided
        if doc_type is None:
            doc_type = self._detect_doc_type(Path(filename))

        # Write to temp file for parsing
        suffix = f".{doc_type.value}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            if isinstance(content, bytes):
                tmp.write(content)
            else:
                tmp.write(content.read())
            tmp_path = tmp.name

        try:
            parsed = self.parse_file(tmp_path)
            parsed.filename = filename  # Use original filename
            
            # If the title defaulted to the temp filename, correct it to the original filename
            if not parsed.title or parsed.title == Path(tmp_path).stem:
                parsed.title = Path(filename).stem
                
            return parsed
        finally:
            os.unlink(tmp_path)

    def _detect_doc_type(self, path: Path) -> DocumentType:
        """Detect document type from file extension."""
        suffix = path.suffix.lower()
        type_map = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".md": DocumentType.MARKDOWN,
            ".markdown": DocumentType.MARKDOWN,
            ".txt": DocumentType.TEXT,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
        }
        return type_map.get(suffix, DocumentType.TEXT)

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _parse_pdf(self, path: Path) -> list[ParsedElement]:
        """Parse PDF document."""
        elements = partition_pdf(
            filename=str(path),
            strategy="hi_res" if self.extract_images else "fast",
            infer_table_structure=self.extract_tables,
            languages=self.ocr_languages,
        )
        return self._convert_elements(elements)

    def _parse_docx(self, path: Path) -> list[ParsedElement]:
        """Parse DOCX document."""
        elements = partition_docx(filename=str(path))
        return self._convert_elements(elements)

    def _parse_markdown(self, path: Path) -> list[ParsedElement]:
        """Parse Markdown document."""
        elements = partition_md(filename=str(path))
        return self._convert_elements(elements)

    def _parse_text(self, path: Path) -> list[ParsedElement]:
        """Parse plain text document."""
        elements = partition_text(filename=str(path))
        return self._convert_elements(elements)

    def _parse_auto(self, path: Path) -> list[ParsedElement]:
        """Parse document with auto-detection."""
        elements = partition(filename=str(path))
        return self._convert_elements(elements)

    def _convert_elements(self, elements: list[Element]) -> list[ParsedElement]:
        """Convert unstructured elements to ParsedElement."""
        parsed_elements = []
        current_heading = None
        current_heading_level = 0

        for elem in elements:
            # Skip headers and footers
            if isinstance(elem, (Header, Footer)):
                continue

            # Determine element type
            if isinstance(elem, Title):
                element_type = ChunkType.HEADING
                heading_level = self._infer_heading_level(elem)
                current_heading = elem.text
                current_heading_level = heading_level
            elif isinstance(elem, Table):
                element_type = ChunkType.TABLE
                heading_level = None
            elif isinstance(elem, ListItem):
                element_type = ChunkType.LIST
                heading_level = None
            elif isinstance(elem, FigureCaption):
                element_type = ChunkType.FIGURE_CAPTION
                heading_level = None
            else:
                element_type = ChunkType.PARAGRAPH
                heading_level = None

            # Extract metadata
            metadata = {}
            if hasattr(elem, "metadata"):
                if hasattr(elem.metadata, "page_number"):
                    metadata["page_number"] = elem.metadata.page_number
                if hasattr(elem.metadata, "coordinates"):
                    metadata["coordinates"] = elem.metadata.coordinates

            text = str(elem.text).strip()
            if not text:
                continue

            parsed_elem = ParsedElement(
                text=text,
                element_type=element_type,
                metadata=metadata,
                heading_level=heading_level,
                parent_heading=current_heading if element_type != ChunkType.HEADING else None,
                page_number=metadata.get("page_number"),
            )
            parsed_elements.append(parsed_elem)

        return parsed_elements

    def _infer_heading_level(self, elem: Element) -> int:
        """Infer heading level from element metadata."""
        # Try to get level from metadata
        if hasattr(elem, "metadata"):
            if hasattr(elem.metadata, "category_depth"):
                return elem.metadata.category_depth
            if hasattr(elem.metadata, "emphasized"):
                return 1 if elem.metadata.emphasized else 2

        # Default heuristics based on text length and formatting
        text = str(elem.text)
        if len(text) < 50:
            return 1
        elif len(text) < 100:
            return 2
        return 3

    def extract_tables_as_markdown(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        """Convert table elements to markdown format with natural language summary.

        Args:
            elements: List of parsed elements

        Returns:
            Elements with tables converted to markdown
        """
        processed = []
        for elem in elements:
            if elem.element_type == ChunkType.TABLE:
                # Table text from unstructured is already in a readable format
                # Add markdown formatting
                markdown_table = f"**Table:**\n\n{elem.text}"
                elem.text = markdown_table
                elem.metadata["original_format"] = "table"
            processed.append(elem)
        return processed
