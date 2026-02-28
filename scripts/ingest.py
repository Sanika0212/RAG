#!/usr/bin/env python3
"""CLI script for document ingestion."""

import argparse
import asyncio
from pathlib import Path

import structlog

from src.database.connection import init_db, close_db, get_db
from src.ingestion.pipeline import IngestionPipeline, ingest_directory

logger = structlog.get_logger(__name__)


async def ingest_single_file(file_path: str, skip_enrichment: bool = False) -> None:
    """Ingest a single file."""
    await init_db()

    try:
        pipeline = IngestionPipeline(skip_enrichment=skip_enrichment)
        async with get_db() as session:
            result = await pipeline.ingest_file(file_path, session)

            if result.success:
                print(f"Successfully ingested: {result.filename}")
                print(f"  Document ID: {result.document_id}")
                print(f"  Chunks: {result.total_chunks}")
                print(f"  Tokens: {result.total_tokens}")
                print(f"  Time: {result.processing_time_ms}ms")
            else:
                print(f"Failed to ingest: {result.filename}")
                print(f"  Error: {result.error}")
    finally:
        await close_db()


async def ingest_dir(directory: str, skip_enrichment: bool = False) -> None:
    """Ingest all documents in a directory."""
    await init_db()

    try:
        results = await ingest_directory(directory, skip_enrichment=skip_enrichment)

        success = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        print(f"\nIngestion complete:")
        print(f"  Successful: {success}")
        print(f"  Failed: {failed}")

        if failed > 0:
            print("\nFailed files:")
            for r in results:
                if not r.success:
                    print(f"  - {r.filename}: {r.error}")
    finally:
        await close_db()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Document ingestion for RAG system")
    parser.add_argument(
        "path",
        help="File or directory path to ingest",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip LLM metadata enrichment (faster but less rich)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        action="store_true",
        help="Treat path as a directory and ingest all supported files",
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return 1

    if args.directory or path.is_dir():
        asyncio.run(ingest_dir(str(path), args.skip_enrichment))
    else:
        asyncio.run(ingest_single_file(str(path), args.skip_enrichment))

    return 0


if __name__ == "__main__":
    exit(main())
