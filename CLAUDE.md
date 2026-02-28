# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-Healing RAG Engine - A production-grade RAG system for intelligent document Q&A with three novel contributions:
1. **Confidence-Calibrated Retrieval** - Multi-signal confidence scoring
2. **Agentic Self-Correction Loops** - Failure diagnosis and correction
3. **Claim-Level Causal Validation** - Hallucination detection and mitigation

## Quick Start

```bash
# Setup environment
python -m venv venv && source venv/bin/activate
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your DATABASE_URL and ANTHROPIC_API_KEY

# Run server
python -m src.api.main
```

## Project Structure

```
RAG/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── ingestion/        # Document parsing & chunking
│   │   ├── parser.py     # PDF/DOCX/MD parsing
│   │   ├── chunker.py    # Hierarchical chunking
│   │   ├── enrichment.py # LLM metadata generation
│   │   └── pipeline.py   # Orchestration
│   ├── embeddings/       # BGE-M3 embedding service
│   ├── database/         # PostgreSQL + pgvector models
│   ├── retrieval/        # Hybrid search, reranking
│   │   ├── search.py     # Hybrid search (vector + keyword + HyDE)
│   │   ├── fusion.py     # Reciprocal Rank Fusion
│   │   ├── reranker.py   # Cross-encoder reranking
│   │   └── confidence.py # Multi-signal confidence estimation
│   ├── agents/           # LangGraph correction loops
│   │   ├── diagnosis.py  # Failure mode classification
│   │   ├── correction.py # Correction strategies
│   │   └── graph.py      # LangGraph state machine
│   ├── generation/       # Confidence-conditioned generation
│   │   ├── generator.py  # Response generation
│   │   └── planner.py    # Query decomposition
│   ├── validation/       # Claim-level validation
│   │   ├── claims.py     # Claim extraction & validation
│   │   └── nli.py        # NLI scoring
│   └── config/           # Settings, constants
├── tests/
│   ├── unit/
│   ├── integration/
│   └── adversarial/      # Robustness tests
├── scripts/
│   ├── ingest.py         # Document ingestion CLI
│   └── evaluate.py       # Evaluation pipeline
└── frontend/             # Next.js frontend (port 3000)
    └── src/
        ├── app/          # Pages
        └── components/   # UI components
```

## Key APIs

### Endpoints
- `POST /ingest` - Upload documents
- `POST /query` - Main RAG query (with corrections)
- `GET /search` - Direct search without generation
- `GET /health` - Health check
- `GET /metrics` - System metrics

### Core Classes
- `IngestionPipeline` - Document processing
- `HybridSearcher` - Multi-signal retrieval
- `ConfidenceEstimator` - Confidence scoring
- `RAGAgentGraph` - LangGraph workflow
- `ClaimValidator` - Hallucination detection

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL with pgvector
- `ANTHROPIC_API_KEY` - Claude API key

Optional:
- `EMBEDDING_MODEL` - Default: BAAI/bge-m3
- `EMBEDDING_DEVICE` - cpu/cuda/mps
- `GENERATION_MODEL` - Default: claude-sonnet-4-20250514
- `AGENT_MODEL` - Default: claude-haiku-4-20250514

## Common Commands

```bash
# Run tests
pytest tests/

# Ingest documents
python scripts/ingest.py /path/to/documents

# Run evaluation
python scripts/evaluate.py eval_dataset.jsonl -o results.json

# Start API server
uvicorn src.api.main:app --reload
```

## Architecture Notes

### Confidence Estimation (Novel)
Multi-signal scoring combining:
1. Top similarity score (baseline quality)
2. Score dropoff (sharp dropoff = good specificity)
3. Inter-chunk coherence (semantic consistency)
4. Query coverage (LLM-assessed relevance)

### Failure Modes
- `AMBIGUITY` → Query decomposition
- `VOCAB_MISMATCH` → Synonym reformulation
- `INFO_SCATTER` → Multi-hop retrieval
- `KNOWLEDGE_GAP` → Abstention
- `GRANULARITY_MISMATCH` → Hierarchy walking

### Claim Validation (Novel)
1. Extract atomic claims from response
2. Generate counterfactual (no-context) response
3. Classify: GROUNDED | RECOVERED | UNGROUNDED
4. Rewrite removing ungrounded claims

## Tech Stack

- **Embeddings**: BGE-M3 (1024-dim dense + sparse)
- **LLM**: Claude Sonnet (generation), Claude Haiku (agents)
- **Database**: PostgreSQL + pgvector (Neon recommended)
- **Framework**: FastAPI + LangGraph
- **Reranker**: bge-reranker-v2-m3
- **NLI**: DeBERTa-v3-large-mnli
