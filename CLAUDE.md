# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-Healing RAG Engine - A production-grade RAG system for intelligent document Q&A with three novel contributions:
1. **Confidence-Calibrated Retrieval** - Multi-signal confidence scoring
2. **Agentic Self-Correction Loops** - Failure diagnosis and correction
3. **Claim-Level Causal Validation** - Hallucination detection and mitigation

**GitHub:** https://github.com/Gustav-Proxi/RAG

## Quick Start

```bash
# Setup environment
python -m venv venv && source venv/bin/activate
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your DATABASE_URL, ANTHROPIC_API_KEY, GOOGLE_API_KEY

# Run backend
python -m uvicorn src.api.main:app --port 8000

# Run frontend (new terminal)
cd frontend && npm install && npm run dev
```

## Contributors & Domain Ownership

### Vaishak (Owner)
**Core RAG System:**
- `src/agents/` - LangGraph correction loops
- `src/validation/` - Claim validation & NLI
- `src/generation/` - Response generation & planning
- `src/retrieval/search.py`, `reranker.py`, `confidence.py`
- `src/ingestion/pipeline.py`, `parser.py`
- `src/providers/` - LLM provider integrations

**Endpoints:** `/sessions/*`, `/auth/*`, `/ingest/batch`, `/metrics`, WebSocket

### Sanika (Contributor)
**Frontend:** All `frontend/src/components/*.tsx`

**Core RAG:**
- `src/ingestion/chunker.py` - Chunking strategies
- `src/config/cache.py` - Caching layer
- `src/retrieval/filters.py` - Search filters

**Endpoints:** `/documents/{id}/tags`, `/rename`, `/bulk-delete`, `/workspaces/{id}/stats`, `/searches/recent`, `/health` expansion

**Task files (gitignored):** `SANIKA_TASKS.md`, `VAISHAK_TASKS.md`

## Project Structure

```
RAG/
├── src/
│   ├── api/              # FastAPI endpoints
│   │   ├── main.py       # Routes and app setup
│   │   └── middleware.py # Security middleware
│   ├── ingestion/        # Document parsing & chunking
│   │   ├── parser.py     # PDF/DOCX/MD parsing
│   │   ├── chunker.py    # Chunking strategies (Sanika)
│   │   ├── enrichment.py # LLM metadata generation
│   │   └── pipeline.py   # Orchestration
│   ├── embeddings/       # BGE-M3 embedding service
│   ├── database/         # PostgreSQL + pgvector models
│   ├── retrieval/        # Hybrid search, reranking
│   │   ├── search.py     # Hybrid search (vector + keyword + HyDE)
│   │   ├── fusion.py     # Reciprocal Rank Fusion
│   │   ├── reranker.py   # Cross-encoder reranking
│   │   ├── confidence.py # Multi-signal confidence estimation
│   │   └── filters.py    # Search filters (Sanika - new)
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
│   ├── providers/        # LLM provider integrations
│   │   ├── __init__.py
│   │   └── mercury.py    # Mercury 2 client (10x faster)
│   └── config/           # Settings, constants
│       ├── settings.py   # Environment config
│       ├── cache.py      # Caching layer (Sanika - new)
│       └── telemetry.py  # Optional OpenTelemetry
├── frontend/             # Next.js 15 (port 3000)
│   └── src/
│       ├── app/          # Pages
│       │   ├── page.tsx  # Main chat interface
│       │   └── globals.css
│       └── components/   # UI components (Sanika)
│           ├── Sidebar.tsx
│           ├── ChatInput.tsx
│           ├── WorkspaceSelector.tsx
│           ├── SettingsModal.tsx    # API key & model config UI
│           ├── ReasoningTrace.tsx
│           ├── StreamingResponse.tsx
│           └── CitationCard.tsx
├── docs/
│   └── MERCURY_2_ANALYSIS.md
├── tests/
└── scripts/
```

## Key APIs

### Endpoints
- `POST /ingest` - Upload documents (with workspace_id)
- `POST /query` - Main RAG query (with corrections)
- `POST /query/stream` - Streaming query via SSE
- `GET /documents` - List documents (filter by workspace_id)
- `DELETE /documents/{id}` - Delete a document
- `GET /workspaces` - List workspaces
- `POST /workspaces` - Create a workspace
- `POST /search` - Direct search without generation
- `GET /health` - Health check with DB stats
- `GET /settings/providers` - List available LLM providers
- `POST /settings/validate-key` - Validate an API key
- `POST /settings/user` - Save user settings

### Core Classes
- `IngestionPipeline` - Document processing
- `HybridSearcher` - Multi-signal retrieval
- `ConfidenceEstimator` - Confidence scoring
- `RAGAgentGraph` - LangGraph workflow
- `ClaimValidator` - Hallucination detection
- `MercuryClient` - Mercury 2 API client

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL with pgvector (Neon recommended)
- `ANTHROPIC_API_KEY` - Claude API key
- `GOOGLE_API_KEY` - Gemini API key

Optional:
- `EMBEDDING_MODEL` - Default: BAAI/bge-m3
- `EMBEDDING_DEVICE` - cpu/cuda/mps
- `GENERATION_MODEL` - Default: gemini-2.0-flash
- `AGENT_MODEL` - Default: claude-haiku-4-5-20251001

### Mercury 2 (Optional - 10x Faster)
- `MERCURY_API_KEY` - Inception Labs API key (https://platform.inceptionlabs.ai)
- `USE_MERCURY_FOR_AGENTS` - Use Mercury for agent loops (default: false)
- `USE_MERCURY_FOR_GENERATION` - Use Mercury for responses (default: false)
- `MERCURY_REASONING_EFFORT` - "low" (fast) or "high" (quality)

Mercury 2 uses diffusion-based generation (~1000 tok/s vs ~89 for Haiku).

## Common Commands

```bash
# Run tests
pytest tests/

# Ingest documents
python scripts/ingest.py /path/to/documents

# Run evaluation
python scripts/evaluate.py eval_dataset.jsonl -o results.json

# Start API server
uvicorn src.api.main:app --reload --port 8000

# Start frontend
cd frontend && npm run dev
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

### Known Issues
- BGE reranker scores are 0-0.2 range, not 0-1 (min_score set to 0.0)
- Settings use `@lru_cache` - restart server after .env changes
- OpenTelemetry is optional (graceful fallback if not installed)

## Tech Stack

- **Embeddings**: BGE-M3 (1024-dim dense + sparse)
- **LLM**: Claude Haiku (agents), Gemini Flash (generation), Mercury 2 (optional, 10x faster)
- **Database**: PostgreSQL + pgvector (Neon)
- **Framework**: FastAPI + LangGraph
- **Reranker**: bge-reranker-v2-m3
- **NLI**: DeBERTa-v3-large-mnli
- **Frontend**: Next.js 15, Tailwind CSS, Framer Motion

## Mercury 2 Provider

Located at `src/providers/mercury.py`. Provides async client for Mercury 2 API.

```python
from src.providers.mercury import MercuryClient, get_mercury_client

# Get client (returns None if MERCURY_API_KEY not set)
client = get_mercury_client()
if client:
    response = await client.generate(
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="low",  # "low" or "high"
    )
```

## Settings UI

The frontend includes a Settings modal (`frontend/src/components/SettingsModal.tsx`) that allows users to:
- Add API keys for Anthropic, Google, Mercury 2, OpenAI
- Configure Ollama for local models
- Select models for agents vs generation
- Validate API keys before saving
- Settings stored in localStorage

Access via the gear icon (⚙️) in the header.
