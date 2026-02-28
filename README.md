<div align="center">

  <h1>Self-Healing RAG Engine</h1>
  <p><strong>Production-ready Retrieval-Augmented Generation with Agentic Self-Correction, Hybrid Search, and Glassmorphism UI</strong></p>

  <p>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" /></a>
    <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js%2015-000000?style=for-the-badge&logo=next.js" alt="Next.js" /></a>
    <a href="https://postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" /></a>
    <a href="https://www.langchain.com/langgraph"><img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" /></a>
    <a href="https://anthropic.com/"><img src="https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=anthropic&logoColor=white" alt="Claude" /></a>
  </p>

</div>

---

## Overview

This RAG engine goes beyond basic vector retrieval by implementing a **self-healing architecture** that evaluates, diagnoses, and corrects its own retrieval before generating responses. When confidence is low, the system automatically identifies failure modes and applies targeted correction strategies.

### What Makes This Different?

| Standard RAG | Self-Healing RAG |
|--------------|------------------|
| Retrieve → Generate | Retrieve → **Evaluate Confidence** → **Diagnose Issues** → **Correct** → Generate |
| Hopes for good results | Measures retrieval quality with multi-signal scoring |
| Hallucinates when context is poor | Abstains or self-corrects when uncertain |
| Single retrieval strategy | Hybrid search with fallback strategies |

---

## Key Features

### Agentic Self-Correction Loop

The core innovation is a LangGraph state machine that implements intelligent retrieval correction:

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Retrieve   │────▶│ Estimate         │────▶│   Route     │
│  (Hybrid)   │     │ Confidence       │     │             │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │                               │                               │
                    ▼                               ▼                               ▼
            ┌───────────────┐             ┌─────────────────┐             ┌─────────────────┐
            │ HIGH: Generate│             │ MEDIUM: Generate│             │ LOW: Diagnose   │
            │ (Confident)   │             │ (With Hedging)  │             │ & Correct       │
            └───────────────┘             └─────────────────┘             └────────┬────────┘
                                                                                   │
                                                                                   ▼
                                                                          ┌─────────────────┐
                                                                          │ Failure Modes:  │
                                                                          │ • AMBIGUITY     │
                                                                          │ • VOCAB_MISMATCH│
                                                                          │ • INFO_SCATTER  │
                                                                          │ • KNOWLEDGE_GAP │
                                                                          └─────────────────┘
```

**Confidence Estimation** uses four signals:
1. **Top Score** - Best retrieval similarity (baseline quality)
2. **Score Dropoff** - Gap between top results (specificity indicator)
3. **Inter-chunk Coherence** - Semantic consistency across results
4. **Query Coverage** - LLM-assessed relevance to query intent

### Hybrid Search with Reciprocal Rank Fusion

Three complementary search strategies combined via RRF:

| Strategy | Weight | Purpose |
|----------|--------|---------|
| **Vector Search** | 40% | Semantic similarity via BGE-M3 embeddings |
| **Keyword Search** | 20% | Exact term matching via PostgreSQL tsvector |
| **HyDE Search** | 40% | Match against hypothetical question embeddings |

### Multi-Workspace Knowledge Bases

Organize documents into separate workspaces, each with isolated knowledge:

- Create workspaces with custom names, colors, and icons
- Upload documents to specific workspaces
- Queries are scoped to the selected workspace
- Switch between workspaces or search across all

### Glassmorphism Frontend

A dark-mode UI with cinematic reasoning visualization:

- **Animated mesh gradient background** with noise overlay
- **5-phase reasoning trace** animation (Vectorizing → Searching → Confidence → Self-Healing → Generating)
- **SSE streaming** with typewriter text reveal
- **Inline citations** with hover tooltips showing source excerpts
- **Drag-and-drop** document upload

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 15)                        │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  Workspace  │  │   Chat Input    │  │   Reasoning Trace      │  │
│  │  Selector   │  │   + Streaming   │  │   Animation            │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ SSE / REST
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Backend (FastAPI)                            │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  Ingestion  │  │   RAG Agent     │  │   Search Engine         │  │
│  │  Pipeline   │  │   (LangGraph)   │  │   (Hybrid + Rerank)     │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PostgreSQL + pgvector                           │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ Workspaces  │  │   Documents     │  │   Chunks + Embeddings   │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM** | Claude (agents), Gemini (generation) |
| **Embeddings** | BAAI/bge-m3 (1024-dim dense) |
| **Reranker** | BAAI/bge-reranker-v2-m3 |
| **Orchestration** | LangGraph state machine |
| **Backend** | FastAPI, SQLAlchemy 2.0 (async) |
| **Database** | PostgreSQL + pgvector (Neon) |
| **Frontend** | Next.js 15, Tailwind CSS, Framer Motion |
| **Observability** | OpenTelemetry (optional) |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL with pgvector extension (or [Neon](https://neon.tech) cloud)
- API keys: Anthropic (Claude), Google (Gemini)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/Gustav-Proxi/RAG.git
cd RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your credentials:
#   DATABASE_URL=postgresql+asyncpg://user:pass@host/db
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=...

# Start the server (auto-creates tables on first run)
python -m uvicorn src.api.main:app --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Open **http://localhost:3000** to access the UI.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with DB stats |
| `POST` | `/ingest` | Upload and process a document |
| `POST` | `/query` | RAG query with full pipeline |
| `POST` | `/query/stream` | Streaming query via SSE |
| `GET` | `/documents` | List all documents |
| `DELETE` | `/documents/{id}` | Delete a document |
| `GET` | `/workspaces` | List workspaces |
| `POST` | `/workspaces` | Create a workspace |
| `POST` | `/search` | Direct search without generation |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "workspace_id": "optional-uuid"}'
```

Response includes:
- `response` - Generated answer
- `citations` - Source chunks with relevance scores
- `confidence_score` - 0-1 confidence rating
- `confidence_band` - high/medium/low
- `correction_attempts` - Number of self-correction loops

---

## Project Structure

```
RAG/
├── src/
│   ├── api/              # FastAPI endpoints
│   │   ├── main.py       # Routes and app setup
│   │   └── middleware.py # Security middleware
│   ├── agents/           # LangGraph self-correction
│   │   ├── graph.py      # State machine definition
│   │   ├── diagnosis.py  # Failure mode detection
│   │   └── correction.py # Correction strategies
│   ├── retrieval/        # Search and ranking
│   │   ├── search.py     # Hybrid search (vector + keyword + HyDE)
│   │   ├── reranker.py   # Cross-encoder reranking
│   │   └── confidence.py # Multi-signal scoring
│   ├── ingestion/        # Document processing
│   │   ├── parser.py     # PDF/DOCX/MD parsing
│   │   ├── chunker.py    # Hierarchical chunking
│   │   └── pipeline.py   # Orchestration
│   ├── generation/       # Response generation
│   ├── validation/       # Claim validation (NLI)
│   ├── embeddings/       # BGE-M3 embeddings
│   ├── database/         # SQLAlchemy models
│   └── config/           # Settings and constants
├── frontend/
│   └── src/
│       ├── app/          # Next.js pages
│       └── components/   # React components
├── tests/                # Unit and integration tests
└── scripts/              # CLI utilities
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `EMBEDDING_MODEL` | No | Default: `BAAI/bge-m3` |
| `EMBEDDING_DEVICE` | No | `cpu`, `cuda`, or `mps` |
| `AGENT_MODEL` | No | Default: `claude-haiku-4-5-20251001` |
| `GENERATION_MODEL` | No | Default: `gemini-2.0-flash` |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_confidence.py -v
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.
