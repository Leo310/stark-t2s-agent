# STaRK-Prime Text-to-SQL/SPARQL Agent

A LangChain v1 agent that converts natural language queries into SQL or SPARQL for the [STaRK-Prime](https://stark.stanford.edu/dataset_prime.html) biomedical knowledge base.

## Overview

This project implements a text-to-SQL/SPARQL agent using LangChain v1 (with LangGraph) that:

- **Two-stage query approach**: 
  1. **Entity Resolution**: Semantic search to find entity IDs from natural language
  2. **Query Execution**: SQL or SPARQL queries using resolved entity IDs
- **Dual representation**: Materializes STaRK-Prime into both SQL (relational) and RDF (semantic) formats
- **Typed schema**: Prime-specific tables for 10 entity types and 18 relation types
- **Benchmarking**: Evaluate against STaRK-Prime QA datasets (synthesized + human-generated)
- **Docker support**: Use PostgreSQL + Fuseki for production-ready persistent storage
- **Observability**: Langfuse v3 integration for tracing and debugging

## STaRK-Prime Dataset

STaRK-Prime targets complex biomedical inquiries with:
- **10 entity types**: disease, drug, gene/protein, molecular function, pathway, etc.
- **18 relation types**: associated_with, indication, contraindication, side_effect, parent-child, etc.
- **129,375 entities** and **8,100,498 relations**

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd stark-t2s-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with pip
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```env
# Choose your LLM provider: openai or openrouter
LLM_PROVIDER=openai

# For OpenAI (default)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini

# For OpenRouter (alternative)
# OPENROUTER_API_KEY=your-openrouter-api-key
# OPENROUTER_MODEL=openai/gpt-4o-mini
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional: override local cache dir (downloads + benchmark outputs)
# STARK_CACHE_DIR=/absolute/path/to/cache
```

## Quick Start (with Docker - Recommended)

### 1. Start Docker containers

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL** on port 5432 (for SQL queries)
- **Apache Jena Fuseki** on port 3030 (for SPARQL queries)
- **Qdrant** on port 6333 (for entity resolution)

### 2. Build all data stores

```bash
python scripts/build_prime_stores.py
```

This will:
- Download `processed.zip` from HuggingFace (~100MB)
- Parse node/edge types and metadata
- Load data into PostgreSQL (typed tables)
- Load data into Fuseki (RDF triples)
- Create Qdrant vector index (entity embeddings)

### 3. Interactive demo

```bash
python scripts/demo_chat.py
```

### 4. Run benchmark

```bash
python -m stark_prime_t2s.benchmark.run_prime --split synthesized --limit 10
```

## Project Structure

```
stark-t2s-agent/
├── docker-compose.yml            # PostgreSQL + Fuseki containers
├── pyproject.toml
├── README.md
├── .env                          # API keys and config (create this)
├── docker-compose.yml            # PostgreSQL, Fuseki, Qdrant services
├── scripts/
│   ├── build_prime_stores.py     # Build all data stores
│   └── demo_chat.py              # Interactive chat demo
└── src/stark_prime_t2s/
    ├── config.py                 # Docker services config
    ├── data/
    │   ├── download_prime.py     # Download STaRK-Prime from HF
    │   └── parse_prime_processed.py
    ├── materialize/
    │   ├── postgres_prime.py     # PostgreSQL schema + loader
    │   └── fuseki_prime.py       # Fuseki SPARQL endpoint loader
    ├── tools/
    │   ├── entity_resolver.py    # Qdrant vector search
    │   └── execute_query.py      # SQL/SPARQL execution tool
    ├── agent/
    │   └── agent.py              # LangChain agent wiring
    └── benchmark/
        └── run_prime.py          # Benchmark runner
```

## Architecture

The agent uses a **two-stage approach** for better entity matching:

```
┌──────────────────────────┐
│      User Query          │
│ "What drugs treat        │
│  Alzheimer's disease?"   │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   LangChain Agent        │
│   (OpenAI/OpenRouter)    │
└───────────┬──────────────┘
            │
  ┌─────────┴──────────┐
  │                    │
  ▼                    ▼
┌────────────────┐  ┌────────────────────┐
│ STAGE 1:       │  │ STAGE 2:           │
│ Entity Search  │  │ Query Execution    │
│                │  │                    │
│ Vector search  │  │ SQL or SPARQL      │
│ → "Alzheimer"  │  │ using resolved IDs │
│ → ID: 28780    │  │                    │
└────────────────┘  └─────────┬──────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
    ┌───────────────┐                  ┌───────────────┐
    │  PostgreSQL   │                  │    Fuseki     │
    │  (SQL)        │                  │   (SPARQL)    │
    └───────────────┘                  └───────────────┘
```

### Tools

| Tool | Purpose |
|------|---------|
| `search_entities_tool` | Semantic vector search to find entity IDs from natural language |
| `execute_query_tool` | Execute SQL or SPARQL queries against the knowledge base |

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | SQL queries (typed tables) |
| Fuseki | 3030 | SPARQL queries (RDF graph) |
| Qdrant | 6333 | Vector search for entity resolution |

Access Fuseki web UI at: http://localhost:3030
Access Qdrant dashboard at: http://localhost:6333/dashboard

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider: `openai` or `openrouter` |
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-5-mini` | Model to use |
| `OPENROUTER_API_KEY` | (optional) | OpenRouter API key |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | OpenRouter model identifier |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `STARK_CACHE_DIR` | *(unset)* | Override local cache path for downloads + benchmark outputs |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `FUSEKI_HOST` | `localhost` | Fuseki host |
| `FUSEKI_PORT` | `3030` | Fuseki port |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `LANGFUSE_ENABLED` | `true` | Enable Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | - | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | - | Langfuse public key |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse base URL |

## Observability with Langfuse

This project uses [Langfuse v3](https://langfuse.com/docs/observability/sdk/upgrade-path) for observability and tracing.

### Setup

1. Create a free account at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a project and get your API keys
3. Add to your `.env` file:

```env
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
# Optional: for US region use https://us.cloud.langfuse.com
# LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### What's Traced

- Agent invocations
- Tool calls (SQL/SPARQL queries)
- Model responses
- Latency and token usage

### Viewing Traces

Go to your Langfuse dashboard to see:
- All agent runs with full conversation history
- Query execution details
- Performance metrics
- Cost tracking

## License

MIT License. STaRK-Prime dataset is licensed under CC-BY-4.0.
