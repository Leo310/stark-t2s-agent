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
OPENAI_MODEL=gpt-5-mini

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
- **Apache Jena Fuseki** on port 3031 (for SPARQL queries)
- **Qdrant** on port 6333 (for entity resolution)

### 2. Build all data stores

```bash
build-prime-stores
```

This will:
- Download `processed.zip` from HuggingFace (~100MB)
- Parse node/edge types and metadata
- Load data into PostgreSQL (typed tables)
- Load data into Fuseki (RDF triples)
- Create Qdrant vector index (entity embeddings)

### 3. Interactive demo

```bash
demo-chat
```

Select an agent mode when prompted, or pass it directly:

```bash
demo-chat --agent sql
demo-chat --agent sparql
```

### 4. Run benchmark

The benchmark runner uses Langfuse datasets. Make sure these are set in `.env`:

```env
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
```

List available datasets:

```bash
run-benchmark --list-datasets
```

Run a benchmark (will prompt for dataset if not provided):

```bash
run-benchmark --dataset stark_prime_synth --concurrency 2
```

To benchmark SQL-only or SPARQL-only agents:

```bash
run-benchmark --agent sql
run-benchmark --agent sparql
```


## Example benchmark results (gpt-5-mini) — STaRK-Prime human dataset

The following table shows sample benchmark results captured from a Langfuse run using the `gpt-5-mini` model on the STaRK-Prime human dataset (`stark_prime_human`). Metrics are reported by the Langfuse benchmark runner.

| Name                    | Latency       | Total Cost       | Hit@1         | Hit@5         | MRR         | Recall        |
|-------------------------|---------------:|------------------:|--------------:|--------------:|------------:|---------------:|
| Search only             | 32.03s         | $0.623994         | 32.11%        | 37.61%        | 34.33%      | 27.04%        |
| SQL only                | 1m 4s          | $1.226361         | 44.00%        | 57.00%        | 48.66%      | 45.49%        |
| SPARQL only             | 56.43s         | $1.09769          | 47.17%        | 57.55%        | 50.81%      | 44.64%        |
| SQL + SPARQL            | 1m 10s         | $1.106999         | 46.23%        | 53.77%        | 49.63%      | 44.42%        |

Note: these are example results meant to illustrate relative performance between agent modes. Re-run `run-benchmark --dataset stark_prime_human` in your environment to produce up-to-date numbers for your setup.

Metric definitions

- Hit@1 / Hit@5: Percentage of queries where the correct result appears in the top 1 or top 5 returned answers, respectively.
- MRR: Mean Reciprocal Rank — average of the reciprocal ranks of the first correct answer (higher is better).
- Recall: Fraction of relevant items retrieved for the query (higher is better).
- Latency: Average end-to-end time for an agent run (includes entity resolution + query execution).
- Total Cost: Sum of trace/model costs reported by Langfuse for the benchmark run.


## Project Structure

```
stark-t2s-agent/
├── docker-compose.yml            # PostgreSQL, Fuseki, Qdrant services
├── pyproject.toml
├── README.md
├── .env                          # API keys and config (create this)
├── scripts/
│   ├── build_prime_stores.py     # Wrapper for build-prime-stores CLI
│   └── demo_chat.py              # Wrapper for demo-chat CLI
└── src/stark_prime_t2s/
    ├── config.py                 # Docker services + env config
    ├── dataset/
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
    ├── benchmark/
    │   └── run_prime.py          # Benchmark runner
    └── scripts/
        ├── build_stores.py       # Build stores CLI implementation
        └── demo_chat.py          # Demo chat CLI implementation
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
| Fuseki | 3031 | SPARQL queries (RDF graph) |
| Qdrant | 6333 | Vector search for entity resolution |

Access Fuseki web UI at: http://localhost:3031
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
| `POSTGRES_DB` | `stark_prime` | PostgreSQL database |
| `POSTGRES_USER` | `stark` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `stark_password` | PostgreSQL password |
| `FUSEKI_HOST` | `localhost` | Fuseki host |
| `FUSEKI_PORT` | `3031` | Fuseki port |
| `FUSEKI_DATASET` | `prime` | Fuseki dataset name |
| `FUSEKI_ADMIN_PASSWORD` | `admin` | Fuseki admin password |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `stark_entities` | Qdrant collection (default index) |
| `QDRANT_COLLECTION_FULL` | `stark_entities_full` | Qdrant collection (full index) |
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
