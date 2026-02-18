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

Copy the example environment file and fill in your API keys:

```bash
cp .example.env .env
```

Then edit `.env` with your configuration:

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

First, download the QA datasets (stored in `~/.cache/stark-t2s-agent/prime/`):

```bash
python -c "from stark_prime_t2s.dataset import download_prime_qa; download_prime_qa()"
```

Run benchmarks locally:

```bash
run-benchmark --backend local --dataset-file ~/.cache/stark-t2s-agent/prime/stark_qa_human.csv
```

Specify an agent mode:

```bash
run-benchmark --backend local --dataset-file ~/.cache/stark-t2s-agent/prime/stark_qa_human.csv --agent sql
run-benchmark --backend local --dataset-file ~/.cache/stark-t2s-agent/prime/stark_qa_human.csv --agent sparql
```

Additional options:

```bash
# Run with concurrency
run-benchmark --backend local --dataset-file ~/.cache/stark-t2s-agent/prime/stark_qa_human.csv --concurrency 4

# Limit number of items
run-benchmark --backend local --dataset-file ~/.cache/stark-t2s-agent/prime/stark_qa_human.csv --dataset-limit 50

# Custom run name and output directory
run-benchmark --backend local --dataset-file ~/.cache/stark-t2s-agent/prime/stark_qa_human.csv --run-name my-experiment --output-dir my_runs
```

Results are saved to `benchmark_runs/` by default (or the directory specified with `--output-dir`).

### 5. Analyze benchmark results

After running benchmarks, you can analyze the results:

```bash
analyze-benchmark
```

This generates an HTML report with metrics comparison across different runs.

## Example benchmark results (gpt-5-mini) — STaRK-Prime human dataset

The following table shows benchmark results using the `gpt-5-mini` model on the STaRK-Prime human dataset (`stark_prime_human`). Metrics are averaged across 6 runs.

| Name         | Latency |  Hit@1 |  Hit@5 |    MRR | Recall |
| ------------ | ------: | -----: | -----: | -----: | -----: |
| Search only  |   23.8s | 34.10% | 42.05% | 37.12% | 30.94% |
| SQL only     |   47.6s | 44.94% | 57.82% | 49.92% | 46.46% |
| SPARQL only  |   42.5s | 46.70% | 56.53% | 50.62% | 47.17% |
| SQL + SPARQL |   39.4s | 46.94% | 59.79% | 52.12% | 47.04% |

Note: these are example results meant to illustrate relative performance between agent modes. Re-run `run-benchmark --dataset stark_prime_human` in your environment to produce up-to-date numbers for your setup.

Metric definitions

- Hit@1 / Hit@5: Percentage of queries where the correct result appears in the top 1 or top 5 returned answers, respectively.
- MRR: Mean Reciprocal Rank — average of the reciprocal ranks of the first correct answer (higher is better).
- Recall: Fraction of relevant items retrieved for the query (higher is better).
- Latency: Average end-to-end time for an agent run (includes entity resolution + query execution).

## Project Structure

```
stark-t2s-agent/
├── docker-compose.yml            # PostgreSQL, Fuseki, Qdrant services
├── pyproject.toml
├── README.md
├── .env                          # API keys and config (create from .example.env)
├── .example.env                  # Example environment variables template
├── scripts/
│   ├── analyze_benchmark.py      # Wrapper for analyze-benchmark CLI
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
    │   ├── run_prime.py          # Benchmark runner
    │   ├── analyze_local_runs.py # Analyze benchmark results
    │   └── report_template.html  # HTML report template
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

| Tool                   | Purpose                                                         |
| ---------------------- | --------------------------------------------------------------- |
| `search_entities_tool` | Semantic vector search to find entity IDs from natural language |
| `execute_query_tool`   | Execute SQL or SPARQL queries against the knowledge base        |

## Docker Services

| Service    | Port | Purpose                             |
| ---------- | ---- | ----------------------------------- |
| PostgreSQL | 5432 | SQL queries (typed tables)          |
| Fuseki     | 3031 | SPARQL queries (RDF graph)          |
| Qdrant     | 6333 | Vector search for entity resolution |

Access Fuseki web UI at: http://localhost:3031
Access Qdrant dashboard at: http://localhost:6333/dashboard

## Environment Variables

| Variable                            | Default                                  | Description                                                                     |
| ----------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------- |
| `LLM_PROVIDER`                      | `openai`                                 | LLM provider: `openai` or `openrouter`                                          |
| `OPENAI_API_KEY`                    | (required)                               | OpenAI API key                                                                  |
| `OPENAI_MODEL`                      | `gpt-5-mini`                             | Model to use                                                                    |
| `OPENROUTER_API_KEY`                | (optional)                               | OpenRouter API key                                                              |
| `OPENROUTER_MODEL`                  | `openai/gpt-4o-mini`                     | OpenRouter model identifier                                                     |
| `OPENROUTER_BASE_URL`               | `https://openrouter.ai/api/v1`           | OpenRouter API base URL                                                         |
| `STARK_CACHE_DIR`                   | _(unset)_                                | Override local cache path for downloads + benchmark outputs                     |
| `POSTGRES_HOST`                     | `localhost`                              | PostgreSQL host                                                                 |
| `POSTGRES_PORT`                     | `5432`                                   | PostgreSQL port                                                                 |
| `POSTGRES_DB`                       | `stark_prime`                            | PostgreSQL database                                                             |
| `POSTGRES_USER`                     | `stark`                                  | PostgreSQL user                                                                 |
| `POSTGRES_PASSWORD`                 | `stark_password`                         | PostgreSQL password                                                             |
| `FUSEKI_HOST`                       | `localhost`                              | Fuseki host                                                                     |
| `FUSEKI_PORT`                       | `3031`                                   | Fuseki port                                                                     |
| `FUSEKI_DATASET`                    | `prime`                                  | Fuseki dataset name                                                             |
| `FUSEKI_ADMIN_PASSWORD`             | `admin`                                  | Fuseki admin password                                                           |
| `QDRANT_HOST`                       | `localhost`                              | Qdrant host                                                                     |
| `QDRANT_PORT`                       | `6333`                                   | Qdrant port                                                                     |
| `QDRANT_COLLECTION`                 | `stark_entities_full`                    | Qdrant collection name                                                          |
| `EMBEDDING_PROVIDER`                | `openai`                                 | Embedding provider: `openai`, `openrouter`, `huggingface`, `azure`, or `cohere` |
| `EMBEDDING_MODEL`                   | `text-embedding-3-small`                 | Embedding model (provider-specific)                                             |
| `EMBEDDING_BASE_URL`                | _(unset)_                                | Custom base URL for embeddings (self-hosted or proxy)                           |
| `HUGGINGFACE_API_KEY`               | _(optional)_                             | HuggingFace API key (if using huggingface provider)                             |
| `HUGGINGFACE_EMBEDDING_MODEL`       | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model                                                     |
| `AZURE_OPENAI_API_KEY`              | _(optional)_                             | Azure OpenAI API key (if using azure provider)                                  |
| `AZURE_OPENAI_ENDPOINT`             | _(optional)_                             | Azure OpenAI endpoint URL                                                       |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | _(optional)_                             | Azure OpenAI embedding deployment name                                          |
| `COHERE_API_KEY`                    | _(optional)_                             | Cohere API key (if using cohere provider)                                       |
| `COHERE_EMBEDDING_MODEL`            | `embed-english-v3`                       | Cohere embedding model                                                          |
| `LANGFUSE_ENABLED`                  | `true`                                   | Enable Langfuse tracing                                                         |
| `LANGFUSE_SECRET_KEY`               | -                                        | Langfuse secret key                                                             |
| `LANGFUSE_PUBLIC_KEY`               | -                                        | Langfuse public key                                                             |
| `LANGFUSE_BASE_URL`                 | `https://cloud.langfuse.com`             | Langfuse base URL                                                               |
| `MLFLOW_ENABLED`                    | `false`                                  | Enable MLflow tracing (mutually exclusive with Langfuse)                        |
| `DATABRICKS_HOST`                   | -                                        | Databricks workspace host                                                       |
| `DATABRICKS_TOKEN`                  | -                                        | Databricks API token                                                            |
| `MLFLOW_TRACKING_URI`               | `databricks`                             | MLflow tracking URI                                                             |
| `MLFLOW_REGISTRY_URI`               | -                                        | MLflow registry URI (Unity Catalog)                                             |
| `MLFLOW_EXPERIMENT_ID`              | -                                        | MLflow experiment ID                                                            |

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

## Observability with MLflow (Databricks)

This project can send LangChain traces to a Databricks-hosted MLflow Tracking Server
using `mlflow.langchain.autolog()`. MLflow and Langfuse are mutually exclusive; only
enable one at a time.

### Setup

1. Generate a Databricks API token and copy the host URL
2. Add these to your `.env` file:

```env
# Disable Langfuse when using MLflow
LANGFUSE_ENABLED=false
MLFLOW_ENABLED=true

# Databricks + MLflow
DATABRICKS_TOKEN=your_token
DATABRICKS_HOST=https://adb-1234567890.0.azuredatabricks.net
MLFLOW_TRACKING_URI=databricks
MLFLOW_REGISTRY_URI=databricks-uc
MLFLOW_EXPERIMENT_ID=123456789012345
```

### Viewing Traces

Open the MLflow UI in your Databricks workspace and select the experiment ID you
configured to view runs and traces.

### Benchmark datasets

When running `run-benchmark` with MLflow enabled, the benchmark uses MLflow evaluation
datasets by name and creates a single MLflow Evaluation run that contains a trace for
each dataset record. Each record must include:

- `inputs.query`
- `expectations.expected_entity_ids`

## License

MIT License. STaRK-Prime dataset is licensed under CC-BY-4.0.
