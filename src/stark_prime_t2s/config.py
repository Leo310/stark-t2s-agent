"""Configuration for STaRK-Prime T2S Agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Local cache paths (NOT in the repo)
# ---------------------------------------------------------------------------

# Store downloaded datasets and benchmark outputs outside the git repo.
# Override with STARK_CACHE_DIR if you want a different location.
#
# Default on macOS/Linux: ~/.cache/stark-t2s-agent
CACHE_DIR = Path(os.getenv("STARK_CACHE_DIR", Path.home() / ".cache" / "stark-t2s-agent"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRIME_CACHE_DIR = CACHE_DIR / "prime"
PRIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# STaRK-Prime processed data directory (downloaded during build)
PRIME_PROCESSED_DIR = PRIME_CACHE_DIR / "processed"

# ---------------------------------------------------------------------------
# STaRK-Prime download URLs
# ---------------------------------------------------------------------------

# SKB (Semi-structured Knowledge Base) - processed bundle
PRIME_SKB_URL = (
    "https://huggingface.co/datasets/snap-stanford/stark/resolve/main/"
    "skb/prime/processed.zip?download=true"
)

# QA datasets (synthesized and human-generated)
PRIME_QA_SYNTH_URL = (
    "https://huggingface.co/datasets/snap-stanford/stark/resolve/main/"
    "qa/prime/stark_qa/stark_qa.csv?download=true"
)

PRIME_QA_HUMAN_URL = (
    "https://huggingface.co/datasets/snap-stanford/stark/resolve/main/"
    "qa/prime/stark_qa/stark_qa_human_generated_eval.csv?download=true"
)

# Local paths for QA datasets (for benchmarking)
QA_SYNTH_PATH = PRIME_CACHE_DIR / "stark_qa_synth.csv"
QA_HUMAN_PATH = PRIME_CACHE_DIR / "stark_qa_human.csv"

# Benchmark output directory
BENCHMARK_OUTPUT_DIR = PRIME_CACHE_DIR / "benchmark_results"
BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM Provider Configuration
# ---------------------------------------------------------------------------

# Provider selection: "openai" or "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# ---------------------------------------------------------------------------
# OpenAI Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# ---------------------------------------------------------------------------
# OpenRouter Configuration
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------

MAX_QUERY_ROWS = 50
MAX_QUERY_RETRIES = 3
MAX_AGENT_ITERATIONS = 25

# ---------------------------------------------------------------------------
# RDF Namespace
# ---------------------------------------------------------------------------

STARK_PRIME_NS = "http://stark.stanford.edu/prime/"

# ---------------------------------------------------------------------------
# Docker Services Configuration
# ---------------------------------------------------------------------------

# PostgreSQL
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "stark_prime")
POSTGRES_USER = os.getenv("POSTGRES_USER", "stark")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "stark_password")
POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Fuseki SPARQL endpoint
FUSEKI_HOST = os.getenv("FUSEKI_HOST", "localhost")
FUSEKI_PORT = int(os.getenv("FUSEKI_PORT", "3031"))
FUSEKI_DATASET = os.getenv("FUSEKI_DATASET", "prime")
FUSEKI_QUERY_URL = f"http://{FUSEKI_HOST}:{FUSEKI_PORT}/{FUSEKI_DATASET}/sparql"
FUSEKI_UPDATE_URL = f"http://{FUSEKI_HOST}:{FUSEKI_PORT}/{FUSEKI_DATASET}/update"
FUSEKI_DATA_URL = f"http://{FUSEKI_HOST}:{FUSEKI_PORT}/{FUSEKI_DATASET}/data"
FUSEKI_ADMIN_PASSWORD = os.getenv("FUSEKI_ADMIN_PASSWORD", "admin")

# Qdrant vector database
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "stark_entities")
QDRANT_COLLECTION_FULL = os.getenv("QDRANT_COLLECTION_FULL", "stark_entities_full")

# ---------------------------------------------------------------------------
# Langfuse Observability (v3)
# ---------------------------------------------------------------------------

LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
