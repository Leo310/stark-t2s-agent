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
    "https://stark.stanford.edu/data/primekg/stark_qa_human_generated_eval.csv"
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
# Embedding Provider Configuration
# ---------------------------------------------------------------------------

# Provider selection: "openai", "openrouter", "huggingface", "azure", or "cohere"
# If not set, defaults to the same provider as LLM_PROVIDER
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

# Embedding model (provider-specific)
# OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
# OpenRouter: Uses OpenAI-compatible models via OpenRouter
# HuggingFace: e.g., "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Custom base URL for embeddings (for self-hosted or proxy endpoints)
# Examples: "http://localhost:11434", "https://custom-api.example.com"
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")

# HuggingFace-specific settings (only used if EMBEDDING_PROVIDER="huggingface")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Azure OpenAI settings (only used if EMBEDDING_PROVIDER="azure")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Cohere settings (only used if EMBEDDING_PROVIDER="cohere")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3")

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
MAX_AGENT_ITERATIONS = 100

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

# ---------------------------------------------------------------------------
# MLflow Observability (Databricks-hosted)
# ---------------------------------------------------------------------------

MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI")
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")
