#!/usr/bin/env python3
"""Build PostgreSQL, Fuseki, and Qdrant stores from STaRK-Prime data.

This script:
1. Downloads the STaRK-Prime processed data from HuggingFace
2. Parses the node/edge types and metadata
3. Creates PostgreSQL database (SQL queries)
4. Creates Fuseki RDF graph (SPARQL queries)
5. Creates Qdrant vector index (entity resolution)

Usage:
    # Start Docker services first:
    docker-compose up -d

    # Build all stores:
    python scripts/build_prime_stores.py

    # Force rebuild:
    python scripts/build_prime_stores.py --force

    # Skip specific stores:
    python scripts/build_prime_stores.py --skip-sql --skip-rdf  # Only build Qdrant
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stark_prime_t2s.scripts.build_stores import main

if __name__ == "__main__":
    main()
