#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

"""Wrapper for the interactive demo chat.

Run with:
  python scripts/demo_chat.py

This delegates to `stark_prime_t2s.scripts.demo_chat`, which uses Docker
services (PostgreSQL + Fuseki + Qdrant).
"""

from stark_prime_t2s.scripts.demo_chat import main


if __name__ == "__main__":
    main()

