#!/usr/bin/env python3
"""Analyze benchmark run artifacts and generate HTML reports.

This script auto-discovers benchmark_runs* directories and generates
analysis reports with interactive visualizations.

Usage:
    # Auto-discover all benchmark directories and generate combined report:
    python scripts/analyze_benchmark.py

    # Filter by dataset name (substring match):
    python scripts/analyze_benchmark.py human        # Only human-generated
    python scripts/analyze_benchmark.py synth        # Only synthesized
    python scripts/analyze_benchmark.py human synth  # Both explicitly

    # Specify output directory:
    python scripts/analyze_benchmark.py --output-dir ./reports
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import directly from the module file to avoid __init__.py dependencies
import importlib.util

module_path = (
    Path(__file__).parent.parent / "src" / "stark_prime_t2s" / "benchmark" / "analyze_local_runs.py"
)
spec = importlib.util.spec_from_file_location("analyze_local_runs", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {module_path}")
analyze_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_module)

if __name__ == "__main__":
    analyze_module.main()
