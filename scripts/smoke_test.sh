#!/usr/bin/env bash
# Fast confidence check (~1 min, no API calls needed).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.ingestion.build_graph --structured-only
python -m src.eval.stress_set --n 10
python -m src.eval.run_benchmark --limit 5 --pipelines graphrag,agentic_graphrag
python -m pytest tests/ -q
