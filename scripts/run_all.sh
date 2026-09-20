#!/usr/bin/env bash
# End-to-end reproduction. Safe to re-run; every step is idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/6  TigerGraph =="
if python -m scripts.setup_tigergraph --check 2>/dev/null; then
  python -m scripts.setup_tigergraph
else
  echo "   No TigerGraph configured — using the offline store."
  echo "   See README 'Connecting TigerGraph' to run against a live instance."
  python -m src.ingestion.build_graph
fi

echo "== 2/6  Generate the chained-reasoning stress set =="
python -m src.eval.stress_set --n 60

echo "== 3/6  Benchmark: 100 public questions =="
python -m src.eval.run_benchmark

echo "== 4/6  Benchmark: 60 stress questions =="
python -m src.eval.run_benchmark --stress

echo "== 5/6  Benchmark: 50 hidden questions (submission) =="
python -m src.eval.run_benchmark --hidden

echo "== 6/6  Ablation: regex fast path + triage disabled =="
python -m src.eval.run_benchmark --ablation
python -m src.eval.run_benchmark --stress --ablation

echo
echo "Done. Dashboard: python -m http.server 8000  ->  http://localhost:8000/dashboard/"
