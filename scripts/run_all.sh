#!/usr/bin/env bash
# Reproduce every number in RESULTS.md from scratch.
#
# Safe to re-run: each step is idempotent and every benchmark is checkpointed,
# so an interruption costs one question rather than the run. Re-running after a
# stop resumes where it left off.
#
# Expect roughly 3 hours end to end with a local model. Most of that is the two
# controlled comparisons, which put an LLM call in front of every pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/7  TigerGraph =="
if python -m scripts.setup_tigergraph --check >/dev/null 2>&1; then
  python -m scripts.setup_tigergraph
else
  echo "   No reachable TigerGraph — using the offline store."
  echo "   See README 'Connecting TigerGraph'. Note the benchmarks below will"
  echo "   REFUSE to run if TG_HOST is set but unreachable, rather than quietly"
  echo "   producing numbers that look like TigerGraph results."
  python -m src.ingestion.build_graph
fi

echo "== 2/7  Generate the chained-reasoning set =="
python -m src.eval.stress_set --n 60

# The two configurations answer different questions, and both are reported.
#   controlled  — every pipeline compiles with the same LLM. Isolates whether
#                 ADAPTIVITY helps, by denying our pipelines a prior on the
#                 question templates that RAG never gets.
#   production  — regex fast path on, as you would actually deploy it.
#                 Answers how well the system works.

echo "== 3/7  Controlled comparison: 100 provided questions =="
python -m src.eval.run_benchmark --suffix _final100

echo "== 4/7  Controlled comparison: 60 chained questions =="
python -m src.eval.run_benchmark --stress --suffix _finalstress

echo "== 5/7  Production configuration: 100 provided questions =="
python -m src.eval.run_benchmark --fast-path --suffix _final_fastpath

echo "== 6/7  Hidden set (submission) =="
# RAG is excluded: the submission asks for our system's answers, tokens and
# agentic trace, and RAG is a baseline rather than the system.
python -m src.eval.run_benchmark --hidden --pipelines graphrag,agentic_graphrag \
       --suffix _tg_hidden
python -m scripts.export_submission --suffix _tg_hidden

echo "== 7/7  Regenerate RESULTS.md from the outputs =="
python -m scripts.make_report

echo
echo "Done."
echo "  RESULTS.md                        every number, with run provenance"
echo "  outputs/submission_hidden.jsonl   hidden-set answers + tokens + traces"
echo "  dashboard:  python -m http.server 8000  ->  localhost:8000/dashboard/"
echo
echo "Optional — the agent without its cheap deterministic triage, which"
echo "isolates how much that first pass contributes:"
echo "  python -m src.eval.run_benchmark --ablation --suffix _no_triage"
