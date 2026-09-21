#!/usr/bin/env bash
# Full benchmark sweep against live TigerGraph. RAG first because it is the
# long pole: Groq's on-demand tier allows 8,000 tokens/minute and a RAG
# request is ~4.5k, so RAG paces the whole sweep at roughly two questions
# per minute. The other pipelines are nearly free by comparison.
cd "$(dirname "$0")"
export EMBEDDING_PROVIDER=ollama PYTHONUNBUFFERED=1
set -x
python -u -m src.eval.run_benchmark --suffix _tg_public_full            > outputs/run_tg_public_full.log 2>&1
python -u -m src.eval.run_benchmark --stress --suffix _tg_stress_full   > outputs/run_tg_stress_full.log 2>&1
python -u -m src.eval.run_benchmark --hidden --pipelines graphrag,agentic_graphrag --suffix _tg_hidden > outputs/run_tg_hidden.log 2>&1
python -u -m src.eval.run_benchmark --ablation --pipelines graphrag,agentic_graphrag --suffix _tg_public_ablation > outputs/run_tg_ablation.log 2>&1
python -u -m src.eval.run_benchmark --stress --ablation --pipelines graphrag,agentic_graphrag --suffix _tg_stress_ablation > outputs/run_tg_stress_ablation.log 2>&1
echo SWEEP_COMPLETE
