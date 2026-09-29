#!/usr/bin/env bash
# Final benchmark sweep: temperature 0, LLM planner in every pipeline,
# live TigerGraph. Checkpointed, so an interruption costs one question.
cd "$(dirname "$0")"
unset TG_HOST
export LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama LLM_TOKENS_PER_MINUTE=0 PYTHONUNBUFFERED=1
python -u -m src.eval.run_benchmark --suffix _final100  > outputs/run_final100.log  2>&1
python -u -m src.eval.run_benchmark --stress --suffix _finalstress > outputs/run_finalstress.log 2>&1
python -u -m src.eval.run_benchmark --fast-path --suffix _final_fastpath > outputs/run_final_fastpath.log 2>&1
echo SWEEP_DONE
