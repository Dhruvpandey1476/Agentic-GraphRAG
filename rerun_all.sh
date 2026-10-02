#!/usr/bin/env bash
# Re-measure everything on the current code, one model throughout.
#
# Today's fixes changed behaviour for all three pipelines (token caps raised
# so reasoning models can answer, a retry when the compiler returns
# "unstructured", and the agent no longer offered tools its graph cannot
# support). Numbers measured before them describe code that no longer
# exists, so every published figure is re-measured here rather than partly
# refreshed.
cd "$(dirname "$0")"; unset TG_HOST
export LLM_PROVIDER=ollama OLLAMA_MODEL=llama3 EMBEDDING_PROVIDER=ollama
export LLM_TOKENS_PER_MINUTE=0 PYTHONUNBUFFERED=1

# Wait for the hidden-set run already in flight; running two at once would
# make both slower and neither finish sooner.
while [ ! -f outputs/summary_hidden_final.json ]; do sleep 20; done
echo "=== hidden set done, starting the sweep ==="

python -u -m src.eval.run_benchmark --suffix _v2_provided            > outputs/run_v2_provided.log 2>&1
echo "=== provided 100 done ==="
python -u -m src.eval.run_benchmark --stress --suffix _v2_chained    > outputs/run_v2_chained.log 2>&1
echo "=== chained 60 done ==="
python -u -m src.eval.run_benchmark --fast-path --suffix _v2_fastpath > outputs/run_v2_fastpath.log 2>&1
echo "=== production config done ==="

python -m scripts.export_submission --suffix _hidden_final > outputs/export.log 2>&1
python -m scripts.make_report
echo RERUN_COMPLETE
