#!/usr/bin/env bash
# Start the sweep only once the chunk fetch has proven itself.
#
# The cache file is written only after a COMPLETE fetch (the code refuses to
# cache a short read), so its existence is the verification. If the fetch
# fails the file never appears and no benchmark starts on a partial corpus —
# which is the failure mode worth avoiding, since a short read would quietly
# lower RAG's score rather than error.
cd "$(dirname "$0")"; unset TG_HOST
export LLM_PROVIDER=ollama OLLAMA_MODEL=llama3 EMBEDDING_PROVIDER=ollama
export LLM_TOKENS_PER_MINUTE=0 PYTHONUNBUFFERED=1

waited=0
while ! ls outputs/chunk_vectors_*.pkl >/dev/null 2>&1; do
  sleep 30; waited=$((waited+30))
  if [ $waited -gt 2400 ]; then echo "FETCH_NEVER_COMPLETED"; exit 1; fi
done
echo "=== chunk cache present, starting sweep ==="

python -u -m src.eval.run_benchmark --suffix _v2_provided             > outputs/run_v2_provided.log 2>&1
echo "=== provided 100 done ==="
python -u -m src.eval.run_benchmark --stress --suffix _v2_chained     > outputs/run_v2_chained.log 2>&1
echo "=== chained 60 done ==="
python -u -m src.eval.run_benchmark --fast-path --suffix _v2_fastpath > outputs/run_v2_fastpath.log 2>&1
echo "=== production config done ==="

python -m scripts.export_submission --suffix _hidden_final > outputs/export.log 2>&1
python -m scripts.make_report
echo SWEEP_COMPLETE
