#!/usr/bin/env bash
cd "$(dirname "$0")"; unset TG_HOST
export LLM_PROVIDER=ollama OLLAMA_MODEL=llama3 EMBEDDING_PROVIDER=ollama
export LLM_TOKENS_PER_MINUTE=0 PYTHONUNBUFFERED=1

echo "=== verifying the chunk fetch ==="
python -u - <<'PY'
import time
from src.tigergraph_client import RealGraph
from src.embeddings import Embedder
g = RealGraph(); e = Embedder()
t = time.time(); chunks = g._fetch_all_chunks()
print(f"FETCHED {len(chunks)} chunks in {time.time()-t:.0f}s")
expected = g.conn.getVertexCount("Chunk")
assert len(chunks) == expected, f"SHORT READ: {len(chunks)} of {expected}"
top = g.vector_search(e.embed("who won gold in the men 20 kilometres walk 2012"), k=3)
for s, cid, c in top:
    print(f"   {s:.3f} {cid} {c.get('text','')[:60]}")
assert top and top[0][0] > 0.5, "vector search returned nothing useful"
print("FETCH_VERIFIED")
PY
[ $? -ne 0 ] && { echo "VERIFY_FAILED — not starting the sweep"; exit 1; }

echo "=== sweep: provided 100 ==="
python -u -m src.eval.run_benchmark --suffix _v2_provided             > outputs/run_v2_provided.log 2>&1
echo "=== sweep: chained 60 ==="
python -u -m src.eval.run_benchmark --stress --suffix _v2_chained     > outputs/run_v2_chained.log 2>&1
echo "=== sweep: production config ==="
python -u -m src.eval.run_benchmark --fast-path --suffix _v2_fastpath > outputs/run_v2_fastpath.log 2>&1
python -m scripts.export_submission --suffix _hidden_final > outputs/export.log 2>&1
python -m scripts.make_report
echo SWEEP_COMPLETE
