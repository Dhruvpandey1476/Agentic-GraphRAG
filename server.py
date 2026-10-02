"""
Live query server: ask one question, watch all three pipelines answer it.

    pip install flask
    python server.py           ->  http://localhost:5000

Why one endpoint PER PIPELINE rather than one that runs all three: with a
local model a single question takes ~60s per pipeline, and a combined
endpoint would leave the page blank for three minutes. The frontend fires
three requests at once and renders each as it lands, so RAG's answer appears
while the agent is still investigating.

The graph, embedder and LLM client are built once and shared. That matters
more than it looks: RealGraph.vector_search pulls 9,065 chunk vectors on
first use (~54s) and caches them, so a per-request client would pay that
every time.
"""
import os
import sys
import threading
import time
import traceback

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.pipelines import rag_pipeline, graphrag_pipeline
from src.agents import orchestrator as agentic_pipeline

app = Flask(__name__, static_folder=None)

_lock = threading.Lock()
_state = {"graph": None, "embedder": None, "llm": None, "ready": False, "error": None}


def _resources():
    """Build the shared graph/embedder/LLM once, under a lock.

    Flask's dev server is threaded, and the frontend deliberately fires three
    requests simultaneously — without the lock all three would race to build
    their own graph client and each pay the chunk-fetch cost.
    """
    if _state["ready"]:
        return _state
    with _lock:
        if _state["ready"]:
            return _state
        try:
            from src.tigergraph_client import get_graph
            from src.embeddings import Embedder
            from src.llm_client import make_llm
            _state["graph"] = get_graph()
            _state["embedder"] = Embedder()
            _state["llm"] = make_llm()
            _state["ready"] = True
        except Exception as e:
            _state["error"] = f"{type(e).__name__}: {e}"
            raise
    return _state


@app.get("/api/health")
def health():
    try:
        r = _resources()
    except Exception:
        return jsonify({"ready": False, "error": _state["error"]}), 503
    g = r["graph"]
    info = {
        "ready": True,
        "graph_backend": g.backend,
        "embedder": r["embedder"].describe(),
        "llm_provider": config.LLM_PROVIDER if r["llm"] else None,
        "llm_model": r["llm"].model if r["llm"] else None,
        "temperature": config.LLM_TEMPERATURE,
        "regex_fast_path": config.USE_REGEX_FAST_PATH,
    }
    try:
        info["olympic_events"] = (g.conn.getVertexCount("OlympicEvent")
                                  if g.backend == "tigergraph"
                                  else len(g.store.get("olympic_events", {})))
        info["chunks"] = (g.conn.getVertexCount("Chunk") if g.backend == "tigergraph"
                          else len(g.store.get("chunks", {})))
    except Exception:
        pass
    return jsonify(info)


@app.get("/api/samples")
def samples():
    """A few real evaluation questions, so the page is usable without the
    visitor having to invent Olympic trivia. Gold answers are included so the
    UI can mark each pipeline right or wrong live."""
    import json
    out = []
    for path, src in ((config.EVAL_QUESTIONS_PATH, "provided"),
                      (os.path.join(config.DATA_DIR, "eval_stress.jsonl"), "chained")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        by_type = {}
        for r in rows:                      # one example per question type
            by_type.setdefault(r.get("qtype"), r)
        for qt, r in by_type.items():
            out.append({"question": r["question"], "qtype": qt,
                        "gold": r.get("answer"), "set": src})
    return jsonify(out)


@app.post("/api/query")
def query():
    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    pipeline = body.get("pipeline", "agentic_graphrag")
    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        r = _resources()
    except Exception:
        return jsonify({"error": _state["error"]}), 503
    graph, embedder, llm = r["graph"], r["embedder"], r["llm"]

    runners = {
        "rag": lambda: rag_pipeline.run(question, graph=graph, embedder=embedder,
                                        llm=llm, qtype=body.get("qtype")),
        "graphrag": lambda: graphrag_pipeline.run(question, graph=graph, llm=llm),
        "agentic_graphrag": lambda: agentic_pipeline.run(question, graph=graph,
                                                         llm=llm, embedder=embedder),
    }
    if pipeline not in runners:
        return jsonify({"error": f"unknown pipeline {pipeline!r}"}), 400

    t0 = time.time()
    try:
        result = runners[pipeline]()
    except Exception as e:
        # Surface the failure in the UI instead of a blank panel — a pipeline
        # that raises is itself a result worth seeing.
        return jsonify({"pipeline": pipeline, "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-1200:],
                        "wall_time_s": round(time.time() - t0, 2)}), 200

    usage = result.get("usage", {}) or {}
    return jsonify({
        "pipeline": pipeline,
        "answer": result.get("answer"),
        "confidence": result.get("confidence"),
        "citations": (result.get("citations") or [])[:12],
        "tokens": {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "context": usage.get("context_tokens", 0),
            "total": usage.get("total_tokens", 0),
            "calls": usage.get("num_calls", 0),
        },
        "wall_time_s": round(time.time() - t0, 2),
        "steps": result.get("steps_taken", result.get("steps")),
        "escalated": result.get("escalated"),
        "escalation_reason": result.get("escalation_reason"),
        "repair": result.get("repair"),
        "stopped_because": result.get("stopped_reason"),
        "planner": result.get("planner"),
        "query_spec": result.get("query_spec"),
        "chunks_in_context": result.get("chunks_in_context"),
        "trace": [
            {"step": t.get("step"), "action": t.get("action"), "why": t.get("reason"),
             "outcome": t.get("output_summary"), "tokens": t.get("tokens"),
             "seconds": t.get("latency_s")}
            for t in (result.get("trace") or [])
        ],
    })


@app.get("/")
def index():
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "dashboard"), "live.html")


@app.get("/dashboard/<path:name>")
def dashboard_files(name):
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "dashboard"), name)


@app.get("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(config.OUTPUT_DIR, name)


if __name__ == "__main__":
    print("Live query UI   ->  http://localhost:5000")
    print("Benchmark dash  ->  http://localhost:5000/dashboard/index.html")
    print("\nFirst query is slow: the graph client fetches and caches chunk "
          "vectors on first use.\n")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
