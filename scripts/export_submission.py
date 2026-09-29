"""
Export the hidden-set answers in the shape the hackathon asks for:
"run your system on all 50 and submit the raw outputs — tokens used,
answers generated, and agentic trace".

Writes two files:

  outputs/submission_hidden.jsonl   one self-contained record per question,
                                    easy to diff and to grade line by line
  outputs/submission_hidden.json    the same content as one array, plus a
                                    header describing how it was produced

The internal results file carries a lot that a grader does not need (raw
evidence blobs, per-pipeline duplicates). This keeps exactly the three things
asked for, plus the provenance needed to trust them: which backend, which
model, and which configuration produced each answer.

    python -m scripts.export_submission
    python -m scripts.export_submission --suffix _tg_hidden
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _trace_step(t):
    """One trace step, trimmed to what explains the investigation."""
    return {
        "step": t.get("step"),
        "action": t.get("action"),
        "why": t.get("reason"),
        "outcome": t.get("output_summary"),
        "new_evidence": t.get("new_evidence_count"),
        "strategy_changed": t.get("strategy_changed"),
        "tokens": t.get("tokens"),
        "seconds": t.get("latency_s"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_tg_hidden")
    ap.add_argument("--pipeline", default="agentic_graphrag")
    args = ap.parse_args()

    results = os.path.join(config.OUTPUT_DIR, f"results{args.suffix}.json")
    summary = os.path.join(config.OUTPUT_DIR, f"summary{args.suffix}.json")
    if not os.path.exists(results):
        print(f"No results at {results}")
        sys.exit(1)

    with open(results, encoding="utf-8") as f:
        rows = json.load(f)
    meta = {}
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            meta = json.load(f).get("_meta", {})

    records, totals = [], {"tokens": 0, "calls": 0, "escalated": 0, "answered": 0}
    for r in rows:
        d = r.get(args.pipeline) or {}
        usage = d.get("usage", {}) or {}
        answer = d.get("answer")
        rec = {
            "qid": r.get("qid"),
            "question": r.get("question"),
            "answer": answer,
            "confidence": d.get("confidence"),
            "citations": d.get("citations", []),
            "supporting_doc_ids": d.get("matched_doc_ids", []),
            "tokens": {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "context": usage.get("context_tokens", 0),
                "total": usage.get("total_tokens", 0),
                "llm_calls": usage.get("num_calls", 0),
            },
            "agentic_trace": {
                "steps_taken": d.get("steps_taken"),
                "escalated": d.get("escalated"),
                "escalation_reason": d.get("escalation_reason"),
                "deterministic_repair": d.get("repair"),
                "retrieval_methods": d.get("retrieval_methods_used", []),
                "specialised_agents": d.get("agents_invoked", []),
                "strategy_changes": d.get("num_strategy_changes"),
                "stopped_because": d.get("stopped_reason"),
                "wall_time_s": d.get("wall_time_s"),
                "steps": [_trace_step(t) for t in d.get("trace", [])],
            },
        }
        records.append(rec)
        totals["tokens"] += usage.get("total_tokens", 0)
        totals["calls"] += usage.get("num_calls", 0)
        totals["escalated"] += 1 if d.get("escalated") else 0
        totals["answered"] += 1 if answer not in (None, "") else 0

    n = max(len(records), 1)
    header = {
        "system": "Agentic GraphRAG on TigerGraph",
        "pipeline": args.pipeline,
        "n_questions": len(records),
        "graph_backend": meta.get("graph_backend"),
        "llm_model": meta.get("llm_model"),
        "embedder": meta.get("embedder"),
        "regex_fast_path": meta.get("regex_fast_path"),
        "answered": totals["answered"],
        "total_tokens": totals["tokens"],
        "avg_tokens_per_question": round(totals["tokens"] / n, 1),
        "total_llm_calls": totals["calls"],
        "escalation_rate": round(totals["escalated"] / n, 3),
        "note": ("escalated=false means the question was resolved by an exact graph "
                 "query that passed verification, with no reasoning tokens spent"),
    }

    jsonl = os.path.join(config.OUTPUT_DIR, "submission_hidden.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    combined = os.path.join(config.OUTPUT_DIR, "submission_hidden.json")
    with open(combined, "w", encoding="utf-8") as f:
        json.dump({"_meta": header, "answers": records}, f, indent=2, ensure_ascii=False)

    print(json.dumps(header, indent=2))
    print(f"\nwrote {jsonl}")
    print(f"wrote {combined}")
    missing = [r["qid"] for r in records if r["answer"] in (None, "")]
    if missing:
        print(f"\n!! {len(missing)} question(s) with no answer: {missing}")


if __name__ == "__main__":
    main()
