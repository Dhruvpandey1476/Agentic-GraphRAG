"""
Re-run only the questions that errored in a finished results file, patch
them back in, and regenerate the summary.

A long benchmark is expensive — the stress set takes ~40 minutes — so a bug
fixed mid-run should not cost a full re-run. A running process holds the
code it started with, so a fix landed while a sweep is in flight applies to
nothing already in flight; those rows come back with an `error` field.

This re-runs exactly those rows against the current code and recomputes the
summary through the same `_summarize` the benchmark uses, so patched
results cannot drift from freshly-generated ones.

    python -m scripts.patch_failed --suffix _fairstress
    python -m scripts.patch_failed --suffix _fair100 --dry-run

The alternative — re-running everything — is also correct, just slower;
prefer it if more than a handful of rows failed, since a large failure count
usually means the run's configuration was wrong rather than one question
hitting a bug.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.eval.metrics import exact_match, retrieval_recall
from src.eval.run_benchmark import _summarize, STRESS_PATH

PIPELINES = ("rag", "graphrag", "agentic_graphrag")


def _questions_for(results_path):
    """Pick the question file whose qids match this results file."""
    name = os.path.basename(results_path)
    if "stress" in name:
        path = STRESS_PATH
    elif "hidden" in name:
        path = config.HIDDEN_QUESTIONS_PATH
    else:
        path = config.EVAL_QUESTIONS_PATH
    with open(path, encoding="utf-8") as f:
        return {str(json.loads(l)["qid"]): json.loads(l) for l in f if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True,
                    help="which results file to patch, e.g. _fairstress")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the failed rows without re-running anything")
    args = ap.parse_args()

    results_path = os.path.join(config.OUTPUT_DIR, f"results{args.suffix}.json")
    if not os.path.exists(results_path):
        print(f"No results file at {results_path}")
        sys.exit(1)

    with open(results_path, encoding="utf-8") as f:
        rows = json.load(f)

    failed = [(r, p) for r in rows for p in PIPELINES
              if p in r and "error" in r[p]]
    if not failed:
        print(f"No errored rows in results{args.suffix}.json — nothing to patch.")
        return

    print(f"{len(failed)} errored row(s):")
    for r, p in failed:
        print(f"  {r['qid']:12} {p:18} {r[p]['error'][:80]}")
    if args.dry_run:
        return

    from src.tigergraph_client import get_graph
    from src.embeddings import Embedder
    from src.llm_client import make_llm
    from src.pipelines import rag_pipeline, graphrag_pipeline
    from src.agents import orchestrator as agentic_pipeline

    graph, embedder, llm = get_graph(), Embedder(), make_llm()
    if config.TG_HOST and not config.FORCE_MOCK_GRAPH and graph.backend != "tigergraph":
        print(f"\n!! refusing to patch: TigerGraph is configured but the backend is "
              f"'{graph.backend}'. Patched rows would not match the rest of the file.")
        sys.exit(2)

    questions = _questions_for(results_path)
    has_gold = "hidden" not in os.path.basename(results_path)
    patched = 0

    for r, p in failed:
        q = questions.get(str(r["qid"]))
        if q is None:
            print(f"  {r['qid']}: not found in the question file, skipping")
            continue
        qt = q["question"]
        print(f"\nre-running {r['qid']} / {p}: {qt[:70]}")
        runner = {
            "rag": lambda: rag_pipeline.run(qt, graph=graph, embedder=embedder,
                                            llm=llm, qtype=q.get("qtype")),
            "graphrag": lambda: graphrag_pipeline.run(qt, graph=graph, llm=llm),
            "agentic_graphrag": lambda: agentic_pipeline.run(
                qt, graph=graph, llm=llm, embedder=embedder),
        }[p]
        try:
            res = runner()
        except Exception as e:
            print(f"   still failing: {type(e).__name__}: {e}")
            continue

        entry = dict(res)
        if has_gold:
            entry["exact_match"] = exact_match(res.get("answer"), q.get("answer"))
            if q.get("gold_doc_ids"):
                entry["retrieval_recall"] = retrieval_recall(
                    res.get("matched_doc_ids") or res.get("citations"), q["gold_doc_ids"])
            print(f"   answer={str(res.get('answer'))[:40]!r} "
                  f"gold={q.get('answer')} match={entry['exact_match']} "
                  f"tok={res.get('usage', {}).get('total_tokens', 0)}")
        else:
            print(f"   answer={str(res.get('answer'))[:40]!r}")
        r[p] = entry
        patched += 1

    if not patched:
        print("\nNothing patched.")
        return

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(config.OUTPUT_DIR, f"summary{args.suffix}.json")
    old_meta = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            old_meta = json.load(f).get("_meta", {})
    summary = _summarize(rows, list(PIPELINES), has_gold)
    summary["_meta"] = {**old_meta,
                        "patched_rows": patched,
                        "note": f"{patched} errored row(s) re-run by scripts/patch_failed.py"}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nPatched {patched} row(s) -> {results_path}")
    print(f"Regenerated {summary_path}")
    for p in PIPELINES:
        s = summary.get(p, {})
        if s.get("n_questions"):
            print(f"  {p:18} acc={s.get('exact_match_accuracy')} "
                  f"tok/correct={s.get('tokens_per_correct_answer')}")


if __name__ == "__main__":
    main()
