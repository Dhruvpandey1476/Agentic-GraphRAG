"""
Runs all three pipelines over an evaluation set and writes:
  outputs/results[_suffix].json   per-question, per-pipeline detail + traces
  outputs/summary[_suffix].json   aggregate table + the agentic-value analysis

The summary is built to answer the hackathon's actual research question,
not just to report three accuracy numbers. Alongside accuracy and tokens
it computes, per question type:

  * accuracy delta of agentic over graphrag
  * the extra tokens agentic spent to get it
  * tokens-per-accuracy-point ("what did the agent's win cost?")
  * escalation rate — how often the agent decided it needed to reason at
    all, versus resolving deterministically for free
  * a verdict per question type: is agentic WORTH IT, NOT WORTH IT, or
    NEEDED (graphrag can't do it at all)

Usage:
  python -m src.eval.run_benchmark                      # 100 public questions
  python -m src.eval.run_benchmark --stress             # 60 chained questions
  python -m src.eval.run_benchmark --hidden             # 50 hidden (submission)
  python -m src.eval.run_benchmark --limit 10           # smoke test
  python -m src.eval.run_benchmark --ablation           # disable regex fast path
  python -m src.eval.run_benchmark --judge              # + LLM-as-judge scoring
  python -m src.eval.run_benchmark --pipelines graphrag,agentic_graphrag
  python -m src.eval.run_benchmark --no-resume           # ignore a checkpoint

Long runs are CHECKPOINTED. Each question's result is appended to
outputs/results<suffix>.partial.jsonl as soon as it completes, and a re-run
skips questions already present. A sweep paced against a hosted free tier
takes the better part of an hour, and losing all of it to a laptop running
out of memory — or a rate limit, or a dropped connection — is the
difference between finishing and not. Pass --no-resume to start clean.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.pipelines import rag_pipeline, graphrag_pipeline
from src.agents import orchestrator as agentic_pipeline
from src.eval.metrics import exact_match, judge_answer, retrieval_recall
from src.embeddings import Embedder
from src.llm_client import make_llm
from src.tigergraph_client import get_graph

STRESS_PATH = os.path.join(config.DATA_DIR, "eval_stress.jsonl")


def _load_checkpoint(path, fingerprint):
    """Completed rows from a previous attempt, keyed by qid.

    Rows are only reused when they were produced by the same backend,
    embedder and model. Resuming across a config change would silently
    splice two different experiments into one results file — exactly the
    kind of quietly-wrong output this project is built to avoid.
    """
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue     # a partial last line from a hard kill
            if row.get("_fingerprint") and row["_fingerprint"] != fingerprint:
                print(f"[resume] discarding checkpoint: it was produced with "
                      f"{row['_fingerprint']}, this run is {fingerprint}")
                return {}
            done[str(row.get("qid"))] = row
    return done


def load_jsonl(path):
    if not os.path.exists(path):
        print(f"No questions file at {path}. See data/README.md.")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--stress", action="store_true",
                    help="run the generated chained-reasoning set (src/eval/stress_set.py)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge", action="store_true", help="also run LLM-as-judge scoring")
    ap.add_argument("--ablation", action="store_true",
                    help="also disable the agent's deterministic triage, forcing every "
                         "question through the full adaptive loop")
    ap.add_argument("--fast-path", action="store_true",
                    help="re-enable the zero-token regex question-parser (OFF by default: "
                         "it encodes prior knowledge of the question templates, which RAG "
                         "is not given, so it measures eval-fitting rather than architecture)")
    ap.add_argument("--pipelines", default="rag,graphrag,agentic_graphrag")
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any checkpoint and re-run every question")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="permit running on the offline store even though TigerGraph "
                         "is configured (by default that is treated as a failed run)")
    args = ap.parse_args()
    pipelines = args.pipelines.split(",")

    if args.stress:
        path, suffix, has_gold = STRESS_PATH, "_stress", True
    elif args.hidden:
        path, suffix, has_gold = config.HIDDEN_QUESTIONS_PATH, "_hidden", False
    else:
        path, suffix, has_gold = config.EVAL_QUESTIONS_PATH, "", True
    if args.ablation:
        suffix += "_ablation"
    if args.suffix:
        suffix = args.suffix

    questions = load_jsonl(path)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        return

    llm = make_llm()
    embedder = Embedder()
    graph = get_graph()

    # If TigerGraph is configured but unreachable, get_graph() falls back to
    # the offline store. That fallback is right for development and wrong
    # for a benchmark: the run completes, the numbers look plausible, and
    # only a small "backend" field in the output says they did not come from
    # TigerGraph at all. That happened — a workspace idled out mid-sweep and
    # produced a full set of results labelled "mock". Fail loudly instead.
    if (config.TG_HOST and not config.FORCE_MOCK_GRAPH
            and graph.backend != "tigergraph" and not args.allow_fallback):
        print(f"\n!! TigerGraph is configured in .env but the run fell back to "
              f"'{graph.backend}'. Refusing to produce results that would be "
              f"mistaken for TigerGraph numbers.\n"
              f"   Start the workspace and re-run, or pass --allow-fallback if "
              f"you genuinely want offline numbers.")
        sys.exit(2)

    run_meta = {
        "questions_file": os.path.basename(path),
        "n_questions": len(questions),
        "graph_backend": graph.backend,
        "embedder": embedder.describe(),
        "llm_provider": config.LLM_PROVIDER if llm else None,
        "llm_model": llm.model if llm else None,
        "ablation_skip_triage": args.ablation,
        # False here is the headline configuration: every pipeline compiles
        # its query with the LLM, so all three spend real tokens.
        "regex_fast_path": args.fast_path,
        "temperature": config.LLM_TEMPERATURE,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(run_meta, indent=2))
    if llm is None:
        print("\n!! No LLM configured. RAG runs in proxy mode and the agentic loop "
              "cannot escalate. Token columns will be zero and the comparison is NOT "
              "publishable. Set a provider key in .env (groq/openrouter/ollama).\n")

    ckpt_path = os.path.join(config.OUTPUT_DIR, f"results{suffix}.partial.jsonl")
    if args.no_resume and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    # Temperature belongs in the fingerprint: a checkpoint written at a
    # non-zero temperature is not comparable with one written at 0, and
    # resuming across the two would splice a sampled run into a deterministic
    # one without any visible sign.
    fingerprint = (f"{graph.backend}/{embedder.describe()}/{run_meta['llm_model']}"
                   f"/T{config.LLM_TEMPERATURE}")
    done = _load_checkpoint(ckpt_path, fingerprint)
    if done:
        print(f"[resume] {len(done)} question(s) already completed in "
              f"{os.path.basename(ckpt_path)}; skipping those")

    all_results = []
    t0 = time.time()
    ckpt = open(ckpt_path, "a", encoding="utf-8")
    for i, q in enumerate(questions):
        qt = q["question"]
        qid = str(q.get("qid", i))
        if qid in done:
            all_results.append(done[qid])
            continue
        print(f"[{i+1}/{len(questions)}] ({q.get('qtype','?')}) {qt[:75]}")
        row = {"qid": q.get("qid", i), "question": qt, "qtype": q.get("qtype"),
               "hops_required": q.get("hops_required"), "_fingerprint": fingerprint}

        runners = {
            "rag": lambda: rag_pipeline.run(qt, graph=graph, embedder=embedder,
                                            llm=llm, qtype=q.get("qtype")),
            "graphrag": lambda: graphrag_pipeline.run(
                qt, graph=graph, llm=llm, force_llm_planner=not args.fast_path),
            "agentic_graphrag": lambda: agentic_pipeline.run(
                qt, graph=graph, llm=llm, embedder=embedder,
                force_llm_planner=not args.fast_path, skip_triage=args.ablation),
        }

        for name in pipelines:
            fn = runners.get(name)
            if fn is None:
                continue
            try:
                res = fn()
            except Exception as e:
                row[name] = {"error": f"{type(e).__name__}: {e}", "answer": None,
                             "usage": {"total_tokens": 0, "num_calls": 0}}
                print(f"    !! {name} raised {type(e).__name__}: {e}")
                continue
            entry = dict(res)
            if has_gold:
                entry["exact_match"] = exact_match(res.get("answer"), q.get("answer"))
                if q.get("gold_doc_ids"):
                    entry["retrieval_recall"] = retrieval_recall(
                        res.get("matched_doc_ids") or res.get("citations"), q["gold_doc_ids"])
                if args.judge and llm:
                    entry["judge"] = judge_answer(qt, res.get("answer", ""),
                                                  q.get("answer"), llm=llm)
            row[name] = entry
        all_results.append(row)
        # Flush immediately: an unflushed buffer is exactly what gets lost
        # when the process is killed rather than exiting.
        ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
        ckpt.flush()
        os.fsync(ckpt.fileno())

    ckpt.close()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, f"results{suffix}.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = _summarize(all_results, pipelines, has_gold)
    summary["_meta"] = {**run_meta, "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(config.OUTPUT_DIR, f"summary{suffix}.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Drop the checkpoint only now that the real outputs are safely on disk.
    # Leaving it would make a deliberate re-run of this suffix skip every
    # question and silently reuse the old results.
    try:
        os.remove(ckpt_path)
    except OSError:
        pass

    print(f"\nDone in {time.time()-t0:.1f}s -> outputs/summary{suffix}.json")
    _print_table(summary, has_gold)


# ---------------------------------------------------------------- summarising

def _rows_for(all_results, p):
    """Every row this pipeline produced, rows that errored included.

    A question that errored is a question the pipeline did not answer, so it
    belongs in the accuracy denominator. Filtering it out here silently
    shrinks the denominator and inflates the score: the chained set reported
    0.500 when two of its sixty rows had errored and the honest figure was
    0.483. Cost averages are taken over the rows that ran, since a row that
    crashed has no meaningful token count, and `n_errors` records the gap.
    """
    return [r[p] for r in all_results if p in r]


def _agg(rows, has_gold):
    ran = [r for r in rows if "error" not in r]
    n_all = max(len(rows), 1)
    n_ran = max(len(ran), 1)
    tokens = [r.get("usage", {}).get("total_tokens", 0) for r in ran]
    entry = {
        "n_questions": len(rows),
        "n_errors": len(rows) - len(ran),
        "avg_tokens": round(sum(tokens) / n_ran, 1),
        "total_tokens": sum(tokens),
        "avg_input_tokens": round(sum(r.get("usage", {}).get("input_tokens", 0)
                                      for r in ran) / n_ran, 1),
        "avg_output_tokens": round(sum(r.get("usage", {}).get("output_tokens", 0)
                                       for r in ran) / n_ran, 1),
        "avg_context_tokens": round(sum(r.get("usage", {}).get("context_tokens", 0)
                                        for r in ran) / n_ran, 1),
        "avg_llm_calls": round(sum(r.get("usage", {}).get("num_calls", 0)
                                   for r in ran) / n_ran, 2),
        "avg_steps": round(sum(r.get("steps_taken", r.get("steps", 1))
                               for r in ran) / n_ran, 2),
        "avg_latency_s": round(sum(r.get("usage", {}).get("latency_s", 0)
                                   for r in ran) / n_ran, 3),
        # Over every question asked: a pipeline that errored did not answer.
        "answered_rate": round(sum(1 for r in rows
                                   if r.get("answer") not in (None, "")) / n_all, 3),
    }
    if has_gold:
        m = [bool(r.get("exact_match")) for r in rows]
        entry["exact_match_accuracy"] = round(sum(m) / n_all, 3)
        # Cost per RESULT, not per question. Pipelines that answer different
        # numbers of questions correctly are not comparable on avg_tokens
        # alone: a pipeline that is cheap because it answers nothing is not
        # efficient, it is just cheap.
        n_correct = sum(m)
        entry["tokens_per_correct_answer"] = (
            round(sum(tokens) / n_correct, 1) if n_correct else None)
        rec = [r["retrieval_recall"] for r in rows if r.get("retrieval_recall") is not None]
        if rec:
            entry["avg_retrieval_recall"] = round(sum(rec) / len(rec), 3)
        judged = [r["judge"]["overall"] for r in rows if "judge" in r]
        if judged:
            entry["avg_judge_score"] = round(sum(judged) / len(judged), 3)
    return entry


def _summarize(all_results, pipelines, has_gold):
    summary = {}
    for p in pipelines:
        rows = _rows_for(all_results, p)
        summary[p] = _agg(rows, has_gold) if rows else {"n_questions": 0, "note": "no results"}

    # ---- agentic behaviour ----
    ag = _rows_for(all_results, "agentic_graphrag")
    if ag:
        escalated = [r for r in ag if r.get("escalated")]
        method_counts, agent_counts = defaultdict(int), defaultdict(int)
        for r in ag:
            for m in r.get("retrieval_methods_used", []):
                method_counts[m] += 1
            for a in r.get("agents_invoked", []):
                agent_counts[a] += 1
        stop_reasons = defaultdict(int)
        for r in ag:
            stop_reasons[(r.get("stopped_reason") or "unknown").split(" (")[0]] += 1
        summary["agentic_behavior"] = {
            "escalation_rate": round(len(escalated) / len(ag), 3),
            "n_escalated": len(escalated),
            "n_resolved_free": len(ag) - len(escalated),
            "avg_tokens_when_escalated": round(
                sum(r.get("usage", {}).get("total_tokens", 0) for r in escalated)
                / max(len(escalated), 1), 1),
            "avg_tokens_when_not_escalated": round(
                sum(r.get("usage", {}).get("total_tokens", 0)
                    for r in ag if not r.get("escalated")) / max(len(ag) - len(escalated), 1), 1),
            "avg_strategy_changes": round(
                sum(r.get("num_strategy_changes", 0) for r in ag) / len(ag), 2),
            "avg_citations": round(sum(r.get("num_citations", 0) for r in ag) / len(ag), 2),
            "avg_evidence_items": round(
                sum(r.get("num_evidence_items", 0) for r in ag) / len(ag), 2),
            "retrieval_methods_used": dict(sorted(method_counts.items(),
                                                  key=lambda kv: -kv[1])),
            "specialized_agents_invoked": dict(sorted(agent_counts.items(),
                                                      key=lambda kv: -kv[1])),
            "stopping_reasons": dict(sorted(stop_reasons.items(), key=lambda kv: -kv[1])),
        }

    # ---- planner provenance: how many answers came from templates? ----
    planners = defaultdict(int)
    for r in all_results:
        p = r.get("graphrag", {}).get("planner")
        if p:
            planners[p] += 1
    if planners:
        summary["planner_provenance"] = dict(planners)

    # ---- the headline analysis: when is agentic worth it? ----
    if has_gold:
        summary["agentic_value_by_qtype"] = _value_analysis(all_results)
    return summary


def _value_analysis(all_results):
    """Per question type: did the agent's extra tokens buy extra accuracy?

    The verdict thresholds are deliberately conservative. A gain under 3
    points is inside the noise of a 12-20 question bucket, so it is not
    claimed as a win; any accuracy the fixed pipeline simply cannot reach
    (it scores 0) is reported as NEEDED rather than as a percentage gain,
    because a ratio against zero is meaningless."""
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r.get("qtype") or "unknown"].append(r)

    out = {}
    for qt, rows in sorted(by_type.items()):
        def acc(p):
            vals = [bool(r[p].get("exact_match")) for r in rows if p in r and "error" not in r[p]]
            return round(sum(vals) / len(vals), 3) if vals else None

        def tok(p):
            vals = [r[p].get("usage", {}).get("total_tokens", 0)
                    for r in rows if p in r and "error" not in r[p]]
            return round(sum(vals) / len(vals), 1) if vals else None

        a_rag, a_gr, a_ag = acc("rag"), acc("graphrag"), acc("agentic_graphrag")
        t_gr, t_ag = tok("graphrag"), tok("agentic_graphrag")

        entry = {"n": len(rows), "accuracy": {"rag": a_rag, "graphrag": a_gr,
                                              "agentic_graphrag": a_ag},
                 "avg_tokens": {"graphrag": t_gr, "agentic_graphrag": t_ag}}

        if a_gr is not None and a_ag is not None:
            delta = round(a_ag - a_gr, 3)
            extra = round((t_ag or 0) - (t_gr or 0), 1)
            entry["accuracy_delta"] = delta
            entry["extra_tokens"] = extra
            if delta > 0:
                entry["tokens_per_accuracy_point"] = round(extra / (delta * 100), 1)

            if a_gr == 0 and a_ag > 0:
                entry["verdict"] = "AGENT NEEDED — fixed pipeline cannot answer these at all"
            elif delta >= 0.03:
                entry["verdict"] = f"AGENT WORTH IT — +{delta*100:.1f} pts for {extra:.0f} extra tokens"
            elif delta <= -0.03:
                entry["verdict"] = "AGENT HARMFUL — fixed pipeline is more accurate here"
            else:
                entry["verdict"] = ("AGENT NOT WORTH IT — no accuracy gain beyond noise; "
                                    "the extra tokens buy nothing")
        out[qt] = entry
    return out


def _print_table(summary, has_gold):
    cols = ["rag", "graphrag", "agentic_graphrag"]
    print(f"\n{'pipeline':<20}{'acc':>8}{'tokens':>10}{'calls':>8}{'steps':>8}{'answered':>10}")
    print("-" * 64)
    for c in cols:
        s = summary.get(c) or {}
        if not s.get("n_questions"):
            continue
        acc = s.get("exact_match_accuracy")
        print(f"{c:<20}{(f'{acc:.3f}' if acc is not None else '—'):>8}"
              f"{s['avg_tokens']:>10.1f}{s['avg_llm_calls']:>8.2f}"
              f"{s['avg_steps']:>8.2f}{s['answered_rate']:>10.3f}")

    b = summary.get("agentic_behavior")
    if b:
        print(f"\nescalation rate: {b['escalation_rate']:.1%} "
              f"({b['n_escalated']} escalated / {b['n_resolved_free']} resolved free)")
        print(f"tokens when escalated: {b['avg_tokens_when_escalated']:.0f} | "
              f"when not: {b['avg_tokens_when_not_escalated']:.0f}")

    va = summary.get("agentic_value_by_qtype")
    if va:
        print("\n--- is the agent worth it? ---")
        for qt, e in va.items():
            if "verdict" in e:
                print(f"  {qt:<22} {e['verdict']}")


if __name__ == "__main__":
    main()
