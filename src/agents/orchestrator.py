"""
Pipeline 3 — Agentic GraphRAG.

The orchestrator decides its next action at every step from (question,
graph schema, evidence so far, what is still missing). It is not a fixed
sequence, and the concrete behaviours that make it agentic — each of
which Pipeline 2 structurally cannot do — are:

  1. TRIAGE + ESCALATION. A free deterministic attempt runs first. Its
     result is then VERIFIED, and verification failure escalates into the
     adaptive loop. Pipeline 2 accepts its first result unconditionally,
     including results the executor itself flagged as ambiguous.

  2. CONSTRAINT RELAXATION. When a graph query returns zero rows, the
     agent identifies which constraint was over-specified and reissues a
     relaxed query. Pipeline 2 has one shot and reports nothing.

  3. DECOMPOSITION. Questions needing two or more independent graph
     queries (compare two editions, resolve an entity then query it) are
     split into sub-questions, each answered separately, then combined.
     Pipeline 2 compiles exactly one query and cannot express this.

  4. EVIDENCE-DRIVEN STOPPING. The agent stops when an explicit
     sufficiency check passes, not after a fixed number of steps — and
     every stop records when and why.

The cost discipline matters as much as the capability: the agent spends
ZERO tokens on questions the deterministic path already answers
verifiably, and only pays for reasoning on questions that actually need
it. `escalated` in the result marks which questions those were, and that
flag is the raw material for the hackathon's headline question — which
questions need an agent, and which don't.
"""
import sys
import os
import json
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.agents.harness import AgentState, EvidenceItem
from src.agents import specialized_agents as agents
from src.agents import structured_agent as sa
from src.agents import query_planner as qp
from src.llm_client import UsageTracker, parse_json_safely
from src.embeddings import Embedder
from src.tigergraph_client import get_graph
from src.ingestion.infobox import normalize


PLANNER_SYSTEM = """You are the orchestrator of an investigative agent
answering questions from a knowledge graph of Olympic events plus a document
store. At each step you choose exactly ONE next action, based on the evidence
gathered so far and what is still missing.

Actions:
- graph_query(question: str) — compile ONE natural-language sub-question into a
  structured graph query and run it exactly (counting, argmax, chronology,
  medal lookup, venue/date identification). Prefer this for anything factual
  about event stats. Pass a self-contained sub-question, not the original.
- relax_query(drop: str) — reissue the last graph query with one constraint
  removed. Use ONLY after a graph query returned zero rows. `drop` names the
  constraint to remove: "discipline", "date", "venue", "games_id" or
  "event_name".
- similarity_search(query: str) — vector search over raw document text. Use
  when the graph layer found nothing, or the question needs prose context
  rather than a structured field.
- entity_linking(name: str) — resolve a surface-form name to graph entities.
- graph_traversal(entity_id: str, hops: int) — expand from a linked entity.
- document_retrieval(entity_id: str) — pull raw supporting text for an entity.
- multihop_reason — one explicit reasoning hop over the evidence you have.
- answer — you can now answer completely and correctly. Stop.

A first, cheap attempt at this question ALREADY FAILED verification. This is
why, and it tells you exactly what is missing:

    {escalation_reason}

If that reason says a value was guessed rather than read from the question,
your FIRST action must be a graph_query whose sub-question RESOLVES that value
(e.g. "which discipline held the most events at the 2004 Summer Olympics?").
Only once its answer appears in the evidence may you issue the real query,
naming the resolved value literally.
If it says the question covers more than one year or edition, ask about each
one in a SEPARATE graph_query, then compare the answers.
If it says a query matched no records, use relax_query to drop the constraint
that was too narrow.

Evidence so far is shown below. What you have ALREADY tried is listed too —
do not repeat an action that produced nothing; change strategy instead.

Actions already tried: {tried}
Linked entities: {linked}

Return ONLY JSON: {{"action": str, "args": {{}}, "reason": str}}
`reason` must state what specific gap this action closes — one sentence,
specific to this question, not a generic justification."""


FINAL_ANSWER_SYSTEM = """Using ALL evidence gathered during this investigation,
give the final answer.

- Answer with the shortest exact string that answers the question (a name, a
  number, an event title) — not a sentence, unless the question asks for one.
- Cite the evidence ids you actually used.
- If the investigation genuinely failed to find the answer, set answer to null
  rather than guessing.

Return ONLY JSON: {"answer": str|null, "citations": [str], "confidence": float}"""


# ==========================================================================
#  Deterministic triage
# ==========================================================================

def _run_graph_query(state, graph, llm, question, force_llm=None, usage=None):
    """Compile + execute one graph query, logging it as a trace step.
    Returns (result, spec). Shared by the triage path and the agent loop so
    both are measured identically."""
    t0 = time.time()
    disciplines = graph.known_disciplines()
    spec, planner_r = qp.plan(question, disciplines, llm=llm, force_llm=force_llm)
    tokens = 0
    if planner_r is not None:
        tokens = planner_r.total_tokens
        if usage is not None:
            usage.record("query_planner", planner_r)

    if spec.get("type") == "unstructured":
        state.log_step("graph_query", "tried to compile this as a structured graph query",
                       {"sub_question": question, "planner": spec.get("_planner")},
                       "not expressible as a graph query; need text retrieval instead",
                       0, tokens=tokens, latency_s=time.time() - t0)
        return None, spec

    result = sa.execute(spec, graph)
    n_rows = len(result.get("evidence", []))
    if result["answer"] is None:
        summary = f"query ran but matched no records ({result.get('note','')})"
    elif result.get("ambiguous"):
        summary = (f"answered {result['answer']!r} but {len(result.get('tied_candidates', []))} "
                   f"candidates tied — result is NOT unambiguous")
    else:
        summary = f"resolved exactly: {result['answer']!r} from {n_rows} graph record(s)"

    state.log_step("graph_query",
                   f"compiled to a '{spec['type']}' graph query via the "
                   f"{spec.get('_planner')} planner",
                   {"sub_question": question, "spec": {k: v for k, v in spec.items()
                                                       if not k.startswith("_")}},
                   summary, n_rows, tokens=tokens, latency_s=time.time() - t0)

    # Record the query's ANSWER as its own evidence item, not just the rows
    # it was computed from. Chained questions depend on this: step 2's
    # sub-question ("how many athletics events...") can only be written once
    # step 1's answer ("Athletics") is visible in the evidence block.
    if result["answer"] is not None:
        state.add_evidence([EvidenceItem(
            step=state.step_count, source_agent="graph_query", kind="result",
            ref_id=f"q{state.step_count}",
            content=f"ANSWER to sub-question {question!r} = {result['answer']!r}"
                    + (f" (ambiguous: {len(result.get('tied_candidates', []))} tied candidates)"
                       if result.get("ambiguous") else ""),
            metadata={"spec_type": spec.get("type"), "sub_question": question},
        )])

    for e in result.get("evidence", [])[:12]:
        state.add_evidence([EvidenceItem(
            step=state.step_count, source_agent="graph_query", kind="fact",
            ref_id=e["doc_id"],
            content=f"{e['title']}: " + json.dumps(
                {k: v for k, v in e.items() if k not in ("doc_id", "url", "medalists")}),
            metadata={"medalists": e.get("medalists", [])},
        )])

    state.last_spec = spec
    return result, spec


GROUNDED_FILTER_KEYS = ("discipline", "event_name_substr", "venue", "venue_substr",
                        "title", "noc")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _check_query_grounding(spec, question):
    """Is the compiled query actually GROUNDED IN the question, or did the
    planner invent the values it filtered on?

    This is the check that matters most, and the one a structural
    validity check misses entirely. Given "in the discipline that held the
    most events at the 2004 Summer Olympics, how many of its events had
    more than 41 competitors", a planner will happily emit
    `aggregation(discipline="Athletics", ...)`. That query is perfectly
    well-formed, returns real rows, cites real documents — and answers a
    question nobody asked, because "Athletics" was a guess. Every
    structural check passes and the system reports a confidently wrong
    answer.

    Two deterministic, domain-general rules catch it:

      1. LITERAL GROUNDING. Every string a query filters on must actually
         appear in the question. If it doesn't, the planner supplied it
         rather than read it, which means the question's real subject was
         a referring expression ("the discipline that...") that has to be
         resolved by a prior query.

      2. CONSTRAINT COVERAGE. If the question names more distinct years
         than the query consumed, the query is answering about part of
         the question only — the signature of a comparison that needs two
         sub-queries rather than one.

    Neither rule looks at question phrasing templates, so neither is
    fitted to this dataset's wording.
    """
    if not spec:
        return True, ""
    q_norm = normalize(question)

    for key in GROUNDED_FILTER_KEYS:
        val = spec.get(key)
        if not val or not isinstance(val, str):
            continue
        if normalize(val) and normalize(val) not in q_norm:
            return False, (f"the query filtered on {key}={val!r}, which the question never "
                           f"states — the planner guessed it, so the real subject has to be "
                           f"resolved by a prior query")

    # 1b. NUMERIC GROUNDING. Same principle applied to numbers. A small
    #     model reading "more than 41 competitors" will sometimes emit
    #     min_competitors=42 — a query that is well-formed, returns rows,
    #     and answers a question with a different threshold. Counting
    #     questions are exactly where an off-by-one is invisible in the
    #     output, so the number has to be checked against the question
    #     rather than trusted.
    q_numbers = set(re.findall(r"\d+", question))
    for key in ("min_competitors", "max_competitors"):
        val = spec.get(key)
        if isinstance(val, int) and str(val) not in q_numbers:
            return False, (f"the query used {key}={val}, a number the question never mentions "
                           f"(it names {sorted(q_numbers)}) — the threshold was misread")

    q_years = set(_YEAR_RE.findall(question))
    spec_years = set(_YEAR_RE.findall(json.dumps(spec)))
    if len(q_years) >= 2 and len(spec_years) < len(q_years):
        return False, (f"the question names {len(q_years)} distinct years {sorted(q_years)} but "
                       f"the query only used {sorted(spec_years) or 'none'} — it covers part of "
                       f"the question, so this needs more than one query")

    return True, ""


def _verify_deterministically(result, spec, question=""):
    """Free, rule-based verification of a triage result — no LLM.

    Returns (ok, reason). This gate decides whether a question needs an
    agent at all, so it is deliberately conservative: anything the
    executor flagged, anything empty, and anything whose query wasn't
    grounded in the question fails and escalates."""
    if result is None:
        return False, "question did not compile to a graph query"

    grounded, why = _check_query_grounding(spec, question)
    if not grounded:
        return False, why

    if result.get("answer") is None:
        return False, f"graph query matched no records ({result.get('note', 'empty result')})"
    if result.get("ambiguous"):
        return False, (f"executor flagged {len(result.get('tied_candidates', []))} tied "
                       f"candidates — answer is not uniquely determined")
    if str(result["answer"]).strip() == "":
        return False, "graph query returned an empty value"
    if not result.get("matched_doc_ids"):
        return False, "answer has no supporting document to cite"
    return True, "graph query resolved uniquely, grounded in the question, with citable records"


# ==========================================================================
#  Deterministic query repair
# ==========================================================================

_SUPERLATIVE_MIN = ("fewest", "least", "lowest", "smallest")


def _repair_query(state, graph, spec, question, reason, _fixed=None):
    """Repair a query the grounding check rejected — deterministically,
    for zero tokens, before falling back to the LLM loop.

    The grounding diagnostic doesn't just say "this failed", it says
    exactly WHICH slot was unresolved. That is enough to fix two whole
    classes of failure without any reasoning:

      * COVERAGE failure ("the question names 2 years, the query used 1"):
        re-run the same query once per year the question names, then
        compare the results. This is pure decomposition — the sub-queries
        are the original query with one field varied.

      * GUESSED DISCIPLINE, where the query is scoped to a Games edition:
        the discipline the question refers to is recoverable from the
        graph itself via a count-per-discipline argmax over that edition.
        The question's own comparative word ("most" vs "fewest") gives the
        direction.

    Anything else returns None and the LLM loop takes over. This is
    deliberately a narrow, verifiable repair rather than a general one:
    each case is repaired only because the fix is fully determined by the
    diagnostic plus the graph, with nothing guessed. A repaired result is
    re-verified like any other, so a bad repair escalates rather than
    being trusted.

    The practical payoff is large: these repairs make the agent correct on
    chained questions even with a weak local model, because the hard part
    (knowing what was missing) was solved by a rule, not by the model.
    """
    if not spec:
        return None
    # Slots already repaired on this chain. Without it, fixing a threshold
    # re-checks grounding, sees the (already-substituted) discipline still
    # absent from the question, and resolves it a second time — correct but
    # it doubles the trace and hides what actually happened.
    _fixed = set(_fixed or ())

    # ---- coverage failure: decompose over the years the question names ----
    if "distinct years" in reason:
        years = sorted(set(_YEAR_RE.findall(question)))
        season = "winter" if "winter" in question.lower() else "summer"
        sides = []
        for y in years:
            variant = {**spec, "games_id": f"{y}-{season}"}
            variant.pop("_planner", None)
            res = sa.execute(variant, graph)
            sides.append((y, res))
            state.log_step(
                "decompose", f"question covers {len(years)} editions but one query holds one — "
                             f"running the same query for {y} as a separate sub-query",
                {"games_id": variant["games_id"], "type": variant.get("type")},
                f"{y}: {res.get('answer')!r} from {len(res.get('evidence', []))} record(s)",
                len(res.get("evidence", [])))

        numeric = [(y, float(r["answer"])) for y, r in sides
                   if r.get("answer") not in (None, "") and str(r["answer"]).replace(".", "").isdigit()]
        if len(numeric) == len(years) and len(numeric) >= 2:
            want_min = any(w in question.lower() for w in _SUPERLATIVE_MIN)
            pick = min(numeric, key=lambda t: t[1]) if want_min else max(numeric, key=lambda t: t[1])
            if sum(1 for _, v in numeric if v == pick[1]) > 1:
                return None            # a tie is not a repair, it's ambiguity
            evidence = [e for _, r in sides for e in r.get("evidence", [])]
            state.log_step("compare", "comparing the sub-query answers to pick the edition the "
                                      "question asks for",
                           {"sides": {y: v for y, v in numeric}},
                           f"{pick[0]} wins ({pick[1]:g} vs "
                           f"{[v for y, v in numeric if y != pick[0]]})", 0)
            return {"answer": pick[0], "evidence": evidence,
                    "matched_doc_ids": [e["doc_id"] for e in evidence], "repaired": "coverage"}
        return None

    # ---- misread numeric threshold, recoverable from the question ----
    if "a number the question never mentions" in reason:
        q_numbers = [int(n) for n in re.findall(r"\d+", question)]
        # Years are not thresholds; a threshold is whichever number is left
        # once the edition years are accounted for.
        years = {int(y) for y in _YEAR_RE.findall(question)}
        candidates = [n for n in q_numbers if n not in years]
        if len(candidates) != 1:
            return None
        key = "min_competitors" if "min_competitors" in spec else "max_competitors"
        fixed = {**spec, key: candidates[0]}
        fixed.pop("_planner", None)
        out = sa.execute(fixed, graph)
        state.log_step(
            "repair_literal",
            f"the compiled query's {key} did not appear in the question; substituting the "
            f"only threshold the question actually states",
            {"was": spec.get(key), "now": candidates[0]},
            f"re-ran with {key}={candidates[0]}: {out.get('answer')!r} from "
            f"{len(out.get('evidence', []))} record(s)",
            len(out.get("evidence", [])))
        if out.get("answer") is None:
            return None
        # The threshold is fixed, but the discipline may still be a guess.
        ok, why = _check_query_grounding(fixed, question)
        if not ok and "discipline" not in _fixed:
            nested = _repair_query(state, graph, fixed, question, why,
                                   _fixed | {"literal"})
            return nested if nested else None
        return {**out, "repaired": "literal_threshold"}

    # ---- guessed discipline, recoverable from the graph ----
    if "discipline=" in reason and spec.get("games_id") and "discipline" not in _fixed:
        direction = "min" if any(w in question.lower() for w in _SUPERLATIVE_MIN) else "max"
        resolver = {"type": "discipline_superlative", "games_id": spec["games_id"],
                    "direction": direction}
        res = sa.execute(resolver, graph)
        state.log_step(
            "resolve_slot",
            "the discipline was never named in the question, so resolving it from the graph "
            "before re-issuing the real query",
            {"games_id": spec["games_id"], "direction": direction},
            f"resolved discipline = {res.get('answer')!r}"
            + (" (tied — not usable)" if res.get("ambiguous") else ""),
            0)
        if res.get("answer") is None or res.get("ambiguous"):
            return None

        state.add_evidence([EvidenceItem(
            step=state.step_count, source_agent="resolve_slot", kind="result",
            ref_id=f"slot{state.step_count}",
            content=f"RESOLVED the question's unnamed discipline = {res['answer']!r} "
                    f"(the {direction}-event discipline at {spec['games_id']})")])

        repaired = {**spec, "discipline": res["answer"]}
        repaired.pop("_planner", None)
        # The resolved discipline is grounded now, but any OTHER slot the
        # planner got wrong (a misread threshold, say) is still wrong —
        # re-check and chain the next repair rather than trusting it.
        ok, why = _check_query_grounding({k: v for k, v in repaired.items()
                                          if k != "discipline"}, question)
        if not ok and "literal" not in _fixed:
            nested = _repair_query(state, graph, repaired, question, why,
                                   _fixed | {"discipline"})
            if nested:
                return nested
        out = sa.execute(repaired, graph)
        state.log_step(
            "graph_query", "re-issuing the original query with the resolved discipline "
                           "substituted in",
            {"spec": {k: v for k, v in repaired.items() if not k.startswith("_")}},
            f"resolved exactly: {out.get('answer')!r} from {len(out.get('evidence', []))} record(s)",
            len(out.get("evidence", [])))
        if out.get("answer") is None:
            return None
        return {**out, "repaired": "resolved_discipline"}

    return None


# ==========================================================================
#  Adaptive loop
# ==========================================================================

def _relax_and_retry(state, graph, drop, usage):
    """Reissue the last graph query with one constraint removed. This is the
    recovery behaviour a fixed pipeline has no way to express: it turns
    'zero rows' from a dead end into a second, better-targeted attempt."""
    spec = dict(getattr(state, "last_spec", None) or {})
    if not spec:
        state.log_step("relax_query", "asked to relax a query but none had been issued yet",
                       {"drop": drop}, "nothing to relax", 0)
        return None

    key = {"discipline": "discipline", "date": "date", "venue": "venue",
           "games_id": "games_id", "event_name": "event_name_substr"}.get(drop)
    if not key or key not in spec:
        # Drop whatever optional constraint is actually present, most
        # brittle first — a model naming a constraint that isn't in the
        # spec shouldn't waste the recovery attempt.
        for candidate in ("date", "venue", "event_name_substr", "discipline"):
            if spec.get(candidate):
                key = candidate
                break
    if not key or not spec.get(key):
        state.log_step("relax_query", "no removable constraint left on the last query",
                       {"drop": drop}, "query is already minimally constrained", 0)
        return None

    relaxed = {k: v for k, v in spec.items() if k != key}
    t0 = time.time()
    result = sa.execute(relaxed, graph)
    state.log_step(
        "relax_query", f"last query returned nothing; dropping the '{key}' constraint "
                       f"and retrying — it was the most likely over-specification",
        {"dropped": key, "spec": {k: v for k, v in relaxed.items() if not k.startswith('_')}},
        f"relaxed query matched {len(result.get('evidence', []))} record(s); "
        f"answer={result.get('answer')!r}",
        len(result.get("evidence", [])), latency_s=time.time() - t0)

    for e in result.get("evidence", [])[:8]:
        state.add_evidence([EvidenceItem(
            step=state.step_count, source_agent="relax_query", kind="fact",
            ref_id=e["doc_id"], content=f"{e['title']}: {json.dumps({k: v for k, v in e.items() if k not in ('doc_id','url','medalists')})}",
            metadata={"medalists": e.get("medalists", [])})])
    state.last_spec = relaxed
    return result


def _adaptive_loop(state, graph, llm, embedder, usage, escalation_reason=""):
    """The plan -> act -> evaluate loop. Runs only for questions the free
    deterministic path could not verifiably answer.

    `loop_actions` counts work done INSIDE this loop, deliberately separate
    from state.step_count (which also counts the triage query and its
    verification). Gating the sufficiency check on it fixes a real failure:
    keyed to step_count, the check fired immediately after triage — before
    the agent had taken a single corrective action — and a small model
    asked "is this evidence sufficient?" answers "yes, confidence 1.0"
    about the very evidence that just failed verification. The agent then
    stopped having done nothing. Sufficiency is only a meaningful question
    once there is new evidence to judge.
    """
    tried = []
    last_graph_result = None
    loop_actions = 0
    evidence_at_start = len(state.evidence)

    while True:
        stop = state.should_stop()
        if stop:
            state.stopped_reason = stop
            break

        # Only ask "is this enough?" after the loop has actually gathered
        # something new to be judged.
        if (loop_actions >= 2 and loop_actions % 2 == 0
                and len(state.evidence) > evidence_at_start):
            verdict, r = agents.evidence_evaluation(state, graph, llm, embedder)
            usage.record("sufficiency_check", r)
            state.total_tokens += r.total_tokens
            if verdict.get("sufficient") and verdict.get("confidence", 0) >= 0.6:
                state.stopped_reason = (
                    f"agent judged evidence sufficient after {state.step_count} steps "
                    f"(confidence={verdict.get('confidence')})")
                break

        prompt = (f"Question: {state.question}\n\n"
                  f"Evidence so far ({len(state.evidence)} items):\n"
                  f"{state.evidence_text_block() or '(none yet)'}")
        r = llm.complete(
            PLANNER_SYSTEM.format(tried=tried or "(nothing yet)",
                                  linked=_format_linked(state),
                                  escalation_reason=escalation_reason or "(unknown)"),
            prompt, max_tokens=250, json_mode=True, context_text=state.evidence_text_block())
        usage.record("orchestrator_plan", r)

        plan_obj = parse_json_safely(r.text, default={"action": "answer", "args": {},
                                                      "reason": "planner output unparseable"})
        action = plan_obj.get("action", "answer")
        args = plan_obj.get("args", {}) or {}
        reason = plan_obj.get("reason", "")
        tried.append(action)

        if action == "answer":
            state.stopped_reason = "orchestrator judged the evidence sufficient to answer"
            state.log_step(action, reason, args, "chose to stop and answer", 0,
                           tokens=r.total_tokens, latency_s=r.latency_s)
            break

        if action == "graph_query":
            sub_q = args.get("question") or args.get("sub_question") or state.question
            last_graph_result, _ = _run_graph_query(state, graph, llm, sub_q, usage=usage)
            loop_actions += 1
            state.total_tokens += r.total_tokens
            continue

        if action == "relax_query":
            last_graph_result = _relax_and_retry(state, graph, args.get("drop", ""), usage)
            loop_actions += 1
            state.total_tokens += r.total_tokens
            continue

        if action == "multihop_reason":
            items, r2 = agents.multihop_reasoning(state, graph, llm, embedder)
            usage.record("multihop_reasoning", r2)
            n = state.add_evidence(items)
            loop_actions += 1
            state.log_step(action, reason, args, "added one explicit reasoning hop", n,
                           tokens=r.total_tokens + r2.total_tokens,
                           latency_s=r.latency_s + r2.latency_s)
            continue

        fn = agents.AGENT_REGISTRY.get(action)
        if fn is None:
            state.log_step(action, reason, args, "unknown action — skipped", 0,
                           tokens=r.total_tokens, latency_s=r.latency_s)
            continue

        try:
            items = fn(state, graph, llm, embedder, **args)
        except TypeError as e:
            state.log_step(action, reason, args, f"bad arguments for this tool ({e}) — skipped",
                           0, tokens=r.total_tokens, latency_s=r.latency_s)
            continue
        n = state.add_evidence(items)
        loop_actions += 1
        state.log_step(action, reason, args, f"retrieved {n} evidence item(s)", n,
                       tokens=r.total_tokens, latency_s=r.latency_s)

    return last_graph_result


def _format_linked(state):
    return {eid: e.get("name", "?") for eid, e in state.linked_entities.items()} or "(none)"


def _final_answer(state, llm, usage):
    prompt = (f"Question: {state.question}\n\n"
              f"All evidence gathered:\n{state.evidence_text_block()}")
    r = llm.complete(FINAL_ANSWER_SYSTEM, prompt, max_tokens=500, json_mode=True,
                     context_text=state.evidence_text_block())
    usage.record("final_answer", r)
    return parse_json_safely(r.text, default={"answer": None, "citations": [], "confidence": 0.3})


# ==========================================================================
#  Entry point
# ==========================================================================

def run(question: str, graph=None, llm=None, embedder=None, verbose=False,
        force_llm_planner=None, skip_triage=False) -> dict:
    """force_llm_planner / skip_triage exist for the ablation benchmark:
    they disable the free fast paths so the agent's true cost and accuracy
    can be measured without any template caching."""
    graph = graph or get_graph()
    embedder = embedder or Embedder()
    usage = UsageTracker()
    state = AgentState(question)
    state.last_spec = None
    t_start = time.time()

    escalated = True
    escalation_reason = "triage skipped"
    triage_result = None

    # ---- Step 0: free deterministic triage, then verify it ----
    if not skip_triage:
        triage_result, triage_spec = _run_graph_query(
            state, graph, llm, question, force_llm=force_llm_planner, usage=usage)
        ok, why = _verify_deterministically(triage_result, triage_spec, question)
        state.log_step("verify", "checking whether the deterministic result can be trusted "
                                 "without spending reasoning tokens",
                       {}, why, 0)
        if ok:
            escalated, escalation_reason = False, why
            state.stopped_reason = (
                "deterministic graph query resolved and passed verification — "
                "no agentic reasoning needed")
        else:
            escalation_reason = why

    # ---- Resolved for free ----
    if not escalated:
        return _result(state, triage_result["answer"], triage_result["matched_doc_ids"],
                       0.95, usage, graph, escalated=False,
                       escalation_reason=escalation_reason, t_start=t_start)

    # ---- Try a free deterministic repair before paying for reasoning ----
    if triage_result is not None or state.last_spec:
        repaired = _repair_query(state, graph, state.last_spec, question, escalation_reason)
        if repaired is not None:
            ok2, why2 = _verify_deterministically(repaired, None, question)
            state.log_step("verify", "re-checking the repaired query's result", {}, why2, 0)
            if ok2:
                state.stopped_reason = (
                    f"escalated ({escalation_reason[:60]}...) then repaired deterministically "
                    f"via {repaired['repaired']} — resolved without reasoning tokens")
                return _result(state, repaired["answer"], repaired["matched_doc_ids"],
                               0.9, usage, graph, escalated=True,
                               escalation_reason=escalation_reason, t_start=t_start,
                               repair=repaired["repaired"])

    # ---- Needs investigation, but no LLM available ----
    if llm is None:
        best = triage_result.get("answer") if triage_result else None
        state.stopped_reason = (
            f"escalation needed ({escalation_reason}) but no LLM is configured; "
            f"reporting the unverified deterministic result" if best else
            f"escalation needed ({escalation_reason}) and no LLM is configured")
        return _result(state, best,
                       triage_result.get("matched_doc_ids", []) if triage_result else [],
                       0.35 if best else 0.0, usage, graph, escalated=True,
                       escalation_reason=escalation_reason, t_start=t_start)

    # ---- Adaptive investigation ----
    last = _adaptive_loop(state, graph, llm, embedder, usage, escalation_reason)

    final = _final_answer(state, llm, usage)
    answer = final.get("answer")
    citations = final.get("citations") or []
    confidence = final.get("confidence", 0.5)

    # The generator refusing to commit does not mean the investigation found
    # nothing — prefer a concrete graph result over a null when one exists.
    if answer in (None, "", "null") and last and last.get("answer") is not None:
        answer = last["answer"]
        citations = last.get("matched_doc_ids", [])
        confidence = 0.45

    if not citations:
        citations = [e.ref_id for e in state.evidence][:10]

    return _result(state, answer, citations, confidence, usage, graph,
                   escalated=True, escalation_reason=escalation_reason, t_start=t_start)


def _result(state, answer, citations, confidence, usage, graph, escalated,
            escalation_reason, t_start, repair=None):
    """Assembles the full agentic-trace record. Everything the hackathon's
    'Trace & Agentic Behavior' section asks for is here explicitly, rather
    than left for a reader to infer from a log."""
    d = state.to_dict()
    trace = d["trace"]
    methods = [t["action"] for t in trace]
    d.update({
        "pipeline": "agentic_graphrag",
        "answer": answer,
        "citations": citations,
        "matched_doc_ids": citations,
        "confidence": confidence,
        "usage": usage.summary(),
        "retrieval_backend": graph.backend,
        # --- agentic behaviour metrics ---
        "escalated": escalated,
        "escalation_reason": escalation_reason,
        "repair": repair,
        "retrieval_methods_used": sorted(set(methods)),
        "agents_invoked": sorted({t["action"] for t in trace
                                  if t["action"] not in ("verify", "answer")}),
        "num_strategy_changes": sum(1 for t in trace if t.get("strategy_changed")),
        "num_citations": len(citations),
        "num_evidence_items": len(state.evidence),
        "wall_time_s": round(time.time() - t_start, 3),
    })
    return d
