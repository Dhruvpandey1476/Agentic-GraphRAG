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

Only these actions are available on this graph — anything else does not exist
here and must not be chosen: {available}

Actions already tried: {tried}
Linked entities: {linked}

Return ONLY JSON: {{"action": str, "args": {{}}, "reason": str}}
`reason` must state what specific gap this action closes — one sentence,
specific to this question, not a generic justification."""


REPLAN_SYSTEM = """You compiled a graph query for a question and it was
REJECTED. Compile a corrected one.

The rejection reason is precise and literal — it names exactly what is wrong.
The usual causes:
  * you filtered on a value the question never states (you guessed it). If the
    question refers to something indirectly ("the discipline that held the most
    events"), you cannot name it — instead emit a query that RESOLVES it, such
    as discipline_superlative.
  * you used a number the question does not contain. Copy the number from the
    question exactly.
  * you covered only part of the question (one edition when two were named).
  * you omitted a field the query type requires.

Emit the SAME JSON shapes as before, corrected. Output ONLY the JSON."""


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



# The keys each spec type's executor branch actually reads. A key outside
# its type's set is not a harmless extra: the branch ignores it silently,
# so a spec that is the wrong SHAPE for the type it declares still runs and
# still returns rows. llama3 emitted aggregation(title="How many shooting
# events at the 2004 Summer Olympics had more than 37 competitors?",
# field="competitors", games_id="2004-summer") with no discipline at all --
# `title` and `field` were dropped on the floor, the discipline constraint
# was never applied, and the query counted all 79 events at that edition
# instead of the 8 shooting ones. Checking the spec against its own type
# turns that into a rejection the planner can be told about.
_SPEC_KEYS = {
    "lookup_field": {"title", "field"},
    "lookup_nations": {"title", "field"},
    "medalist": {"title", "medal"},
    "aggregation": {"discipline", "games_id", "min_competitors",
                    "max_competitors", "venue_substr"},
    "superlative": {"discipline", "games_id", "metric", "direction"},
    "temporal": {"discipline", "event_name_substr", "season", "target_year",
                 "direction", "field"},
    "venue_date": {"venue", "date", "games_id", "field"},
    "nation_medals": {"noc", "games_id", "discipline", "medal"},
    "comparison": {"sub", "games_id_a", "games_id_b", "want"},
    "discipline_superlative": {"games_id", "direction", "min_competitors"},
    "chronology_lookup": {"season", "target_year", "direction"},
    "venue_games": {"venue", "discipline"},
    "unstructured": set(),
}
# The key that carries each query's SUBJECT. A spec that has shed all of
# these has nothing left to constrain it but the Games edition.
_TYPE_SUBJECT_KEYS = {
    "aggregation": ("discipline", "venue_substr"),
    "superlative": ("discipline",),
    "lookup_field": ("title",),
    "lookup_nations": ("title",),
    "medalist": ("title",),
    "temporal": ("event_name_substr", "discipline"),
    "venue_date": ("venue",),
    "nation_medals": ("noc",),
    "venue_games": ("venue",),
}

# Bookkeeping the compiler adds; never part of the query itself.
_META_SPEC_KEYS = {"type", "_planner"}

# A question that says "men's" is not answered by the women's event. The
# qualifier is a constraint like any other, and dropping it from the
# event-name filter silently returns the wrong medallist -- which is a
# failure that looks like a perfectly good answer.
_QUALIFIER_RE = re.compile(r"\b(women's|men's|mixed)\b")
# What a compiled query RETURNS, used for answer-shape grounding below.
_COUNT_SPECS = ("aggregation",)
_NAME_SPECS = ("superlative", "medalist", "discipline_superlative", "venue_games")

_COUNT_CUES = ("how many", "number of", "how much", "count of")
_NAME_CUES = ("which", "what event", "what discipline", "who ", "whom", "name the")

# "the highest NUMBER OF competitors" names the METRIC being ranked on; it
# is not a request for a number. Left in, it makes every "which event had
# the most ..." question look like a counting question and the shape rule
# below never fires on the very case it exists for.
_METRIC_PHRASE = re.compile(
    r"\b(highest|greatest|largest|biggest|most|lowest|smallest|fewest|least)\s+"
    r"(number|amount|count)\s+of\b")


def _expected_answer_shape(question):
    """'count' | 'name' | None — what KIND of thing the question asks for.

    Counting cues outrank naming cues, because a chained question states
    both: "in the discipline that held the most events at the 2004 Summer
    Olympics, how many of its events had more than 41 competitors" names
    its subject by description and then asks for a number. The outermost
    ask is the count, and that is the shape the final answer must have.
    """
    q = _METRIC_PHRASE.sub(" ", " " + (question or "").lower().strip() + " ")
    if any(c in q for c in _COUNT_CUES):
        return "count"
    if any(c in q for c in _NAME_CUES):
        return "name"
    return None


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

    # 0. SHAPE VALIDITY. The executor reads only the keys its branch knows
    #    and ignores the rest without complaint, so a spec that is the
    #    wrong shape for the type it declares still runs and still returns
    #    rows. That is harmless while the query keeps its subject: an
    #    aggregation carrying a stray field="competitors" alongside a real
    #    discipline is simply noisy. It is fatal when the stray keys are
    #    where the subject WENT -- llama3 emitted aggregation(title="How
    #    many shooting events at the 2004 Summer Olympics had more than 37
    #    competitors?", field="competitors", games_id="2004-summer") with
    #    no discipline at all, so the discipline constraint was never
    #    applied and the query counted all 79 events at that edition rather
    #    than the 8 shooting ones. Rejecting only the subject-losing case
    #    keeps the rule from firing on specs that are merely untidy.
    stype_declared = spec.get("type")
    allowed = _SPEC_KEYS.get(stype_declared)
    if allowed is not None:
        stray = sorted(set(spec) - allowed - _META_SPEC_KEYS)
        subject = _TYPE_SUBJECT_KEYS.get(stype_declared, ())
        if stray and subject and not any(spec.get(k) for k in subject):
            return False, (f"the query declares type={stype_declared!r} but carries {stray}, "
                           f"which that type does not read, and it names none of "
                           f"{list(subject)} — the subject of the question ended up in a key "
                           f"the query ignores, so it would run unconstrained")

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
    for key in ("min_competitors", "max_competitors", "target_year"):
        val = spec.get(key)
        if isinstance(val, int) and str(val) not in q_numbers:
            return False, (f"the query used {key}={val}, a number the question never mentions "
                           f"(it names {sorted(q_numbers)}) — the threshold was misread")

    # 3. ANSWER-SHAPE GROUNDING. A query can filter on nothing but values
    #    the question actually states and still answer a different
    #    question, because it returns the wrong KIND of thing. "Which
    #    athletics event at the 2008 Summer Olympics had the highest
    #    number of competitors" compiled to aggregation(discipline=
    #    "Athletics", games_id="2008-summer") — every literal grounded,
    #    real rows, real citations, and it answered "43": the NUMBER of
    #    athletics events, to a question asking for an event's NAME.
    #    Rules 1 and 2 both pass it, and a reader shown only the answer
    #    cannot tell. Comparing the question's interrogative form against
    #    what the spec type returns catches it for zero tokens, and tests
    #    the one thing structural validity never does — whether the query
    #    answers the question that was asked.
    # 2b. QUALIFIER COVERAGE. Same principle as the year check, applied to
    #     the one constraint that is invisible in a result: a question
    #     asking about the men's event is not answered by the women's.
    #     llama3 compiled "the men's 20 kilometres walk" to
    #     event_name_substr="20 kilometres walk" -- grounded (every word of
    #     it appears in the question) but missing the qualifier, matching
    #     the women's event, and returning that medallist as a confident
    #     exact answer.
    q_quals = set(_QUALIFIER_RE.findall(question.lower()))
    if q_quals:
        for key in ("event_name_substr", "title"):
            val = spec.get(key)
            if not isinstance(val, str) or not val:
                continue
            dropped = q_quals - set(_QUALIFIER_RE.findall(val.lower()))
            if dropped:
                return False, (f"the question asks about the {sorted(dropped)[0]} event but "
                               f"{key}={val!r} does not carry that qualifier — the query "
                               f"would match the other event and return the wrong record")
    want = _expected_answer_shape(question)
    stype = spec.get("type")
    if want == "name" and stype in _COUNT_SPECS:
        return False, (f"the question asks which/who, so it wants a name, but a {stype!r} "
                       f"query returns a count — the answer shape does not match the "
                       f"question")
    if want == "count" and stype in _NAME_SPECS:
        return False, (f"the question asks how many, so it wants a count, but a {stype!r} "
                       f"query returns a name — the answer shape does not match the "
                       f"question")
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

    # ---- wrong answer shape: a counting query for a "which ..." question ----
    if "answer shape does not match" in reason and "shape" not in _fixed:
        if _expected_answer_shape(question) != "name" or spec.get("type") != "aggregation":
            return None
        ql = question.lower()
        direction = "min" if any(w in ql for w in _SUPERLATIVE_MIN) else "max"
        metric = ("nations" if "nation" in ql else
                  "teams" if "team" in ql else "competitors")
        # Rule 1 ran first and passed, so discipline/games_id are already
        # known to be grounded in the question; only the shape was wrong.
        fixed = {k: v for k, v in spec.items() if k in ("discipline", "games_id")}
        fixed.update({"type": "superlative", "metric": metric, "direction": direction})
        out = sa.execute(fixed, graph)
        state.log_step(
            "repair_shape",
            "the question asks WHICH event, but the compiled query counted events; "
            "re-issuing it as an argmax over the same scope so the answer is the event "
            "the question asked for rather than how many there were",
            {"was": spec.get("type"), "now": "superlative",
             "metric": metric, "direction": direction},
            f"{out.get('answer')!r} from {len(out.get('evidence', []))} record(s)"
            + (" (tied — not usable)" if out.get("ambiguous") else ""),
            len(out.get("evidence", [])))
        if out.get("answer") is None or out.get("ambiguous"):
            return None
        return {**out, "repaired": "answer_shape"}
    # ---- mis-stated target_year, recoverable from the question ----
    if "target_year=" in reason and "the question never mentions" in reason \
            and "year" not in _fixed:
        years = sorted({int(y) for y in _YEAR_RE.findall(question)})
        if len(years) != 1:
            return None
        fixed = {**spec, "target_year": years[0]}
        fixed.pop("_planner", None)
        out = sa.execute(fixed, graph)
        state.log_step(
            "repair_literal",
            "the compiled query's target_year is not a year the question names; "
            "substituting the only year it does name",
            {"was": spec.get("target_year"), "now": years[0]},
            f"re-ran with target_year={years[0]}: {out.get('answer')!r} from "
            f"{len(out.get('evidence', []))} record(s)",
            len(out.get("evidence", [])))
        if out.get("answer") is None:
            return None
        ok, why = _check_query_grounding(fixed, question)
        if not ok:
            nested = _repair_query(state, graph, fixed, question, why, _fixed | {"year"})
            return nested if nested else None
        return {**out, "repaired": "literal_year"}

    # ---- dropped men's/women's qualifier, recoverable from the question ----
    if "does not carry that qualifier" in reason and "qualifier" not in _fixed:
        quals = set(_QUALIFIER_RE.findall(question.lower()))
        key = "event_name_substr" if spec.get("event_name_substr") else "title"
        val = spec.get(key)
        if len(quals) != 1 or not isinstance(val, str) or not val:
            return None
        qual = quals.pop()
        fixed = {**spec, key: f"{qual} {val}"}
        fixed.pop("_planner", None)
        out = sa.execute(fixed, graph)
        state.log_step(
            "repair_literal",
            "the question names the " + qual + " event but the compiled filter dropped "
            "that qualifier, so it would match the other event; restoring it",
            {"was": val, "now": fixed[key]},
            f"re-ran with {key}={fixed[key]!r}: {out.get('answer')!r} from "
            f"{len(out.get('evidence', []))} record(s)",
            len(out.get("evidence", [])))
        if out.get("answer") is None:
            return None
        ok, why = _check_query_grounding(fixed, question)
        if not ok:
            nested = _repair_query(state, graph, fixed, question, why,
                                   _fixed | {"qualifier"})
            return nested if nested else None
        return {**out, "repaired": "literal_qualifier"}
    # ---- misread numeric threshold, recoverable from the question ----
    if "a number the question never mentions" in reason and "competitors=" in reason:
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


def _replan_with_feedback(state, graph, llm, question, reason, usage):
    """One targeted recompile: hand the compiler its own rejection reason.

    Cheaper and more accurate than the generic loop, because the grounding
    check has already localised the fault — there is no search to do, only a
    correction to make. Returns an executed result, or None if the retry is
    no better than the original.
    """
    t0 = time.time()
    disciplines = graph.known_disciplines()
    prior = {k: v for k, v in (state.last_spec or {}).items() if not k.startswith("_")}
    gap = chr(10) * 2
    prompt = (f"QUESTION: {question}{gap}"
              f"KNOWN DISCIPLINES: {', '.join(disciplines[:120])}{gap}"
              f"YOUR REJECTED QUERY: {json.dumps(prior)}{gap}"
              f"WHY IT WAS REJECTED: {reason}")
    r = llm.complete(REPLAN_SYSTEM, prompt, max_tokens=1200, json_mode=True)
    usage.record("replan_with_feedback", r)
    state.total_tokens += r.total_tokens

    spec = parse_json_safely(r.text, default={})
    if not isinstance(spec, dict) or "type" not in spec:
        state.log_step("replan", "asked the compiler to correct its rejected query",
                       {}, "the correction was unparseable", 0,
                       tokens=r.total_tokens, latency_s=r.latency_s)
        return None

    spec = qp._repair_games_id(qp._coerce_numerics(spec))
    # A "correction" identical to what was just rejected is not a correction.
    if {k: v for k, v in spec.items() if not k.startswith("_")} == prior:
        state.log_step("replan", "asked the compiler to correct its rejected query",
                       {}, "it returned the same query unchanged", 0,
                       tokens=r.total_tokens, latency_s=r.latency_s)
        return None

    result = sa.execute(spec, graph)
    state.last_spec = spec
    state.log_step(
        "replan", f"recompiled after rejection: {reason[:70]}",
        {"spec": {k: v for k, v in spec.items() if not k.startswith("_")}},
        f"corrected query answered {result.get('answer')!r} from "
        f"{len(result.get('evidence', []))} record(s)",
        len(result.get("evidence", [])), tokens=r.total_tokens, latency_s=time.time() - t0)

    if result.get("answer") is None:
        return None
    for e in result.get("evidence", [])[:8]:
        state.add_evidence([EvidenceItem(
            step=state.step_count, source_agent="replan", kind="fact",
            ref_id=e["doc_id"],
            content=f"{e['title']}: " + json.dumps(
                {k: v for k, v in e.items() if k not in ("doc_id", "url", "medalists")}),
            metadata={"medalists": e.get("medalists", [])})])
    return result


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
    # Actions that have already returned nothing. The prompt asks the model
    # not to repeat them and the model does anyway, so this is enforced here
    # rather than requested there.
    barren = set()
    tools = _available_tools(graph)

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
                                  available=", ".join(sorted(tools - barren)),
                                  escalation_reason=escalation_reason or "(unknown)"),
            prompt, max_tokens=1200, json_mode=True, context_text=state.evidence_text_block())
        usage.record("orchestrator_plan", r)

        plan_obj = parse_json_safely(r.text, default={})
        action = plan_obj.get("action")
        args = plan_obj.get("args", {}) or {}
        reason = plan_obj.get("reason", "")

        if not action:
            # An unusable plan is not a reason to stop: retry the question as
            # a graph query, which is the single most likely useful move, and
            # only give up if that has already been tried.
            if "graph_query" not in barren:
                action, args = "graph_query", {"question": state.question}
                reason = "planner returned nothing usable; retrying the graph query"
            else:
                action, reason = "answer", "planner returned nothing usable twice"

        if action in barren:
            state.log_step(action, reason, args,
                           f"skipped: {action} already returned nothing this run", 0,
                           tokens=r.total_tokens, latency_s=r.latency_s)
            if barren >= (tools - {"answer"}):
                state.stopped_reason = "every available tool has been tried without result"
                break
            continue
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
        if n == 0:
            barren.add(action)
        state.log_step(action, reason, args,
                       f"retrieved {n} evidence item(s)"
                       + (" — not offering this tool again" if n == 0 else ""), n,
                       tokens=r.total_tokens, latency_s=r.latency_s)

    return last_graph_result


def _available_tools(graph):
    """Tools worth offering the planner, given what this graph actually holds.

    Offering a tool that cannot succeed is worse than not having it: the
    model picks it, gets nothing, and — having no better idea — picks it
    again. Observed in a live trace, the agent called entity_linking three
    times in a row for 2,736 tokens against an Entity layer that is empty,
    so every call was guaranteed to return nothing.
    """
    tools = {"graph_query", "relax_query", "multihop_reason", "answer"}
    try:
        if graph.all_chunk_count() > 0:
            tools.add("similarity_search")
    except Exception:
        pass
    try:
        # entity_linking / traversal / document_retrieval all start from an
        # Entity lookup, so they stand or fall together.
        #
        # Count the rows, do not just test truthiness: RealGraph returns
        # [{"Result": []}] for an empty lookup, which is a non-empty list and
        # therefore truthy. Testing the wrapper rather than its contents
        # enabled all three tools against an empty Entity layer — exactly the
        # bug this function exists to prevent.
        if _entity_rows(graph) > 0:
            tools |= {"entity_linking", "graph_traversal", "document_retrieval"}
    except Exception:
        pass
    return tools


def _entity_rows(graph) -> int:
    """How many Entity vertices the graph actually holds."""
    try:
        if graph.backend == "tigergraph":
            return int(graph.conn.getVertexCount("Entity"))
        return len(graph.store.get("entities", {}))
    except Exception:
        probe = graph.entity_lookup("a") or []
        if probe and isinstance(probe[0], dict) and "Result" in probe[0]:
            return len(probe[0]["Result"])
        return len(probe)


def _format_linked(state):
    return {eid: e.get("name", "?") for eid, e in state.linked_entities.items()} or "(none)"


def _final_answer(state, llm, usage):
    prompt = (f"Question: {state.question}\n\n"
              f"All evidence gathered:\n{state.evidence_text_block()}")
    r = llm.complete(FINAL_ANSWER_SYSTEM, prompt, max_tokens=1200, json_mode=True,
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

    # ---- Ask the compiler to fix its own query, given the diagnostic ----
    #
    # The deterministic repairs cover three failure classes exactly. Anything
    # else previously fell straight into the generic plan->act loop, which is
    # both expensive and vague — it asks "what should I do next?" when we
    # already know precisely what went wrong. Handing the compiler its own
    # rejection reason is a far more direct instruction, and it is the natural
    # use of a diagnostic that already names the faulty slot.
    if llm is not None and state.last_spec:
        fixed = _replan_with_feedback(state, graph, llm, question, escalation_reason, usage)
        if fixed is not None:
            ok3, why3 = _verify_deterministically(fixed, state.last_spec, question)
            state.log_step("verify", "re-checking the recompiled query", {}, why3, 0)
            if ok3:
                state.stopped_reason = (
                    "escalated, then the compiler corrected its own query when given "
                    "the rejection reason")
                return _result(state, fixed["answer"], fixed["matched_doc_ids"],
                               0.85, usage, graph, escalated=True,
                               escalation_reason=escalation_reason, t_start=t_start,
                               repair="replan_with_feedback")

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
