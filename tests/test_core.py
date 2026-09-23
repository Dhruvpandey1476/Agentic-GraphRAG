"""
Tests for the parts where a silent regression would corrupt the benchmark
rather than crash it — which is the dangerous kind here. A pipeline that
throws gets noticed; a grounding check that stops firing just turns into
confidently wrong answers with a clean-looking trace.

Run:  python -m pytest tests/ -q
The graph-backed tests skip themselves if outputs/graph_store.json hasn't
been built yet, so a fresh clone can still run the suite.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ingestion.infobox import (games_id, normalize, contains_phrase,
                                   parse_infobox, parse_title, build_games_chronology)
from src.agents import query_planner as qp
from src.agents import structured_agent as sa
from src.agents.orchestrator import _check_query_grounding, _verify_deterministically
from src.llm_client import parse_json_safely, estimate_tokens
from src.embeddings import cosine_sim


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def graph():
    store = os.path.join(config.OUTPUT_DIR, "graph_store.json")
    if not os.path.exists(store):
        pytest.skip("graph not built — run: python -m src.ingestion.build_graph")
    from src.tigergraph_client import MockGraph
    g = MockGraph()
    if not g.store.get("olympic_events"):
        pytest.skip("graph has no structured events")
    return g


# ---------------------------------------------------------------- infobox

def test_games_id_format():
    assert games_id(2012, "Summer") == "2012-summer"
    assert games_id(2018, "winter") == "2018-winter"


def test_normalize_strips_dash_variants():
    # The corpus uses en-dashes in titles; matching must not depend on which
    # dash character a question happens to use.
    assert normalize("Men's K-2 1,000 m") == normalize("mens k2 1000 m")
    assert normalize("A – B") == normalize("A - B")


def test_contains_phrase_is_word_bounded():
    assert contains_phrase("Olympic Tennis Centre", "tennis centre")
    assert not contains_phrase("Athletics", "ath")


def test_parse_infobox_collects_stacked_boxes():
    # Tennis pages stack a second infobox directly after the first with no
    # blank line; fields from both must be captured.
    text = ("[Infobox tennis event]\n  name: X\n[Infobox Olympic event]\n"
            "  competitors: 64\n  nations: 31\n\nBody text here.")
    itype, fields = parse_infobox(text, prefer_type="Olympic event")
    assert fields["competitors"] == "64"
    assert fields["nations"] == "31"


def test_parse_title():
    p = parse_title("Athletics at the 2008 Summer Olympics – Men's decathlon")
    assert p["discipline"] == "Athletics" and p["year"] == 2008
    assert p["season"] == "Summer" and p["event_name"] == "Men's decathlon"


def test_chronology_skips_other_season():
    events = [{"games_id": games_id(y, s), "year": y, "season": s}
              for y, s in [(2012, "Summer"), (2014, "Winter"), (2016, "Summer")]]
    for e in events:
        e.setdefault("discipline", "X")
    chrono = build_games_chronology(events)
    assert chrono["2016-summer"]["prev_id"] == "2014-winter"


# ---------------------------------------------------------------- planner

@pytest.mark.parametrize("q,expected", [
    ("According to the provided corpus, how many biathlon events at the 2018 Winter "
     "Olympics had more than 73 competitors?", "aggregation"),
    ("Which athletics event at the 2008 Summer Olympics had the highest number of "
     "competitors?", "superlative"),
    ("Who won the gold medal in the men's 20 kilometres walk athletics event at the "
     "Summer Olympics held immediately before 2016?", "temporal"),
    ("Who won the gold medal in the event held at Olympic Tennis Centre on 15 to 22 "
     "August 2004?", "venue_date"),
    ("How many nations competed in Fencing at the 1988 Summer Olympics – Men's foil?",
     "lookup_field"),
])
def test_regex_planner_types(q, expected):
    spec = qp.compile_regex(q, ["Athletics", "Biathlon", "Fencing"])
    assert spec is not None and spec["type"] == expected


def test_planner_marks_provenance():
    spec, _ = qp.plan("how many nations competed in X?", [], force_llm=False)
    assert spec["_planner"] == "regex"


def test_llm_planner_is_the_default():
    """The regex cache must not be reachable without asking for it: with no
    LLM available and no fast-path opt-in, planning yields nothing rather
    than silently falling back to the template matcher."""
    spec, _ = qp.plan("how many nations competed in X?", [])
    assert spec["_planner"] == "none" and spec["type"] == "unstructured"


def test_planner_coerces_numeric_strings():
    """An LLM emits "73" as happily as 73 — valid JSON, wrong type. The
    executor then raises TypeError deep inside a filter comparison, which is
    how a whole run lost questions. The regex planner never did this, so it
    only surfaced once the LLM compiler became the default."""
    from src.agents.query_planner import _coerce_numerics
    assert _coerce_numerics({"min_competitors": "73"})["min_competitors"] == 73
    assert _coerce_numerics({"min_competitors": "1,200"})["min_competitors"] == 1200
    assert _coerce_numerics({"target_year": "2016.0"})["target_year"] == 2016
    assert _coerce_numerics({"min_competitors": 41.0})["min_competitors"] == 41
    # unusable values are dropped, not passed through to crash the executor
    assert "min_competitors" not in _coerce_numerics({"min_competitors": "many"})
    assert "min_competitors" not in _coerce_numerics({"min_competitors": ["73"]})


def test_planner_repairs_games_id_shapes():
    for raw in ("2012Summer", "2012 summer", "2012-Summer"):
        fixed = qp._repair_games_id({"type": "aggregation", "games_id": raw})
        assert fixed["games_id"] == "2012-summer", raw
    fixed = qp._repair_games_id({"type": "aggregation", "year": 2018, "season": "Winter"})
    assert fixed["games_id"] == "2018-winter"


# ---------------------------------------------------------------- grounding

def test_grounding_rejects_invented_discipline():
    """The regression that matters most: a well-formed query filtering on a
    value the question never mentions."""
    spec = {"type": "aggregation", "discipline": "Athletics",
            "games_id": "2004-summer", "min_competitors": 41}
    q = "In the discipline that held the most events at the 2004 Summer Olympics, " \
        "how many of its events had more than 41 competitors?"
    ok, why = _check_query_grounding(spec, q)
    assert not ok and "discipline" in why


def test_grounding_accepts_stated_discipline():
    spec = {"type": "aggregation", "discipline": "Biathlon",
            "games_id": "2018-winter", "min_competitors": 73}
    q = "how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?"
    ok, _ = _check_query_grounding(spec, q)
    assert ok


def test_grounding_rejects_misread_threshold():
    spec = {"type": "aggregation", "discipline": "Athletics",
            "games_id": "2004-summer", "min_competitors": 42}
    q = "how many athletics events at the 2004 Summer Olympics had more than 41 competitors?"
    ok, why = _check_query_grounding(spec, q)
    assert not ok and "41" in why


def test_grounding_rejects_partial_year_coverage():
    spec = {"type": "aggregation", "discipline": "Badminton", "games_id": "1996-summer"}
    q = "Did badminton have more events with over 30 competitors at the 2008 or the " \
        "1996 Summer Olympics?"
    ok, why = _check_query_grounding(spec, q)
    assert not ok and "distinct years" in why


def test_verify_escalates_on_ambiguity():
    ok, why = _verify_deterministically(
        {"answer": "X", "matched_doc_ids": ["d1"], "ambiguous": True,
         "tied_candidates": ["d1", "d2"]}, {}, "q")
    assert not ok and "tied" in why


def test_verify_escalates_on_uncitable_answer():
    ok, why = _verify_deterministically(
        {"answer": "0", "matched_doc_ids": []}, {}, "q")
    assert not ok


def test_verify_passes_clean_result():
    ok, _ = _verify_deterministically(
        {"answer": "5", "matched_doc_ids": ["d1"], "ambiguous": False}, {}, "q")
    assert ok


# ---------------------------------------------------------------- executor

def test_aggregation_matches_known_gold(graph):
    spec = {"type": "aggregation", "discipline": "Biathlon",
            "games_id": "2018-winter", "min_competitors": 73}
    assert sa.execute(spec, graph)["answer"] == "5"


def test_temporal_resolves_previous_summer_games(graph):
    spec = {"type": "temporal", "discipline": "Athletics",
            "event_name_substr": "men's 20 kilometres walk",
            "season": "Summer", "target_year": 2016, "direction": "before",
            "field": "gold"}
    assert sa.execute(spec, graph)["answer"] == "Chen Ding"


def test_discipline_superlative_is_a_real_argmax(graph):
    r = sa.execute({"type": "discipline_superlative", "games_id": "2004-summer",
                    "direction": "max"}, graph)
    counts = r["counts"]
    assert counts[r["answer"]] == max(counts.values())


@pytest.mark.parametrize("spec,expect_in_note", [
    ({"type": "temporal", "discipline": "Rowing", "field": "nations"}, "target_year"),
    ({"type": "venue_date", "venue": "Somewhere"}, "date"),
    ({"type": "aggregation"}, "entire corpus"),
    ({"type": "discipline_superlative"}, "games_id"),
    ({"type": "venue_games"}, "venue"),
])
def test_executor_validates_incomplete_specs(graph, spec, expect_in_note):
    """A compiled spec is untrusted input. The model omits fields, and
    _coerce_numerics drops values it cannot parse — either way the executor
    would reach `year < None` and kill the question. It must return an
    honest empty result the grounding check can act on instead."""
    r = sa.execute(spec, graph)
    assert r["answer"] is None
    assert expect_in_note in r.get("note", "")


def test_resolve_relative_games_guards_itself(graph):
    """The function doing the arithmetic guards its own inputs rather than
    trusting every call site."""
    from src.agents.structured_agent import _resolve_relative_games
    chrono = graph.games_chronology()
    assert _resolve_relative_games(chrono, None, 2016) is None
    assert _resolve_relative_games(chrono, "Summer", None) is None
    assert _resolve_relative_games(chrono, "Summer", 2016) == "2012-summer"


def test_executor_returns_none_not_zero_when_nothing_matches(graph):
    """A lookup that finds nothing must return None so the agent keeps
    investigating — returning a 0 or "" would be read as a real answer."""
    r = sa.execute({"type": "lookup_field", "title": "No Such Event Ever", "field": "nations"},
                   graph)
    assert r["answer"] is None


def test_public_set_accuracy_does_not_regress(graph):
    """End-to-end guard on the REGEX path over the real eval set.

    force_llm=False is explicit because the regex parser is no longer the
    default — the LLM compiles every query now, so that all three pipelines
    spend real tokens and the benchmark measures architecture rather than a
    prior on the question templates. This test still pins the cached path,
    which remains available via --fast-path.
    """
    path = config.EVAL_QUESTIONS_PATH
    if not os.path.exists(path):
        pytest.skip("eval_public.jsonl missing")
    qs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    disciplines = graph.known_disciplines()
    hits = 0
    for q in qs:
        spec, _ = qp.plan(q["question"], disciplines, force_llm=False)
        if spec["type"] == "unstructured":
            continue
        ans = sa.execute(spec, graph)["answer"]
        if ans is not None and any(normalize(str(ans)) == normalize(str(g))
                                   for g in q["answer"]):
            hits += 1
    assert hits >= 95, f"deterministic path regressed to {hits}/{len(qs)}"


# ---------------------------------------------------------------- utilities

def test_parse_json_safely_recovers_from_prose_wrapping():
    """Small local models wrap JSON in prose even when told not to."""
    assert parse_json_safely('Sure! Here you go: {"a": 1} Hope that helps.') == {"a": 1}
    assert parse_json_safely('```json\n{"b": 2}\n```') == {"b": 2}
    assert parse_json_safely("not json at all", default={"x": 0}) == {"x": 0}


def test_estimate_tokens_never_zero_for_text():
    """A zero here would silently corrupt the token-efficiency comparison."""
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0


def test_cosine_sim_handles_mixed_sparse_and_dense():
    import numpy as np
    dense = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sparse = {"0": 1.0}
    assert cosine_sim(dense, dense) == pytest.approx(1.0)
    assert cosine_sim(sparse, sparse) == pytest.approx(1.0)
    assert cosine_sim(dense, sparse) == pytest.approx(1.0)
    assert cosine_sim(dense, np.array([0.0, 1.0, 0.0], dtype=np.float32)) == pytest.approx(0.0)


def test_backends_expose_the_same_interface():
    """RAG silently returned nothing against TigerGraph because it reached
    into MockGraph's .store directly. Both backends must satisfy one
    interface so that can't recur."""
    from src.tigergraph_client import BaseGraph, MockGraph, RealGraph
    required = [name for name, v in vars(BaseGraph).items()
                if not name.startswith("_") and callable(v)]
    assert required, "BaseGraph declares no interface methods"
    for impl in (MockGraph, RealGraph):
        for m in required:
            assert callable(getattr(impl, m, None)), f"{impl.__name__} is missing {m}()"
        assert isinstance(getattr(impl, "backend", None), str)


# ---------------------------------------------------------------- matching semantics

@pytest.mark.parametrize("haystack,needle,expected", [
    # The bug that live TigerGraph exposed: GSQL LIKE is plain substring
    # containment, so a men's event query silently matched women's events
    # and returned the wrong medallist on 4 of 100 questions.
    ("Women's 20 kilometres walk", "men's 20 kilometres walk", False),
    ("Men's 20 kilometres walk", "men's 20 kilometres walk", True),
    ("Women's pole vault", "women's pole vault", True),
    ("Men's 10 km sprint", "women's 10 km sprint", False),
    # Weight classes: '+' is significant, "+80 kg" is not "80 kg".
    ("+80 kg", "80 kg", False),
    ("80 kg", "80 kg", True),
])
def test_search_key_has_word_boundary_semantics(haystack, needle, expected):
    """search_key() is the single definition of matching shared by both
    backends. Padded containment must behave exactly like contains_phrase,
    or TigerGraph and the in-memory store drift apart again."""
    from src.ingestion.infobox import search_key
    padded = search_key(needle) in search_key(haystack)
    assert padded is expected
    # and it must agree with the Python-side matcher it mirrors
    assert padded == contains_phrase(haystack, needle)


def test_search_key_is_space_padded():
    from src.ingestion.infobox import search_key
    k = search_key("Men's foil")
    assert k.startswith(" ") and k.endswith(" ")


@pytest.mark.parametrize("q,expected_type", [
    ("how many nations competed in Fencing at the 1988 Summer Olympics – Men's foil?",
     "lookup_field"),
])
def test_regex_planner_survives_endash_titles(q, expected_type):
    """The corpus uses en-dashes in page titles; the repo previously crashed
    on Windows reading them at all (no encoding="utf-8" on open())."""
    spec = qp.compile_regex(q, ["Fencing"])
    assert spec and spec["type"] == expected_type
