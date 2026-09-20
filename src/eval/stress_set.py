"""
Generates `data/eval_stress.jsonl` — a held-out benchmark of questions
that a SINGLE graph query structurally cannot answer.

Why this exists
---------------
On the provided 100-question set, GraphRAG scores ~0.99 and Agentic
GraphRAG ~1.00. That is a real result, and it is worth reporting honestly:
for templated single-hop questions over a well-modelled graph, agentic
reasoning is NOT worth its token cost. A one-point gap does not justify a
planner, a tool loop and a verifier.

But it also means the provided set cannot answer the hackathon's actual
question — *when* does agentic reasoning start to pay? A benchmark
saturated at the ceiling measures nothing above it. So this module
generates a second, harder set on the same corpus, designed so that the
discriminating variable is exactly the one Pipeline 2 lacks: the ability
to feed the result of one graph query into the next.

Every question here has the shape "resolve X, then query using X", where
X is not stated in the question. A fixed pipeline compiles one spec and
must guess X or fail. An agent resolves X first, then queries.

Gold answers are computed directly from the structured graph at
generation time, so they are correct by construction — there is no LLM in
the labelling path and no human judgement to disagree with. Generation is
seeded, so the set is reproducible.

Question families
-----------------
  chained_discipline  identify a discipline by a superlative, then count within it
  chained_chronology  resolve a relative Games edition, then look up a field in it
  chained_venue       resolve a Games edition from a venue, then query that edition
  cross_edition       run the same aggregation on two editions and compare
  relaxation          over-specified constraints that return zero rows until relaxed

Usage:
    python -m src.eval.stress_set --n 60
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.tigergraph_client import get_graph

OUT_PATH = os.path.join(config.DATA_DIR, "eval_stress.jsonl")


def _index(graph):
    """Group every structured event by games edition and discipline once,
    so generation is a few dict lookups rather than repeated full scans."""
    events = graph.filter_olympic_events()
    by_games = defaultdict(list)
    by_games_disc = defaultdict(list)
    for e in events:
        if not e.get("games_id"):
            continue
        by_games[e["games_id"]].append(e)
        by_games_disc[(e["games_id"], e["discipline"])].append(e)
    return events, by_games, by_games_disc


def _season_year(games_id):
    """'2012-summer' -> (2012, 'Summer'). The id format is produced by
    infobox.games_id(); parse it rather than slicing by offset."""
    year, _, season = games_id.partition("-")
    return int(year), season.capitalize()


def _unique_max(counts):
    """Returns the single argmax, or None when the top is tied — a tied
    superlative makes an ambiguous question, and an ambiguous question
    makes an ungradeable gold answer, so those are skipped."""
    if not counts:
        return None
    best = max(counts, key=lambda k: counts[k])
    if sum(1 for v in counts.values() if v == counts[best]) > 1:
        return None
    return best


# ---------------------------------------------------------------- families

def gen_chained_discipline(by_games, by_games_disc, rng, n):
    """"In the discipline with the most events at the 2012 Summer Olympics,
    how many events had more than N competitors?"

    Hop 1: count events per discipline -> argmax -> a discipline name.
    Hop 2: aggregation constrained to that discipline.
    The discipline is never named in the question, so a single aggregation
    spec has nothing to put in its `discipline` field."""
    out = []
    for games_id in rng.sample(list(by_games), min(len(by_games), n * 3)):
        counts = defaultdict(int)
        for e in by_games[games_id]:
            counts[e["discipline"]] += 1
        disc = _unique_max(counts)
        if not disc or counts[disc] < 4:
            continue
        pool = [e for e in by_games_disc[(games_id, disc)] if e["competitors"] is not None]
        if len(pool) < 4:
            continue
        threshold = sorted(e["competitors"] for e in pool)[len(pool) // 2]
        gold = sum(1 for e in pool if e["competitors"] > threshold)
        year, season = _season_year(games_id)
        out.append({
            "question": f"In the discipline that held the most events at the {year} {season} "
                        f"Olympics, how many of its events had more than {threshold} competitors?",
            "answer": [str(gold)],
            "qtype": "chained_discipline",
            "hops_required": 2,
            "intermediate": {"discipline": disc, "games_id": games_id},
            "gold_doc_ids": [e["doc_id"] for e in pool if e["competitors"] > threshold],
        })
        if len(out) >= n:
            break
    return out


def gen_chained_chronology(graph, by_games_disc, rng, n):
    """"How many nations competed in <event> at the Summer Olympics held
    immediately before the Games that used <venue>?"

    Hop 1: venue -> games edition. Hop 2: chronology -> previous edition.
    Hop 3: look up the field on that edition's event. Three resolutions,
    none of which the question states directly."""
    chrono = graph.games_chronology()
    out = []
    keys = [k for k in by_games_disc if len(by_games_disc[k]) >= 3]
    rng.shuffle(keys)

    for games_id, disc in keys:
        year, season = _season_year(games_id)
        node = chrono.get(games_id)
        if not node or not node.get("prev_id"):
            continue
        prev = node["prev_id"]
        while prev and chrono.get(prev, {}).get("season") != season:
            prev = chrono[prev].get("prev_id")
        if not prev:
            continue

        for e in by_games_disc[(games_id, disc)]:
            prev_events = [p for p in by_games_disc.get((prev, disc), [])
                           if p["event_name"] == e["event_name"] and p["nations"]]
            if len(prev_events) != 1:
                continue
            target = prev_events[0]
            venue = (e.get("venue") or "").strip()
            if not venue or len(venue) < 6:
                continue
            out.append({
                "question": f"How many nations competed in the {disc.lower()} "
                            f"{e['event_name']} event at the {season} Olympics held "
                            f"immediately before the Games where {venue} was used?",
                "answer": [str(target["nations"])],
                "qtype": "chained_chronology",
                "hops_required": 3,
                "intermediate": {"venue_games": games_id, "prev_games": prev},
                "gold_doc_ids": [target["doc_id"]],
            })
            break
        if len(out) >= n:
            break
    return out


def gen_chained_venue(by_games, rng, n):
    """"Which event held at <venue> had the highest number of competitors?"
    where the venue spans several editions — so the answer depends on
    resolving the venue's full event set first, not on any single edition."""
    by_venue = defaultdict(list)
    for events in by_games.values():
        for e in events:
            v = (e.get("venue") or "").strip()
            if v and e["competitors"] is not None:
                by_venue[v].append(e)

    out = []
    venues = [v for v, es in by_venue.items() if len(es) >= 5 and len(v) > 6]
    rng.shuffle(venues)
    for v in venues:
        es = by_venue[v]
        top = max(es, key=lambda e: e["competitors"])
        if sum(1 for e in es if e["competitors"] == top["competitors"]) > 1:
            continue
        out.append({
            "question": f"Among all events held at {v}, which one had the highest "
                        f"number of competitors?",
            "answer": [top["title"]],
            "qtype": "chained_venue",
            "hops_required": 2,
            "intermediate": {"venue": v, "n_candidates": len(es)},
            "gold_doc_ids": [top["doc_id"]],
        })
        if len(out) >= n:
            break
    return out


def gen_cross_edition(by_games_disc, rng, n):
    """"Did <discipline> have more events with over N competitors at <year A>
    or <year B>?" — two independent aggregations, then a comparison. A
    single aggregation spec holds one games_id."""
    out = []
    by_disc = defaultdict(list)
    for (games_id, disc) in by_games_disc:
        by_disc[disc].append(games_id)

    discs = [d for d, gs in by_disc.items() if len(gs) >= 2]
    rng.shuffle(discs)
    for disc in discs:
        pairs = [g for g in by_disc[disc]]
        rng.shuffle(pairs)
        a, b = pairs[0], pairs[1]
        if _season_year(a)[1] != _season_year(b)[1]:   # compare like with like
            continue
        threshold = 30
        ca = sum(1 for e in by_games_disc[(a, disc)]
                 if e["competitors"] and e["competitors"] > threshold)
        cb = sum(1 for e in by_games_disc[(b, disc)]
                 if e["competitors"] and e["competitors"] > threshold)
        if ca == cb:
            continue
        ya, sa_ = _season_year(a)
        yb, _ = _season_year(b)
        gold = str(ya if ca > cb else yb)
        out.append({
            "question": f"Did {disc.lower()} have more events with over {threshold} "
                        f"competitors at the {ya} or the {yb} {sa_} Olympics? "
                        f"Answer with the year.",
            "answer": [gold],
            "qtype": "cross_edition",
            "hops_required": 2,
            "intermediate": {"a": a, "b": b, "count_a": ca, "count_b": cb},
            "gold_doc_ids": [e["doc_id"] for e in
                             by_games_disc[(a if ca > cb else b, disc)]],
        })
        if len(out) >= n:
            break
    return out


def gen_relaxation(by_games_disc, rng, n):
    """Deliberately over-specified: names a venue that the target event's
    record does NOT list, alongside constraints that do match. A query
    carrying every stated constraint returns zero rows; the answer is only
    reachable by recognising which constraint to drop."""
    out = []
    keys = [k for k in by_games_disc if len(by_games_disc[k]) >= 6]
    rng.shuffle(keys)
    for games_id, disc in keys:
        events = [e for e in by_games_disc[(games_id, disc)] if e["competitors"]]
        if len(events) < 6:
            continue
        # A venue used by this Games but NOT by this discipline.
        other_venues = {e.get("venue") for (g, d), es in by_games_disc.items()
                        if g == games_id and d != disc for e in es if e.get("venue")}
        own_venues = {e.get("venue") for e in events if e.get("venue")}
        decoys = [v for v in other_venues - own_venues if v and len(v) > 6]
        if not decoys:
            continue
        decoy = rng.choice(sorted(decoys))
        threshold = sorted(e["competitors"] for e in events)[len(events) // 2]
        gold = sum(1 for e in events if e["competitors"] > threshold)
        year, season = _season_year(games_id)
        out.append({
            "question": f"At the {year} {season} Olympics, around {decoy}, how many "
                        f"{disc.lower()} events had more than {threshold} competitors?",
            "answer": [str(gold)],
            "qtype": "relaxation",
            "hops_required": 2,
            "intermediate": {"decoy_venue": decoy, "must_drop": "venue"},
            "gold_doc_ids": [e["doc_id"] for e in events if e["competitors"] > threshold],
        })
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- driver

def generate(n_total=60, seed=7):
    graph = get_graph(verbose=False)
    events, by_games, by_games_disc = _index(graph)
    if not events:
        print("No structured events in the graph. Run: python -m src.ingestion.build_graph")
        return []

    rng = random.Random(seed)
    per = max(1, n_total // 5)
    qs = (gen_chained_discipline(by_games, by_games_disc, rng, per)
          + gen_chained_chronology(graph, by_games_disc, rng, per)
          + gen_chained_venue(by_games, rng, per)
          + gen_cross_edition(by_games_disc, rng, per)
          + gen_relaxation(by_games_disc, rng, per))

    for i, q in enumerate(qs):
        q["qid"] = f"stress-{i+1:03d}"
        q["answer_verified"] = True   # computed from the graph, not generated
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    qs = generate(args.n, args.seed)
    if not qs:
        return
    with open(args.out, "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    counts = defaultdict(int)
    for q in qs:
        counts[q["qtype"]] += 1
    print(f"Wrote {len(qs)} stress questions -> {args.out}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
