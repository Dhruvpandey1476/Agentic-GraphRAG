"""
Structured query executor: runs a compiled query spec (see
query_planner.py) against the graph and returns a computed answer.

The computation itself never involves generation — counting, argmax and
chronology walks are done in the graph layer, exactly, and the LLM is
only ever used upstream to decide WHAT to ask. That split is the point:
an LLM asked to count twelve competitor numbers from retrieved text gets
it wrong often enough to matter; a graph query does not get it wrong.

Every result carries `evidence` (the actual event records the answer was
computed from) so the final answer is citable down to the source
document, which is what "Evidence quality & explainability" is scored on.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ingestion.infobox import (normalize, contains_phrase, search_key,
                                   games_id as make_games_id)
from src.agents.query_planner import plan  # re-exported for callers

DEPRIORITIZE_ROUND = ("heat", "semi", "qualif", "quarter", "prelim", "round 1")


def _missing(spec, *fields):
    """Names of required fields the compiler failed to supply.

    A compiled spec is not trustworthy input: the model omits fields, and
    _coerce_numerics deliberately DROPS values it cannot parse rather than
    passing a string into an arithmetic comparison. Either way the field is
    absent, and using it raises deep inside the executor — a temporal query
    missing target_year reached `year < None` and killed the question.
    Validating at the top of each branch turns that into an honest empty
    result, which the grounding check can then act on.
    """
    return [f for f in fields if spec.get(f) in (None, "")]


def _empty(note=""):
    return {"answer": None, "evidence": [], "matched_doc_ids": [], "note": note}


def _ok(answer, evidence, doc_ids=None, **extra):
    return {"answer": answer, "evidence": evidence,
            "matched_doc_ids": doc_ids if doc_ids is not None else [e["doc_id"] for e in evidence],
            **extra}


# ---------------------------------------------------------------- helpers

def _date_match_priority(date_field: str, date_substr: str):
    """Many events list several round dates in one field, e.g.
    "15 August 2008 (heats)16 August 2008 (semi-finals)18 August 2008 (final)".
    A venue+date question almost always means the day medals were awarded,
    not a heat that happened to share a venue. Lower is better.

    Tier 0: the whole date field equals the query date exactly — a much
            stronger signal than substring containment.
    Tier 1: matched inside a "(final)" round label.
    Tier 2: unlabeled single date field, substring match.
    Tier 3: matched inside some other round label.
    Tier 4: matched inside a heats/semis/qualifying round label.
    None if date_substr doesn't match anywhere."""
    date_field = date_field or ""
    if normalize(date_field) == normalize(date_substr):
        return 0
    if "(" not in date_field:
        return 2 if contains_phrase(date_field, date_substr) else None
    best = None
    for text, label in re.findall(r"([^()]+)\(([^)]*)\)", date_field):
        if not contains_phrase(text, date_substr):
            continue
        label_l = label.lower()
        if "final" in label_l and not any(k in label_l for k in ("semi", "quarter")):
            p = 1
        elif any(k in label_l for k in DEPRIORITIZE_ROUND):
            p = 4
        else:
            p = 3
        if best is None or p < best:
            best = p
    return best


def _discipline_venue_affinity(discipline: str, venue: str) -> int:
    """Tiebreak for same-venue/date/priority ties: multi-sport venues are
    often named after one discipline they host (e.g. "Laura Biathlon & Ski
    Complex" hosts biathlon and cross-country skiing)."""
    return 1 if contains_phrase(venue, discipline) else 0


_YEAR_RE = re.compile(r"\b(18[5-9]\d|19\d\d|20\d\d)\b")

# Event page titles join discipline and event with an en dash, not a hyphen.
TITLE_SEP = " – "


def _edition_hint(title: str):
    """The (year, season) an event page title names, when it names one.

    Event titles are uniformly "<Discipline> at the <year> <Summer|Winter>
    Olympics – <Event name>", so both are recoverable by pattern and the
    compiler does not have to have supplied them as separate fields.
    """
    m = _YEAR_RE.search(title or "")
    low = (title or "").lower()
    return (int(m.group(1)) if m else None,
            "summer" if "summer" in low else "winter" if "winter" in low else None)


def _edition_compatible(event, year, season) -> bool:
    """False only when the event CONTRADICTS a hint the title actually gave.
    A title naming no year constrains nothing, so absent hints never filter."""
    if year is None and season is None:
        return True
    e_year, _, e_season = (event.get("games_id") or "").lower().partition("-")
    if year is not None and e_year != str(year):
        return False
    if season is not None and e_season and e_season != season:
        return False
    return True


def _title_overlap(requested: str, actual: str) -> int:
    """How many words of the actual event title the request also mentions.

    Only used to break a tie between events sharing an event_name. Extra
    prose the compiler wrapped around the real title can add unmatched
    words but can never steal the comparison, because this counts words of
    `actual` that `requested` covers, not the reverse.
    """
    want = set(search_key(requested).split())
    return sum(1 for w in set(search_key(actual).split()) if w in want)


def _find_by_title(graph, title: str):
    """Resolve an event page title to one event record. Filters broadly by
    the tail-end event-name phrase first so we don't scan the whole store.

    Every match tier is constrained to the Games edition the title names.
    The loose tier used to match the event-name tail ALONE, which silently
    discards the year: an LLM-compiled title carrying any extra prose
    ("Nations that competed in Sailing at the 2016 Summer Olympics –
    Women's RS:X") fails the exact and partial tiers, and the loose tier
    then returned whichever same-named event the backend happened to list
    first — the 2012 RS:X — reported upstream as "resolved exactly from 1
    graph record". A confidently wrong answer is worse than none, so when
    the loose tier cannot single an event out we return None and let the
    caller escalate rather than guess an edition.
    """
    if not title:
        return None
    year, season = _edition_hint(title)
    tail = title.split(TITLE_SEP)[-1] if TITLE_SEP in title else None
    candidates = graph.filter_olympic_events(event_name_substr=tail) if tail \
        else graph.filter_olympic_events()
    candidates = [e for e in candidates if _edition_compatible(e, year, season)]

    exact = [e for e in candidates if normalize(title) == normalize(e["title"])]
    if exact:
        return exact[0]

    # Word-boundary matching, not a raw normalize() substring test: once
    # punctuation is stripped "womens20kilometreswalk" literally contains
    # "mens20kilometreswalk", so a query for the men's 20 km walk matched
    # the women's event and returned its medallist as an exact answer.
    # contains_phrase pads both sides and is the same comparison the loader
    # and GSQL use, so the backends cannot disagree about what matched.
    tiers = [[e for e in candidates if contains_phrase(e["title"], title)]]
    if tail:
        tiers.append([e for e in candidates if contains_phrase(e["event_name"], tail)])
    for pool in tiers:
        if not pool:
            continue
        if len(pool) == 1:
            return pool[0]
        # Several editions run the same event. Accept one only if the rest
        # of the requested title singles it out; otherwise the edition was
        # never specified and guessing it is how the wrong Games' medallist
        # gets reported as an exact answer.
        ranked = sorted(pool, key=lambda e: -_title_overlap(title, e["title"]))
        if _title_overlap(title, ranked[0]["title"]) > _title_overlap(title, ranked[1]["title"]):
            return ranked[0]
        return None
    return None


def _medal_of(event, medal="gold"):
    return next((m["name"] for m in event["medalists"] if m["medal"] == medal), None)


def _resolve_relative_games(chrono, season, target_year, direction="before"):
    """Walk the GAMES_SEQUENCE chain to the nearest edition of the same
    season before/after target_year. Editions of the other season are
    skipped rather than counted, because 'the Summer Olympics held
    immediately before 2016' means 2012, not the 2014 Winter Games.

    Returns None when either input is missing. Callers validate first, but
    this is the function that actually does the arithmetic (`y < target_year`),
    so it guards itself rather than trusting every call site.
    """
    if season is None or target_year is None:
        return None
    target_gid = make_games_id(target_year, season)
    node = chrono.get(target_gid)
    key = "prev_id" if direction == "before" else "next_id"
    if not node:
        # Target edition isn't in the corpus; fall back to ordering by year.
        same = sorted(((g["year"], gid) for gid, g in chrono.items()
                       if g["season"] == season), key=lambda t: t[0])
        if direction == "before":
            cands = [gid for y, gid in same if y < target_year]
            return cands[-1] if cands else None
        cands = [gid for y, gid in same if y > target_year]
        return cands[0] if cands else None

    gid = node.get(key)
    while gid and chrono.get(gid, {}).get("season") != season:
        gid = chrono[gid].get(key)
    return gid


# ---------------------------------------------------------------- execution

def execute(spec: dict, graph) -> dict:
    """Runs a compiled spec. Returns
    {'answer', 'evidence', 'matched_doc_ids', ...}. `answer` is None when
    the graph genuinely holds no matching records — callers must treat
    that as "keep investigating", not as a zero."""
    t = spec.get("type")

    # ---- single-event attribute lookup ----
    if t in ("lookup_field", "lookup_nations"):
        field = spec.get("field", "nations")
        e = _find_by_title(graph, spec.get("title", ""))
        if not e:
            return _empty("no event matched that title")
        val = e.get(field)
        return _ok(str(val) if val is not None else None, [e])

    if t == "medalist":
        e = _find_by_title(graph, spec.get("title", ""))
        if not e:
            return _empty("no event matched that title")
        return _ok(_medal_of(e, spec.get("medal", "gold")), [e])

    # ---- counting ----
    if t == "aggregation":
        if _missing(spec, "games_id") and _missing(spec, "discipline"):
            return _empty("aggregation query has no games_id and no discipline "
                          "— it would count the entire corpus")
        matches = graph.filter_olympic_events(
            discipline=spec.get("discipline"), games_id=spec.get("games_id"),
            min_competitors=spec.get("min_competitors"),
            venue_substr=spec.get("venue_substr"),
        )
        if spec.get("max_competitors") is not None:
            cap = spec["max_competitors"]
            matches = [e for e in matches
                       if e["competitors"] is not None and e["competitors"] < cap]
        return _ok(str(len(matches)), matches)

    # ---- argmax / argmin ----
    if t == "superlative":
        metric = spec.get("metric", "competitors")
        matches = graph.filter_olympic_events(
            discipline=spec.get("discipline"), games_id=spec.get("games_id"))
        matches = [e for e in matches if e.get(metric) is not None]
        if not matches:
            return _empty("no events with that metric")
        pick = min if spec.get("direction") == "min" else max
        best = pick(matches, key=lambda e: e[metric])
        tied = [e for e in matches if e[metric] == best[metric]]
        return _ok(best["title"], matches, [e["doc_id"] for e in matches],
                   ambiguous=len(tied) > 1,
                   tied_candidates=[e["doc_id"] for e in tied] if len(tied) > 1 else [])

    # ---- relative-chronology resolution, then a query on that edition ----
    if t == "temporal":
        gaps = _missing(spec, "season", "target_year")
        if gaps:
            return _empty(f"temporal query is missing {', '.join(gaps)}")
        chrono = graph.games_chronology()
        prev_gid = _resolve_relative_games(
            chrono, spec.get("season"), spec.get("target_year"),
            spec.get("direction", "before"))
        if not prev_gid:
            return _empty("could not resolve the relative Games edition")
        matches = graph.filter_olympic_events(
            discipline=spec.get("discipline"), games_id=prev_gid,
            event_name_substr=spec.get("event_name_substr"))
        if not matches:
            return _empty(f"resolved edition {prev_gid} but no event matched")
        e = matches[0]
        field = spec.get("field", "gold")
        val = _medal_of(e, field) if field in ("gold", "silver", "bronze") else e.get(field)
        return _ok(str(val) if val is not None else None, [e],
                   resolved_games_id=prev_gid)

    # ---- identify an event by venue + date ----
    if t == "venue_date":
        gaps = _missing(spec, "venue", "date")
        if gaps:
            return _empty(f"venue_date query is missing {', '.join(gaps)}")
        matches = graph.filter_olympic_events(
            venue_substr=spec.get("venue"), date_substr=spec.get("date"),
            games_id=spec.get("games_id"))
        if not matches:
            return _empty("no event at that venue on that date")

        def _rank(e):
            p = _date_match_priority(e["date"], spec.get("date", ""))
            return 9 if p is None else p   # `p or 9` would wrongly demote priority 0

        best_priority = min(_rank(e) for e in matches)
        tied = [e for e in matches if _rank(e) == best_priority]

        if len(tied) > 1:
            affinity = [(_discipline_venue_affinity(e["discipline"], spec.get("venue", "")), e)
                        for e in tied]
            max_aff = max(a for a, _ in affinity)
            if max_aff > 0:
                tied = [e for a, e in affinity if a == max_aff]

        e = tied[0]
        field = spec.get("field", "gold")
        val = _medal_of(e, field) if field in ("gold", "silver", "bronze") else e.get(field)
        return _ok(val, tied, [e["doc_id"]],
                   ambiguous=len(tied) > 1,
                   tied_candidates=[x["doc_id"] for x in tied] if len(tied) > 1 else [])

    # ---- medals by nation ----
    if t == "nation_medals":
        noc = (spec.get("noc") or "").upper()
        medal = spec.get("medal", "any")
        colors = ("gold", "silver", "bronze") if medal == "any" else (medal,)
        matches = graph.filter_olympic_events(
            discipline=spec.get("discipline"), games_id=spec.get("games_id"))
        hits = [e for e in matches
                if any(m["medal"] in colors and (m.get("noc") or "").upper() == noc
                       for m in e["medalists"])]
        return _ok(str(len(hits)), hits)

    # ---- primitives that exist so the AGENT can compose them ----
    # Each of these resolves one hop. A fixed pipeline compiles exactly one
    # spec per question, so it can run any single one of these but cannot
    # feed the output of one into the input of another. That composition is
    # the capability the stress set (src/eval/stress_set.py) probes.

    if t == "discipline_superlative":
        # "which discipline had the most events at <games>"
        if _missing(spec, "games_id"):
            return _empty("discipline_superlative query is missing games_id")
        matches = graph.filter_olympic_events(games_id=spec.get("games_id"))
        if not matches:
            return _empty("no events at that Games edition")
        counts = {}
        for e in matches:
            if spec.get("min_competitors") is not None:
                if e["competitors"] is None or e["competitors"] <= spec["min_competitors"]:
                    continue
            counts[e["discipline"]] = counts.get(e["discipline"], 0) + 1
        if not counts:
            return _empty("no events matched the constraint")
        pick = min if spec.get("direction") == "min" else max
        best = pick(counts, key=lambda d: counts[d])
        tied = [d for d, c in counts.items() if c == counts[best]]
        return _ok(best, [e for e in matches if e["discipline"] == best],
                   ambiguous=len(tied) > 1, counts=counts)

    if t == "chronology_lookup":
        # "which Games edition came immediately before/after <games>"
        gaps = _missing(spec, "season", "target_year")
        if gaps:
            return _empty(f"chronology_lookup is missing {', '.join(gaps)}")
        chrono = graph.games_chronology()
        gid = _resolve_relative_games(chrono, spec.get("season"), spec.get("target_year"),
                                      spec.get("direction", "before"))
        if not gid:
            return _empty("no adjacent edition of that season in the corpus")
        node = chrono[gid]
        return _ok(gid, [], [], resolved={"games_id": gid, "year": node["year"],
                                          "season": node["season"]})

    if t == "venue_games":
        # "at which Games edition was <venue> used" -> a games_id
        if _missing(spec, "venue"):
            return _empty("venue_games query is missing venue")
        matches = graph.filter_olympic_events(venue_substr=spec.get("venue"),
                                              discipline=spec.get("discipline"))
        if not matches:
            return _empty("no events at that venue")
        gids = {}
        for e in matches:
            gids[e["games_id"]] = gids.get(e["games_id"], 0) + 1
        best = max(gids, key=lambda g: gids[g])
        return _ok(best, [e for e in matches if e["games_id"] == best],
                   ambiguous=len(gids) > 1, editions=gids)

    # ---- run one sub-query against two editions and compare ----
    if t == "comparison":
        sub = dict(spec.get("sub") or {})
        if not sub:
            return _empty("comparison spec had no sub-query")
        a_spec = {**sub, "games_id": spec.get("games_id_a")}
        b_spec = {**sub, "games_id": spec.get("games_id_b")}
        a, b = execute(a_spec, graph), execute(b_spec, graph)
        if a["answer"] is None or b["answer"] is None:
            return _empty("one side of the comparison had no data")
        try:
            av, bv = float(a["answer"]), float(b["answer"])
            winner = spec.get("games_id_a") if av > bv else spec.get("games_id_b")
            answer = str(int(abs(av - bv))) if spec.get("want") == "difference" else winner
        except ValueError:
            answer = f"{a['answer']} vs {b['answer']}"
        return _ok(answer, a["evidence"] + b["evidence"],
                   sides={"a": a["answer"], "b": b["answer"]})

    return _empty("unstructured or unknown query type")


# Backwards-compatible alias: older callers imported parse_question from here.
def parse_question(question: str, known_disciplines: list, llm=None):
    spec, _ = plan(question, known_disciplines, llm=llm)
    return spec
