"""
Deterministic parser for the corpus's Wikipedia infobox blocks. No LLM
calls involved — this is pure regex/string parsing, which is both
free and exact, unlike asking an LLM to extract structured facts from
2,951 documents (~5.5M tokens) at real API cost.

The corpus is dominated by "Olympic event" infoboxes with a very
consistent field set (competitors, nations, venue, date, gold/silver/
bronze + NOC, prev/next edition year). That structure is exactly what
the eval question types (aggregation, superlative, temporal, lookup,
multi_hop-by-venue-date) are built to probe, so we extract it into a
proper graph instead of leaving it as prose for vector search to
stumble over.
"""
import re

INFOBOX_TAG_RE = re.compile(r"\[Infobox ([^\]]+)\]")
FIELD_RE = re.compile(r"^(\w+):\s*(.*)$")
TITLE_RE = re.compile(r"^(.+?) at the (\d{4}) (Summer|Winter) Olympics(?: – (.+))?$")

SEASON_RANK = {"Winter": 0, "Summer": 1}  # within a shared year, Winter precedes Summer


def parse_infobox(text: str, prefer_type: str = None):
    """Returns (infobox_type, {field: value}) or (None, {}) if no infobox.
    Some pages (tennis in particular) nest a second, more detailed infobox
    later in the text — e.g. a "tennis tournament event" summary box
    immediately followed by the real "[Infobox Olympic event]" block with
    competitors/nations/medalists, with no blank line separating them.
    This collects field lines from the first infobox tag through to the
    first blank line AFTER the LAST infobox tag, so every field from every
    stacked infobox is captured regardless of spacing between them."""
    tags = list(INFOBOX_TAG_RE.finditer(text))
    if not tags:
        return None, {}
    start = tags[0].start()
    after_last_tag = tags[-1].end()
    blank = re.search(r"\n\s*\n", text[after_last_tag:])
    end = after_last_tag + blank.start() if blank else len(text)
    body = text[start:end]

    fields = {}
    for line in body.split("\n"):
        line = line.strip()
        fm = FIELD_RE.match(line)
        if fm:
            fields[fm.group(1)] = fm.group(2)

    types = [m.group(1) for m in tags]
    itype = prefer_type if prefer_type in types else types[0]
    return itype, fields


def parse_title(title: str):
    """'Athletics at the 2008 Summer Olympics – Men's decathlon' ->
    {'discipline': 'Athletics', 'year': 2008, 'season': 'Summer', 'event_name': "Men's decathlon"}"""
    m = TITLE_RE.match(title)
    if not m:
        return None
    discipline, year, season, event_name = m.groups()
    return {
        "discipline": discipline.strip(),
        "year": int(year),
        "season": season,
        "event_name": (event_name or "").strip(),
    }


def games_id(year: int, season: str) -> str:
    return f"{year}-{season.lower()}"


def normalize(s: str) -> str:
    """Loose match key: lowercase, strip punctuation/whitespace/dashes."""
    if s is None:
        return ""
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def contains_phrase(haystack: str, needle: str) -> bool:
    """Word-boundary substring check. Plain normalize()-based 'in' checks
    are wrong for phrases like "men's sprint" vs "women's sprint" — after
    stripping punctuation, "womenssprint" contains "menssprint" as a raw
    substring even though they're different events. '+'/'-' are kept as
    literal characters (not stripped to space) because Olympic weight
    classes depend on them: "+80 kg" and "80 kg" are different events, and
    stripping the '+' would make "+80 kg" falsely match a search for
    "80 kg". A custom boundary (excluding letters/digits/+/- on the left)
    is used instead of plain \\b so "+80" can't match a search for "80".
    """
    if not needle:
        return True
    h = re.sub(r"[^a-z0-9+\-\s]", " ", haystack.lower())
    n = re.sub(r"[^a-z0-9+\-\s]", " ", needle.lower())
    h = re.sub(r"\s+", " ", h).strip()
    n = re.sub(r"\s+", " ", n).strip()
    if not n:
        return True
    pattern = r"(?<![a-z0-9+\-])" + re.escape(n) + r"(?![a-z0-9])"
    return re.search(pattern, h) is not None


def to_int(s):
    try:
        return int(re.sub(r"[^\d]", "", s))
    except (ValueError, TypeError):
        return None


def parse_olympic_event(doc: dict):
    """doc: {'doc_id', 'title', 'url', 'text', ...} from corpus.jsonl.
    Returns a structured dict, or None if this doc isn't an Olympic event."""
    itype, fields = parse_infobox(doc["text"], prefer_type="Olympic event")
    if itype != "Olympic event":
        return None
    parsed_title = parse_title(doc["title"])
    if not parsed_title:
        return None

    medalists = []
    for color in ("gold", "silver", "bronze"):
        for suffix in ("", "2", "3"):
            name_key, noc_key = f"{color}{suffix}", f"{color}NOC{suffix}"
            if fields.get(name_key):
                medalists.append({
                    "medal": color, "name": fields[name_key],
                    "noc": fields.get(noc_key, ""),
                })

    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "url": doc.get("url", ""),
        "discipline": parsed_title["discipline"],
        "year": parsed_title["year"],
        "season": parsed_title["season"],
        "games_id": games_id(parsed_title["year"], parsed_title["season"]),
        "event_name": parsed_title["event_name"] or fields.get("event", ""),
        "venue": fields.get("venue", ""),
        "venues": fields.get("venues", ""),
        "date": fields.get("date") or fields.get("dates", ""),
        "competitors": to_int(fields.get("competitors")),
        "nations": to_int(fields.get("nations")),
        "teams": to_int(fields.get("teams")),
        "medalists": medalists,
        "win_value": fields.get("win_value", ""),
        "prev_year": to_int(fields.get("prev")),
        "next_year": to_int(fields.get("next")),
    }


def build_games_chronology(events: list):
    """Global ordering of Games editions (year, season) seen in the corpus,
    used to answer 'the Olympics immediately before/after <year>' questions.
    Returns {games_id: {'year', 'season', 'index', 'prev_id', 'next_id'}}."""
    editions = sorted(
        {(e["year"], e["season"]) for e in events},
        key=lambda ys: (ys[0], SEASON_RANK[ys[1]]),
    )
    chrono = {}
    for i, (year, season) in enumerate(editions):
        gid = games_id(year, season)
        chrono[gid] = {
            "year": year, "season": season, "index": i,
            "prev_id": games_id(*editions[i - 1]) if i > 0 else None,
            "next_id": games_id(*editions[i + 1]) if i < len(editions) - 1 else None,
        }
    return chrono
