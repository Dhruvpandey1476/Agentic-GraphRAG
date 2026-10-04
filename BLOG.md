# The wrong answer that passes every check

*Building an agentic GraphRAG system on TigerGraph, and what I found when I
stopped trusting it.*

---

Here is a question from the Olympic benchmark I built this system against:

> Which athletics event at the 2008 Summer Olympics had the highest number of
> competitors?

And here is the query my pipeline compiled for it:

```json
{"type": "aggregation", "discipline": "Athletics", "games_id": "2008-summer"}
```

That query is well-formed. It runs. It filters on a real discipline and a real
Games edition, both of which the question actually names. It returns 43 rows
from TigerGraph, every one a genuine document with a citable ID.

The answer it produced was **43**.

The correct answer is *Athletics at the 2008 Summer Olympics – Men's marathon*.

The query counted the athletics events instead of ranking them. Every
structural check passes — valid syntax, real filters, non-empty result, real
citations. If you only saw the answer and the citations, you could not tell
anything was wrong. **43** is a perfectly plausible number.

This is the failure mode that makes GraphRAG dangerous in a way plain RAG is
not. When RAG fails, it usually fails visibly: it says "I don't know", or it
quotes a passage that obviously doesn't answer you. When a graph query is
*subtly wrong*, it fails invisibly, and it hands you a number with footnotes.

The whole project became an answer to one question: **how do you check that a
compiled query answers the question that was asked?**

---

## The setup

Three pipelines over the same corpus (~3,000 Olympic event pages) and the same
TigerGraph graph, so the only variable is strategy:

1. **RAG** — embed the question, vector-search the chunks, stuff top-k into the
   prompt, generate.
2. **GraphRAG** — compile the question into a structured query, run it against
   TigerGraph, answer from the rows.
3. **Agentic GraphRAG** — compile, then **verify the compiled query against the
   question**, repair it when verification fails, and escalate to a reasoning
   loop only when deterministic repair can't help.

The graph schema is small and deliberately so:

```
VERTEX OlympicEvent(doc_id, title, discipline, event_name, games_id, venue,
                    date, competitors, nations, teams, win_value,
                    gold_name, gold_noc, silver_name, ...)
VERTEX GamesEdition(games_id, year, season, edition_index, prev_id, next_id)
VERTEX Chunk(chunk_id, doc_id, text, embedding)

EDGE OlympicEvent -IN_GAMES->      GamesEdition
EDGE OlympicEvent -IN_DISCIPLINE-> Discipline
EDGE GamesEdition -GAMES_SEQUENCE-> GamesEdition
```

That last edge is the one that earns its keep. "The Summer Olympics held
immediately before 2016" is not stated in any single document — it's a property
of the *chronology*, and walking `GAMES_SEQUENCE` resolves it exactly, skipping
Winter editions rather than counting them. No amount of top-k retrieval gets
you there.

To keep the comparison fair, all three pipelines use the **same model** for
everything (temperature 0). An early version let the graph pipelines use a
deterministic regex compiler while RAG burned tokens on an LLM — which made the
agentic pipeline look brilliant and meant nothing. If one arm gets a free
oracle, you aren't measuring strategy, you're measuring who got the oracle.

---

## Verification is the product

The core insight is that **structural validity and semantic grounding are
different properties**, and only the first one is easy.

A compiled query can be perfectly valid and still answer a different question.
So before trusting a query, check it against the question that produced it.
Five rules, all deterministic, all costing zero tokens:

### 1. Literal grounding

Every string the query filters on must appear in the question.

```
"In the discipline that held the most events at the 2004 Summer Olympics,
 how many of its events had more than 41 competitors?"

→ aggregation(discipline="Athletics", games_id="2004-summer")
```

"Athletics" is nowhere in that question. The compiler *guessed* it. The
question's real subject is a referring expression that has to be resolved
first — and because the diagnostic says exactly which slot was invented, the
system can resolve it from the graph (argmax over event-counts per discipline)
instead of reasoning about it.

### 2. Numeric grounding

"More than 41 competitors" compiles to `min_competitors: 42` about a third of
the time. GSQL's filter is `competitors > minCompetitors`, so 42 is off by one
and silently returns a smaller count. Not an error. Just **18** where the answer
is **20** — and nobody eyeballing the output would ever catch it.

### 3. Constraint coverage

If the question names more years than the query consumed, the query answers
part of the question. That's the signature of a comparison that needs two
sub-queries, and the repair is pure decomposition: run the same query once per
year and compare.

Dropped qualifiers live here too. `"the men's 20 kilometres walk"` compiling to
`event_name_substr: "20 kilometres walk"` passes literal grounding — every word
*is* in the question — but matches the women's event and returns the wrong
medallist with full confidence.

### 4. Shape validity

The executor reads only the keys its branch knows and ignores the rest. So this:

```json
{"type": "aggregation",
 "title": "How many shooting events at the 2004 Summer Olympics...",
 "field": "competitors", "games_id": "2004-summer"}
```

...runs happily. `title` and `field` are dropped on the floor, the discipline
constraint was never applied, and it counts **all 79 events** at that edition
instead of the 8 shooting ones. The subject of the question ended up in a key
the query ignores.

### 5. Answer-shape grounding

The rule the marathon example needed. "Which event…" wants a name; an
aggregation returns a count. Comparing the question's interrogative form
against what the spec type returns catches it instantly.

One wrinkle worth recording: the obvious implementation classifies "had the
highest **number of** competitors" as a counting question, because it contains
"number of". That phrase names the *metric being ranked on*, not the ask. Strip
superlative-metric phrases before classifying, or the rule never fires on the
exact case it exists for.

---

## Repair, not just rejection

Rejecting a bad query is only half of it. Because each diagnostic names
*precisely what is wrong*, most failures can be fixed with a rule rather than a
reasoning loop:

| Diagnostic | Deterministic repair |
|---|---|
| guessed discipline | resolve it from the graph (argmax over the edition) |
| threshold not in the question | substitute the only threshold stated |
| query covers 1 of 2 years | decompose, run per year, compare |
| dropped men's/women's | restore the qualifier |
| count where a name was asked | re-issue the same scope as an argmax |
| count where an attribute was asked | re-issue as a single-event attribute read |
| edition named by description | walk `GAMES_SEQUENCE` to resolve it, then query |
| event named by venue + date | re-issue as a venue+date lookup |

All of these read only the question and the graph. None reads a gold answer.

The last two are worth dwelling on, because they're the same insight twice.
*"Who won gold in the event held at Richmond Olympic Oval on 14 February 2010?"*
names its event by **where and when**, not by title. No event page is *called*
that, so a title-based query matches nothing — and the agent would burn its
whole step budget re-phrasing before answering from whatever text was lying
around. It answered `Shani Davis`. The right answer is `Martina Sáblíková`.

A `venue_date` query takes exactly those two filters. The repair lifts both
straight out of the question. That single rule moved multi-hop questions from
**1/9 to 27/28**.

---

## Results

100 provided questions, all three pipelines on the same model at temperature 0,
live TigerGraph:

| Pipeline | Accuracy | Tokens per correct answer | Answered |
|---|---|---|---|
| RAG | 0.200 | 13,817 | 70% |
| GraphRAG | 0.370 | 3,175 | 97% |
| **Agentic GraphRAG** | **0.960** | **1,497** | **100%** |

| Question type | n | RAG | GraphRAG | Agentic |
|---|---|---|---|---|
| aggregation | 21 | 0.00 | 0.62 | **1.00** |
| lookup | 19 | 0.79 | 0.26 | **0.95** |
| multi_hop | 28 | 0.11 | 0.11 | **0.96** |
| superlative | 10 | 0.00 | 1.00 | **1.00** |
| temporal | 22 | 0.09 | 0.27 | **0.91** |

The number I did not expect: **the agentic pipeline uses fewer tokens than
plain RAG** — 143k vs 276k total — while scoring nearly five times higher.

That inverts the usual assumption that agentic means expensive. It happens
because verification and repair are *deterministic*. 16 of 100 questions
resolved with no reasoning tokens at all. RAG, meanwhile, pays ~2,800 tokens of
stuffed context on **every** question including the 30 it then declines to
answer. Cost per *correct answer* is the honest metric, and it's 9× better.

---

## What doesn't work

Two families on the adversarial chained set score **0.00**, and both are
schema gaps rather than reasoning failures:

- **`chained_venue`** — *"Among all events held at X, which had the most
  competitors?"* is a superlative scoped by **venue**, and the `superlative`
  spec takes a discipline and an edition. The question is not expressible, so
  the agent answers from the wrong scope.
- **`chained_chronology`** — *"the Games where London was the host city"*
  requires host-city → edition resolution, and nothing in the schema maps a
  city to a Games.

Both are fixable by widening the query language. Neither is hidden in the
report.

---

## The part I'd tell you first

Most of this system's accuracy came from **reading its failures**, not from
designing it well up front. The verification rules exist because I watched
well-formed queries return confident nonsense, one category at a time.

Which means an honest caveat belongs on that 0.960: it is measured on the set
whose failures I diagnosed. The repairs are general rules that read only the
question and the graph — never gold answers — and they transfer to the held-out
hidden set, where re-running them changed 30 of 50 answers, including a gold
medallist who never won that event. But the provided 100 is a development
benchmark, and calling it anything else would be overselling it.

The other thing I'd say: **three of the worst bugs I found were in the
reporting, not the system.** The report generator had its run names hardcoded
to an older sweep and silently skipped missing files, so it kept rendering a
chained section from a two-week-old run, a provided section measured with a
different embedder, and a hidden section produced by a *different model* — all
presented side by side as one result. The submission exporter only ever *read*
a results file, so the artifact I nearly shipped had been generated before any
of the fixes. And errored rows were filtered out before aggregation, which
inflated a score from 0.483 to 0.500 by shrinking the denominator.

A system that tells you confidently wrong things is the problem this project
set out to solve. It turns out your benchmark harness can do it too, and it's
much easier to miss there, because nobody is grading your grader.

Both now fail loudly: a missing run is an error, sections that disagree about
model or embedder are an error, and errored questions count against the
pipeline that errored.

---

**Code, full results and reproduction steps:**
<https://github.com/Dhruvpandey1476/Agentic-GraphRAG>

Built on TigerGraph Savanna with pyTigerGraph. Every number regenerates from
`outputs/` with `python -m scripts.make_report`.
