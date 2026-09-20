# Data

The real hackathon dataset is bundled here:

| File | Contents |
|---|---|
| `corpus.jsonl` | 2,951 Wikipedia documents (~5.47M tokens), one JSON object per line: `doc_id`, `title`, `url`, `wikidata_qid`, `approx_tokens`, `text` |
| `eval_public.jsonl` | 100 questions with gold answers + `gold_doc_ids`, tagged by `qtype` (aggregation/temporal/superlative/lookup/multi_hop) |
| `eval_hidden.jsonl` | 50 questions, no gold answers — this is what gets submitted |

## What's actually in the corpus

2,187 of the 2,951 documents are Olympic-event pages (`Athletics at the
2008 Summer Olympics – Men's 110 metres hurdles`, etc.) with a
**consistent structured infobox**: `competitors`, `nations`, `venue`,
`date`, `gold`/`silver`/`bronze` + NOC, `games` edition, `prev`/`next`.
The remaining ~800 are films, officeholders, and a handful of other
Wikipedia categories — not touched by any of the 100 public eval
questions, all of which are Olympics statistics/lookup questions.

That structure is exactly what `src/ingestion/infobox.py` parses out
deterministically (regex, zero LLM calls) into the `OlympicEvent` /
`GamesEdition` / `Discipline` graph — see `ARCHITECTURE.md` for why
that's the whole strategy here.

## Regenerating from scratch

If you get a fresh copy of the dataset (e.g. from the hackathon's
Google Drive link) with different filenames, drop it in here as
`corpus.jsonl` / `eval_public.jsonl` / `eval_hidden.jsonl` (same
shapes as above) and everything downstream just works.

## Generated stress set

`eval_stress.jsonl` is **not** part of the provided dataset — it is generated
from the corpus by `python -m src.eval.stress_set --n 60`, and exists because
the provided 100 questions saturate at ~99% for GraphRAG, leaving no headroom
to measure whether agentic reasoning adds value.

Each question has the shape *"resolve X, then query using X"*, where X is never
stated. Gold answers are computed directly from the structured graph at
generation time — correct by construction, no LLM in the labelling path,
seeded and reproducible. Ambiguous cases (tied superlatives) are skipped rather
than labelled, since an ambiguous question cannot be graded.

Extra fields beyond the provided sets' schema:

| Field | Meaning |
|---|---|
| `hops_required` | how many sequential resolutions the question needs (2 or 3) |
| `intermediate` | the value(s) a system must resolve on the way — for debugging, never shown to any pipeline |
| `answer_verified` | always true; answers come from the graph, not a model |

Fixed GraphRAG scores 0/60 on this set. See `ARCHITECTURE.md` §6.
