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
