# Architecture

![Architecture diagram](docs/architecture.svg)

This document explains *why* the system is shaped the way it is. The README
covers how to run it.

---

## 1. The experiment design

The hackathon's question is not "can we build an agent" but "**when does an
agent earn its cost**". That is a causal question, so the benchmark is built as
a controlled comparison rather than three separately-tuned systems.

Held constant across all three pipelines:

- the same graph (same vertices, same edges, same backend)
- the same LLM and the same model id
- the same embedder and the same chunk store
- the same query executor and the same exact-match scorer

The single independent variable between Pipeline 2 and Pipeline 3 is
**adaptivity**: whether the system may change what it does based on what it
found. If the pipelines also differed in retrieval quality or prompt quality,
any accuracy gap would be unattributable, and the headline finding would be
worthless.

This is also why Pipeline 1 (RAG) is built as a *fair* opponent — `k=8`, a good
prompt, the same embedder — rather than a strawman. The claim we want to support
is "RAG fails on these questions **structurally**", and that claim is only
credible if the baseline was given a real chance first.

---

## 2. Data model

The corpus is 2,951 Wikipedia documents. 2,187 of them carry an
`[Infobox Olympic event]` block with a strikingly regular field set:
competitors, nations, venue, date, gold/silver/bronze plus NOC, previous and
next edition.

That regularity is the whole opportunity. Those fields are exactly what the
evaluation questions probe, so they are extracted into a **structured graph
layer** rather than left as prose for vector search to stumble over:

```
OlympicEvent ──IN_GAMES──▶ GamesEdition ──GAMES_SEQUENCE──▶ GamesEdition
     │
     └────IN_DISCIPLINE──▶ Discipline

Document ──HAS_CHUNK──▶ Chunk ──MENTIONS──▶ Entity ──RELATED_TO──▶ Entity
```

Extraction is **pure regex, zero LLM calls** (`src/ingestion/infobox.py`).
This is not a shortcut, it is the correct engineering choice: asking an LLM to
extract structured facts from 2,951 documents (~5.5M tokens) would cost real
money, take hours, and be *less* accurate than parsing a field whose format is
already fixed. The LLM's job is understanding the question, not re-reading data
we can parse exactly.

### Why the lowercase mirror attributes

`OlympicEvent` carries `discipline_lc`, `venue_lc`, `date_lc`, `event_name_lc`
alongside the originals. GSQL's `LIKE` is case-sensitive; the Python backend
matches case-insensitively via `normalize()`. Without the mirrors, **the same
question would return different answers depending on which backend was
running** — a silent divergence that would invalidate any comparison between a
local run and a TigerGraph run. Writing the lowercase forms at load time makes
the two backends agree by construction rather than by luck.

---

## 3. Backends behind one interface

`BaseGraph` declares the contract; `RealGraph` (pyTigerGraph) and `MockGraph`
(in-memory) both implement it.

This was originally not the case, and the consequence was a genuine bug: the
pipelines reached into `graph.store["chunks"]` directly, which only `MockGraph`
has. The moment you pointed the system at a real TigerGraph instance, RAG and
similarity search silently returned **zero results** — no exception, no warning,
just quietly empty retrieval and a benchmark that looked like it ran.

`tests/test_core.py::test_backends_expose_the_same_interface` exists to stop
that class of bug returning.

Backend selection is loud by design (`get_graph`). A run that quietly fell back
off TigerGraph would produce numbers that look fine and mean nothing, so the
fallback always prints why.

### Savanna vs Community Edition

Two differences, both handled in `RealGraph.__init__`: Savanna terminates TLS on
443 and needs `tgCloud=True`; Community Edition exposes RESTPP on 9000 and GSQL
on 14240 over plain HTTP. Everything above that line is identical.

### Bulk loading

Ingestion writes ~2,200 events, ~2,950 documents and ~9,000 chunks. One vertex
per HTTP round trip would be ~14,000 requests, so all writes go through
`upsertVertices`/`upsertEdges` in blocks.

---

## 4. Query compilation

`src/agents/query_planner.py` compiles a natural-language question into a query
spec. Two tiers, and the distinction between them is reported honestly:

**Tier 1 — regex.** Zero tokens, microseconds, covers this dataset's templated
phrasings. The original version of this project was *only* this, scored 100% on
the public set, and generalised to nothing — the regexes were shaped around the
public set's own question templates. That is eval-fitting, not graph reasoning.

**Tier 2 — LLM compiler.** The general path. Takes the question plus the live
graph schema (real discipline names, real games ids) and emits the same spec
shape.

Every spec carries `_planner: "regex" | "llm" | "none"`, surfaced in the
dashboard as *planner provenance*. `--ablation` disables tier 1 entirely so the
system's true un-cached accuracy is measurable. **The ablation numbers are the
honest ones**; tier 1 is a cache, and the repo says so everywhere rather than
quietly banking its accuracy.

### Executor

`src/agents/structured_agent.py` runs a spec against the graph. Counting,
argmax and chronology walks happen in the graph layer, **exactly** — no
generation is involved in the computation itself. An LLM asked to count twelve
competitor numbers out of retrieved text gets it wrong often enough to matter;
a `COUNT` does not.

A query that matches nothing returns `answer: None`, never `0` or `""`. That
distinction is load-bearing: `None` means "keep investigating", whereas a `0`
would be read downstream as a real answer.

---

## 5. The agentic loop, and the part that actually matters

### Triage first

A free deterministic attempt runs before anything is spent. If it resolves and
**passes verification**, the agent answers immediately for zero reasoning
tokens. The `escalated` flag records which questions needed more — and that flag
is the raw material for the hackathon's headline question.

### Verification is a grounding check, not a validity check

This is the core idea of the submission.

Given *"in the discipline that held the most events at the 2004 Summer Olympics,
how many of its events had more than 41 competitors?"*, a planner emits:

```json
{"type": "aggregation", "discipline": "Athletics",
 "games_id": "2004-summer", "min_competitors": 42}
```

Now look at what a conventional verifier sees. The query is well-formed. It
executes. It returns 18 real rows. It cites 18 real documents. Its answer is a
plausible integer. Every structural check passes — and the answer is wrong,
because `"Athletics"` was **guessed** (the question never says it) and `42` is
not the number the question asked about.

Worse, the failure is invisible in the output. On a counting question, an
off-by-one threshold or a wrong discipline produces a number that looks exactly
like a right number.

So the verifier asks a different question — *is this query grounded in what was
actually asked?* — via two deterministic rules:

1. **Literal grounding.** Every string and number the query filters on must
   appear in the question. If it does not, the planner supplied it rather than
   read it, which means the question's real subject was a referring expression
   ("the discipline that...") that has to be resolved by a prior query.
2. **Constraint coverage.** If the question names more distinct editions than
   the query consumed, one query is covering only part of the question — the
   signature of a comparison needing two sub-queries.

Neither rule inspects question phrasing, so neither is fitted to this dataset.
Both cost zero tokens.

### The diagnostic is what makes repair deterministic

The grounding check does not just say "this failed" — it says *which slot* was
unresolved and *why*. That is enough to repair three whole failure classes with
no reasoning at all:

| Diagnostic | Repair | Cost |
|---|---|---|
| discipline guessed, query scoped to an edition | resolve it from the graph via a count-per-discipline argmax, substitute, re-run | 0 tokens |
| question names N editions, query used fewer | re-run the same query once per edition, then compare | 0 tokens |
| numeric threshold absent from the question | substitute the only threshold the question actually states | 0 tokens |

Repairs chain (a resolved discipline with a still-misread threshold triggers the
next repair) and every repaired result is **re-verified** like any other, so a
bad repair escalates rather than being trusted.

Anything outside these three classes falls through to the LLM plan→act loop.

This design decision is deliberate and worth stating plainly: correctness on
chained questions does **not** depend on model strength. Measured with a local
llama3:8b, the model reliably fails to decompose — it echoes the original
question back as its "sub-question". The deterministic repairs exist precisely
so that the hard part (knowing what is missing) is solved by a rule, and the
model is only asked to do what it is good at.

### The adaptive loop

For everything else: plan → act → evaluate, with eight tools (graph query,
relax, vector search, entity linking, traversal, document retrieval, multi-hop
reasoning, answer). The escalation reason is passed into the planner prompt, so
the model is told exactly what the cheap attempt got wrong instead of starting
blind.

One subtle bug worth recording, because it made the agent look competent while
doing nothing: the sufficiency check was originally keyed to `state.step_count`,
which also counts the triage query and its verification. It therefore fired
*immediately after triage*, before the agent had taken a single corrective
action — and a small model asked "is this evidence sufficient?" answers "yes,
confidence 1.0" about the very evidence that had just failed verification. The
agent stopped, having done nothing, with a clean-looking trace. Sufficiency is
now gated on `loop_actions`, which counts only work done inside the loop.

### Stopping criteria

Every stop records when and why: verified-and-resolved, repaired
deterministically, agent judged evidence sufficient, max steps, or token budget.
These appear aggregated in the dashboard.

---

## 6. Why the provided set could not answer the research question

GraphRAG scores ~0.99 on the provided 100 questions. A benchmark sitting at its
ceiling cannot measure anything above it — if the fixed pipeline is already
right 99 times out of 100, no agent can demonstrate value, and "agentic
reasoning adds one point" is a measurement artifact rather than a finding.

So `src/eval/stress_set.py` generates 60 harder questions **on the same corpus**,
where the discriminating variable is precisely the capability Pipeline 2 lacks:
feeding one query's output into the next. Every question has the shape "resolve
X, then query using X", where X is never stated.

Gold answers are computed directly from the structured graph at generation time
— correct by construction, no LLM in the labelling path, no human judgement to
disagree with, seeded and reproducible. Ambiguous cases (tied superlatives) are
skipped rather than labelled, because an ambiguous question cannot be graded.

Fixed GraphRAG scores **0/60**. That is not a tuning gap to be closed with a
better prompt; it is an expressiveness gap.

---

## 7. Token accounting

The rubric asks for context tokens, input tokens, output tokens and totals, per
operation and per answer. `UsageTracker` records every call with its tag, model,
latency and token split, so `results.json` supports "tokens per operation" and
"time per operation" directly rather than requiring a reader to infer them.

Two details that matter for honesty:

- **Context tokens are measured separately** from total input tokens, so the
  retrieved-evidence cost is distinguishable from instruction overhead.
- **A provider that omits usage gets a tiktoken estimate, flagged
  `estimated: true`** — never a silent zero. A zero would quietly understate a
  pipeline's cost and corrupt exactly the comparison the rubric weights at 15%.

---

## 8. Known limitations

- The public-set result is a ceiling effect. The dashboard says "AGENT NOT WORTH
  IT" for those question types, and that is the correct read.
- Deterministic repair covers three failure classes, not all of them. The rest
  depends on the LLM loop, which is only as good as the model behind it.
- The `Entity`/`RELATED_TO` layer is sparsely populated, so the generic
  entity-linking fallback is weaker than the structured path. Questions outside
  the Olympic-event schema rely on vector search.
- `chunkVectorSearch` computes cosine similarity in GSQL rather than using a
  native vector index. Native `VECTOR` attributes exist only on recent
  TigerGraph versions, and a query that fails to install on a judge's Community
  Edition instance is worse than one that is slightly slower everywhere. The
  client falls back to scoring in Python if the query is missing entirely.
- TF-IDF embeddings are sparse and cannot be stored in a `LIST<DOUBLE>`.
  Ingestion warns and writes chunks without embeddings rather than failing
  silently; use `EMBEDDING_PROVIDER=ollama` for vector search inside TigerGraph.
