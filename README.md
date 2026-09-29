# Agentic GraphRAG on TigerGraph

Three pipelines — **RAG**, **GraphRAG**, **Agentic GraphRAG** — over one TigerGraph
knowledge graph built from 2,951 Wikipedia Olympic documents, benchmarked
side-by-side to answer the question the hackathon actually poses:

> **When does agentic reasoning earn its token cost, and when is it overkill?**

Measured on a **live TigerGraph Savanna workspace** (4.2.5).

### What the system does on the provided questions

| | Accuracy | Avg tokens / question |
|---|---|---|
| **Agentic GraphRAG (production configuration)** | **100 / 100** | **67** |
| GraphRAG | 100 / 100 | 0 |
| RAG | 0.20 | 2,764 |

99 of 100 questions resolve deterministically against the graph — **no LLM call
at all** — because the corpus has been modelled into an `OlympicEvent` /
`GamesEdition` / `Discipline` graph rather than left as prose. The single
escalation is a genuinely ambiguous venue+date tie, which the agent detects and
investigates rather than guessing at.

That is the system as you would deploy it, and it is the number we submit the
hidden set with.

### The controlled experiment: does the *architecture* help?

The production configuration includes a regex question-parser, and that parser
encodes prior knowledge of the question templates — a prior the RAG baseline is
never given. So to answer the hackathon's research question honestly we ran a
second, **deliberately handicapped** configuration in which every pipeline,
including ours, must compile its query with the same LLM:

| Pipeline | Accuracy | Avg tokens | Tokens / correct |
|---|---|---|---|
| RAG | 0.200 | 2,764 | 13,820 |
| GraphRAG | 0.290 | 985 | 3,396 |
| **Agentic GraphRAG** | **0.420** | 3,795 | 9,057 |

Handicapped this way, the agent beats the fixed pipeline by **13 points (45%
relative)** and escalates on 71% of questions. Absolute accuracy is low for all
three because `llama3` is a poor NL→query compiler — **that is the variable
under test**, not a property of the graph. The same questions score 100/100 when
parsing is reliable.

This is the configuration to read when asking *"does adaptivity help?"*. The
table above it is the configuration to read when asking *"how well does the
system work?"*.

### Where the agent pays, and where it doesn't

| Type | RAG | GraphRAG | Agentic | Verdict |
|---|---|---|---|---|
| `aggregation` | 0.00 | 0.62 | **0.91** | worth it — **+28.6 pts for 2 extra tokens** |
| `temporal` | 0.09 | 0.27 | **0.41** | worth it — +13.6 pts for 3,577 tokens |
| `multi_hop` | 0.11 | 0.07 | **0.14** | worth it — +7.2 pts for 4,681 tokens |
| `lookup` | **0.79** | 0.37 | 0.44 | **RAG wins** — plain retrieval beats the graph here |
| `superlative` | 0.00 | 0.10 | 0.10 | not worth it — no gain beyond noise |

Three things worth reading carefully:

**`aggregation` gains 28.6 points for 2 tokens.** The planner misreads the
threshold ("more than 41" compiled as `min_competitors: 42`); the grounding
check catches it and a deterministic repair substitutes the number the question
actually states — no LLM call. That is the cheapest accuracy in the table.

**`lookup` is a loss for us.** Single-document fact retrieval is exactly what
vector search is good at, and our graph planner fumbles exact title resolution.
We report it because it bounds the claim: graph structure helps with counting,
chronology and comparison, not with "find this one fact in this one document".

**`superlative` shows the agent adding nothing.** When the planner fails in a
way the grounding check cannot detect — a plausible-looking argmax over the
wrong set — there is nothing for the agent to repair.

### The zero-token result, reported separately

With the regex fast path enabled (`--fast-path`), GraphRAG and Agentic both
score **100/100 on this set spending no LLM tokens at all**, resolving 99 of
100 questions deterministically. That is a real engineering result — once a
corpus is modelled as a graph, a large class of questions needs no LLM — but it
is *not* an architectural comparison, because the regex parser encodes prior
knowledge of the question templates that RAG is never given. See
[Making the comparison fair](#making-the-comparison-fair).

### On the 60 chained questions — the agent becomes cheaper, not just better

| Pipeline | Accuracy | Avg tokens | **Tokens / correct answer** | Answered |
|---|---|---|---|---|
| RAG | 0.083 | 2,807 | 33,680 | 28% |
| GraphRAG | 0.100 | 923 | 9,229 | 100% |
| **Agentic GraphRAG** | **0.450** | 3,128 | **6,950** | 98% |

**4.5× the accuracy at 75% of GraphRAG's cost per correct answer.** On the
provided set the agent bought accuracy *with* tokens; here it is simultaneously
the most accurate and the most efficient pipeline. The inversion is the point:
the deterministic repairs (`resolve_slot` ×24, `repair_literal` ×21,
`decompose` ×7) produce answers without LLM calls, so the worse the planner
performs, the better the agent's economics look.

| Family | RAG | GraphRAG | Agentic | |
|---|---|---|---|---|
| `chained_discipline` | 0.00 | 0.25 | **0.92** | +66.7 pts for 83 tokens |
| `relaxation` | 0.00 | 0.25 | **0.75** | +50.0 pts for 1,208 tokens |
| `cross_edition` | 0.25 | 0.00 | **0.50** | fixed pipeline cannot express it |
| `chained_chronology` | 0.17 | 0.00 | 0.08 | 3 hops — everything struggles |
| `chained_venue` | 0.00 | 0.00 | 0.00 | **nothing works** |

Two rows we report rather than bury. `chained_venue` is 0.00 across all three
pipelines: resolving a venue's full cross-edition event set and then taking an
argmax defeats every approach here. And `chained_chronology` is the one family
where RAG beats the agent — three sequential resolutions is past what an 8B
planner sustains.

RAG's `answered` rate of 28% is also worth reading: it declined to answer most
of these rather than hallucinating, which is correct behaviour and why its cost
per correct answer is so poor.

### The crossover

Adaptivity pays when a question needs **the output of one graph query as the
input to the next**, and when the planner is unreliable enough that its output
needs checking. Both conditions are common with small or cheap models — which
is precisely when you would want an agent.

Full numbers, with the provenance of every run, are in **[RESULTS.md](RESULTS.md)** —
generated from `outputs/*.json` by `python -m scripts.make_report`, so they cannot
drift from what the benchmark actually produced.

![Architecture](docs/architecture.svg)

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in the two sections below

python -m src.ingestion.build_graph     # build the graph (~80s, no API calls)
python -m src.eval.stress_set --n 60    # generate the chained-reasoning set
python -m src.eval.run_benchmark        # 100 public questions, all 3 pipelines
python -m src.eval.run_benchmark --stress
python -m http.server 8000              # open /dashboard/
```

Nothing above requires a paid API key: `ollama` runs the LLM locally and the
deterministic graph layer is free either way.

By default **every pipeline compiles its query with the LLM**, so all three
spend real tokens and the comparison is fair. `--fast-path` enables the
zero-token regex parser, reported separately — see
[Making the comparison fair](#making-the-comparison-fair).

---

## Connecting TigerGraph

**Savanna (recommended).**

1. Sign in at [tgcloud.io](https://tgcloud.io) → **Create Solution** → TigerGraph
   Savanna. Provisioning takes ~5 minutes.
2. Open the solution and copy its **Domain** (e.g. `abcd1234.i.tgcloud.io`).
3. **Admin → Management → Create Secret**, copy the secret string.
4. Fill in `.env`:

   ```ini
   TG_HOST=https://abcd1234.i.tgcloud.io    # https://, no port
   TG_GRAPH=HackathonGraph
   TG_USERNAME=tigergraph
   TG_PASSWORD=<your signup password>
   TG_SECRET=<the secret from step 3>
   ```

5. One command does the rest:

   ```bash
   python -m scripts.setup_tigergraph
   ```

   This installs the schema, compiles the GSQL queries, loads all 2,951
   documents in batched upserts, and verifies with live queries. Every step is
   idempotent, so re-running after a failure is safe. Run
   `python -m scripts.setup_tigergraph --check` first to test the connection
   alone — on failure it prints the specific cause.

**Community Edition** — same thing, different `.env`:

```ini
TG_HOST=http://localhost
TG_GS_PORT=14240
TG_RESTPP_PORT=9000
TG_USERNAME=tigergraph
TG_PASSWORD=tigergraph
```

With TigerGraph configured, everything else is unchanged: `get_graph()` returns
the live backend and all three pipelines run against it. **Fallback to the
offline store is always loud** — a benchmark that quietly stopped using
TigerGraph would be worthless, so it prints exactly why rather than pretending.

**Verified against a live Savanna workspace** (TigerGraph 4.2.5): schema and
6 GSQL queries installed, 2,187 `OlympicEvent` + 20 `GamesEdition` vertices
loaded in 16s, and **100/100 on the public set** through the deterministic graph
path. The two backends agree on 99/100 — the one difference is a genuinely tied
venue+date question where the orderings differ, which is exactly the ambiguity
the agent's tie-break exists to catch.

Cosine ranking over `Chunk.embedding` runs **client-side**, not in GSQL. That is
not a shortcut: GSQL rejects every method call on a `LIST` parameter ("the
identifier of type list parameter is invalid to call any function"), so a query
taking the query vector as `LIST<DOUBLE>` cannot index into it to compute a dot
product. TigerGraph 4.2+ native `VECTOR` attributes would push this into the
database properly, and that needs dense embeddings — the default TF-IDF vectors
are sparse, so set `EMBEDDING_PROVIDER=ollama` (or `openai`) if you want them.

---

## Choosing an LLM

Five providers, one interface, identical token accounting across all of them:

```ini
LLM_PROVIDER=ollama        # groq | openrouter | ollama | anthropic | openai
OLLAMA_MODEL=llama3
```

| Provider | Notes |
|---|---|
| `ollama` | **What the published numbers were produced with.** Local, free, no rate limit. `ollama serve && ollama pull llama3 && ollama pull nomic-embed-text`. |
| `groq` | Fast per call, but its on-demand tier could not sustain this benchmark — see below. Note Groq has **retired the hosted Llama 3.x models**; use `openai/gpt-oss-120b`. |
| `openrouter` | One key, many models. |
| `anthropic` / `openai` | Supported for completeness. |

### A warning about hosted free tiers

We tried to run the benchmark on Groq's on-demand tier and could not. Three
distinct failures, all worth knowing about before you plan a run:

1. **A single RAG request exceeded the per-request cap.** 8 chunks is ~8.5k
   tokens against an 8,000 TPM limit, returning `413`. Not retryable — the same
   request can never succeed. Fixed by packing context to a budget
   (`RAG_MAX_CONTEXT_TOKENS`) rather than a fixed `k`.
2. **Reasoning tokens are billed but not bounded by `max_tokens`.** `gpt-oss`
   spends far more output than requested, so client-side pacing that reserves
   `max_tokens` under-counts badly. `LLM_TOKENS_PER_MINUTE` now reconciles each
   reservation against actual usage after the call.
3. **A throttle the headers do not expose.** With
   `x-ratelimit-remaining-tokens: 7923` and a sub-second reset, the API still
   returned `429` with `Retry-After` of **301s, 468s, 502s**. Sustained
   benchmarking was impossible regardless of pacing.

`LLM_TOKENS_PER_MINUTE` (0 = off) paces requests *before* they breach a limit,
because these providers punish a burst far more than they reward one. Long runs
are also checkpointed per question, so a rate-limit stall or an OOM kill costs
one question rather than the run.

Token counts come from each provider's own usage object. When a provider omits
them we fall back to a tiktoken estimate and **flag it** (`estimated: true`)
rather than recording a zero — a silent zero would corrupt the efficiency
comparison that the rubric weights at 15%.

---

## What the three pipelines are

Everything is held constant between them — same graph, same model, same
embedder, same executor — so the only independent variable is **adaptivity**.
Without that control, any accuracy gap would be unattributable.

**1 · RAG** (`src/pipelines/rag_pipeline.py`)
Embed the question, vector-search chunks, stuff top-k into the prompt, generate.
Built as a *fair* opponent, not a strawman: generous `k=8`, a good prompt, the
same embedder as the others. It still loses badly on aggregation and temporal
questions — and that loss is **structural, not a prompt-quality artifact**. No
top-k window can hold the dozen-plus documents needed to count "every biathlon
event at the 2018 Winter Olympics", and the Games chronology needed for
"immediately before 2016" is stated in no single document.

**2 · GraphRAG** (`src/pipelines/graphrag_pipeline.py`)
Compile the question into **one** graph query, execute it, accept the result —
including results the executor itself flagged as ambiguous. Falls back to
entity-link + 1-hop + document retrieval when the query layer misses. The
sequence never changes shape based on what comes back.

**3 · Agentic GraphRAG** (`src/agents/orchestrator.py`)
Re-decides its next move after every step. Four capabilities Pipeline 2
structurally lacks:

- **Triage + escalation** — a free deterministic attempt runs first, is
  *verified*, and verification failure escalates.
- **Deterministic repair** — resolve an unresolved slot, decompose across
  editions, correct a misread literal, relax an over-specified constraint.
- **Decomposition** — feed one query's output into the next.
- **Evidence-driven stopping** — stop on an explicit sufficiency check, and
  record when and why.

---

## The part that matters most: grounding verification

A planner asked *"in the discipline that held the most events at the 2004 Summer
Olympics, how many of its events had more than 41 competitors?"* will cheerfully
emit:

```json
{"type": "aggregation", "discipline": "Athletics", "games_id": "2004-summer",
 "min_competitors": 42}
```

That query is **well-formed**. It returns real rows. It cites real documents. It
passes every structural validity check. And it answers a question nobody asked —
`"Athletics"` was a guess, and `42` is not the number in the question.

So the verifier doesn't ask *"is this query valid?"*. It asks **"is this query
grounded in the question?"** — two deterministic, domain-general rules:

1. **Literal grounding.** Every string or number the query filters on must
   actually appear in the question. If it doesn't, the planner supplied it
   rather than read it — which means the question's real subject is a referring
   expression that must be resolved by a prior query.
2. **Constraint coverage.** If the question names more distinct editions than
   the query consumed, one query is answering part of the question only.

Neither rule inspects question phrasing, so neither is fitted to this dataset's
templates. Both run in microseconds for zero tokens, and the diagnostic they
produce names precisely what is missing — which is what makes the repair
deterministic rather than a guess.

Measured effect on the stress set: escalation fires correctly, and chained
questions that silently returned wrong answers now resolve **correctly at ~870
tokens**, because the hard part (knowing what was missing) was solved by a rule
rather than by the model.

---

## Making the comparison fair

The most important methodological decision in this repo, and the one most
worth attacking:

A regex question-parser can resolve this dataset's templated phrasings for
**zero tokens**. With it on, GraphRAG and Agentic score 100/100 spending
nothing, while RAG spends ~3,000 tokens per question and scores far lower.
That looks like a crushing architectural win.

It isn't. The regex parser is only free because it **encodes prior knowledge
of what the questions look like** — a prior the RAG baseline is never given.
Reporting that gap as "architecture" compares a pre-tuned system against an
untuned one.

So the fast path is **off by default**. Every pipeline compiles its query with
the LLM, every pipeline spends real tokens, and the comparison measures
architecture rather than eval-fitting:

```bash
python -m src.eval.run_benchmark              # LLM planner everywhere (default)
python -m src.eval.run_benchmark --fast-path  # the zero-token cached path
```

`run_meta.regex_fast_path` records which configuration produced any given
results file, and every compiled query is labelled `_planner: "llm" | "regex"`.

**What survives the fair comparison.** Structure still wins on cost — one
~300-token planner call versus RAG's ~3,000-token context — but now it is
*earned*. The graph pipelines have to understand the question with the same
model RAG uses; they just need far less text in the prompt to act on it.

**The zero-token result is still worth reporting**, as an engineering finding
rather than an architectural one: once a corpus has been modelled into a
graph, a large class of questions can be answered with **no LLM at all**. That
belongs in its own table, clearly labelled, not blended into the comparison.

### Cost per correct answer

`avg_tokens` alone is misleading across pipelines with different accuracies —
a pipeline that is cheap because it answers nothing is not efficient, just
cheap. The summary therefore reports **`tokens_per_correct_answer`**, which is
the number that actually answers "was the extra reasoning worth it".

## The stress set## The stress set

The provided 100 questions saturate at ~99% for GraphRAG. A benchmark at its
ceiling cannot measure anything above it, so it cannot answer *when* agentic
reasoning starts to pay.

`src/eval/stress_set.py` generates 60 harder questions **on the same corpus**,
where the discriminating variable is exactly the capability Pipeline 2 lacks.
Gold answers are computed directly from the graph at generation time — correct
by construction, no LLM in the labelling path, seeded and reproducible.

| Family | Shape | Hops |
|---|---|---|
| `chained_discipline` | identify a discipline by superlative, then count within it | 2 |
| `chained_chronology` | venue → edition → previous edition → look up a field | 3 |
| `chained_venue` | resolve a venue's full event set, then take an argmax | 2 |
| `cross_edition` | the same aggregation on two editions, then compare | 2 |
| `relaxation` | over-specified constraints returning zero rows until relaxed | 2 |

Fixed GraphRAG scores **7/60 (0.117)** on this set with its LLM planner, and
**0/60** with the regex planner alone. The handful it gets are cases where the
planner's guess at the unstated value happened to be right — which is luck, not
capability. Either way the gap is one of expressiveness, not tuning: one
compiled query cannot consume the result of another.

---

## Layout

```
config.py                     all settings, env-driven
schema/schema.gsql            vertices, edges, lowercase mirror attributes
schema/queries.gsql           installed GSQL incl. chunkVectorSearch
scripts/setup_tigergraph.py   connect → schema → queries → load → verify
src/tigergraph_client.py      RealGraph + MockGraph behind one interface
src/llm_client.py             5 providers, unified token accounting
src/embeddings.py             tfidf | ollama | openai, sparse-aware cosine
src/ingestion/infobox.py      deterministic infobox parser (no LLM)
src/ingestion/build_graph.py  batched ingestion into either backend
src/agents/query_planner.py   NL → graph query, regex cache + LLM compiler
src/agents/structured_agent.py exact executor (counts, argmax, chronology)
src/agents/orchestrator.py    grounding verification, repair, adaptive loop
src/agents/specialized_agents.py entity linking, traversal, vector, docs
src/agents/harness.py         state, evidence, trace, stopping criteria
src/eval/run_benchmark.py     3-way benchmark + agentic-value analysis
src/eval/stress_set.py        generates the chained-reasoning set
dashboard/index.html          metrics dashboard, all runs, trace inspection
tests/test_core.py            29 tests
```

---

## Reproducing everything

```bash
bash scripts/run_all.sh        # or: powershell scripts/run_all.ps1
python -m pytest tests/ -q
```

Outputs land in `outputs/` as `results*.json` (per-question detail with full
agentic traces) and `summary*.json` (aggregates plus the per-question-type
verdict). The dashboard reads them directly and shows a tab per run.

---

## Known limitations

- **The public-set result is a ceiling effect, not a win for the agent.** We say
  so in the dashboard verdict column rather than burying it.
- **Deterministic repair covers three failure classes**, not all of them:
  unresolved discipline, partial edition coverage, and misread numeric literals.
  Anything else falls through to the LLM loop, which is only as good as the
  model behind it. A small local model (llama3:8b) frequently echoes the
  original question instead of decomposing it; the deterministic repairs exist
  precisely so correctness does not depend on that.
- **Entity/relation layer is thin.** `Entity`/`RELATED_TO` are populated only
  sparsely, so the generic entity-linking fallback is weaker than the structured
  path. Questions outside the Olympic-event schema rely on vector search.
- **`chunkVectorSearch` computes cosine in GSQL** rather than using a native
  vector index — a deliberate portability choice, since native `VECTOR`
  attributes exist only on recent TigerGraph versions and a query that fails to
  install on a judge's instance is worse than one that is slightly slower.
