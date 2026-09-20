# Agentic GraphRAG on TigerGraph

Three pipelines — **RAG**, **GraphRAG**, **Agentic GraphRAG** — over one TigerGraph
knowledge graph built from 2,951 Wikipedia Olympic documents, benchmarked
side-by-side to answer the question the hackathon actually poses:

> **When does agentic reasoning earn its token cost, and when is it overkill?**

Our answer, measured rather than asserted:

| | Finding |
|---|---|
| **On the provided 100-question set** | Agentic reasoning is **not worth it**. GraphRAG already answers ~99% of these with a single graph query. The agent adds ~1 point for real extra cost. We report this plainly instead of hiding it. |
| **On chained questions** | Agentic reasoning is **required**. Where the question's subject must be resolved *before* it can be queried, the fixed pipeline scores **0/60** — not "worse", but structurally incapable. |
| **The crossover** | Adaptivity pays exactly when a question needs the output of one graph query as the input to the next. That is the boundary, and it is measurable. |
| **The cheap part** | The agent spends **zero** reasoning tokens on questions its free deterministic path answers *verifiably*, so the cost is paid only where it buys something. |

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

Nothing above requires a paid API key. TF-IDF embeddings and the deterministic
graph layer are free; an LLM is only needed for the generative pipelines.

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

For vector search *inside* TigerGraph, set `EMBEDDING_PROVIDER=ollama` (or
`openai`). The default TF-IDF vectors are sparse and cannot be stored in a
`LIST<DOUBLE>`; ingestion warns and writes chunks without embeddings rather than
failing silently.

---

## Choosing an LLM

Five providers, one interface, identical token accounting across all of them:

```ini
LLM_PROVIDER=groq          # groq | openrouter | ollama | anthropic | openai
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

| Provider | Notes |
|---|---|
| `groq` | Fast, generous free tier. Good default. |
| `openrouter` | One key, many models. |
| `ollama` | Fully local and free — `ollama serve && ollama pull llama3.1:8b`. Needs no key at all. |
| `anthropic` / `openai` | Supported for completeness. |

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

## Honest accounting: the regex fast path

A regex fast path covers this dataset's templated phrasings for zero tokens.
It would be easy to present its accuracy as the system's capability. It isn't —
it is a **cache over the general path**, and the repo treats it that way:

- Every compiled query is labelled `_planner: "regex" | "llm" | "none"`, and the
  counts appear in the dashboard under **planner provenance**.
- `--ablation` disables the fast path *and* deterministic triage entirely, so
  you can measure the system's true un-cached generalisation:

  ```bash
  python -m src.eval.run_benchmark --ablation
  python -m src.eval.run_benchmark --stress --ablation
  ```

Judges should read the ablation numbers as the real ones.

---

## The stress set

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

Fixed GraphRAG scores **0/60** on this set — not a tuning gap, an expressiveness
gap.

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
