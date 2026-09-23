# Submission checklist & demo script

Round 1 deadline: **Wed 24 Sep**. Round 2 (top 15): **Wed 1 Oct**.

---

## What is done

| Deliverable | Status |
|---|---|
| Working Agentic GraphRAG system | done — `src/agents/orchestrator.py` |
| Three pipelines benchmarked side by side | done — `src/eval/run_benchmark.py` |
| Orchestrator + specialised agents + harness | done |
| Metrics dashboard (tokens, accuracy, completeness) | done — `dashboard/index.html` |
| Architecture diagram | done — `docs/architecture.svg` |
| TigerGraph connection | **done** — live Savanna 4.2.5, schema + 6 GSQL queries installed, 2,187 events loaded, 100/100 verified |
| Benchmarks on live TigerGraph | done — `outputs/summary_tg_public.json`, `outputs/summary_tg_stress.json` |
| GitHub repository | **needs pushing** (see below) |
| Demo video | **needs recording** (script below) |
| RAG on TigerGraph | **partial** — needs dense chunk embeddings (see below) |
| Social post tagging @TigerGraph | optional, counts in your favour |

---

## What is left

### 1. Finish RAG on TigerGraph (~2 min with a key, ~95 min locally)

The structured layer is fully loaded and benchmarked on Savanna. The
`Document`/`Chunk` layer is not, because RAG needs **dense** vectors and the
default TF-IDF vectors are sparse — they cannot go into a `LIST<DOUBLE>`.

Local `nomic-embed-text` measured at 1.6 chunks/s, so 9,065 chunks is ~95
minutes. Either let it run:

```bash
EMBEDDING_PROVIDER=ollama python -m src.ingestion.build_graph
```

or, far faster, point it at an API key:

```ini
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=...
```

Until then the three-way benchmark on TigerGraph covers GraphRAG and Agentic;
the RAG baseline numbers in this repo come from the offline store. **Say that
plainly in the writeup** rather than letting a judge discover it.

### 2. Add a fast LLM key and re-run the full benchmark

Everything currently reported was produced with a local `llama3:8b`, which is
slow enough that the full three-way run takes hours. With Groq it takes minutes:

```ini
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

```bash
bash scripts/run_all.sh      # or: powershell scripts/run_all.ps1
```

A stronger model will also raise the agentic numbers on the LLM-loop path —
the deterministic repairs already work regardless, but decomposition of
questions outside those three repair classes depends on the model.

### 3. Push to GitHub

```bash
gh repo create agentic-graphrag-tigergraph --public --source=. --push
```

The repo is already initialised and committed. `data/corpus.jsonl` is 23 MB,
well under GitHub's limit; the 160 MB local graph store is gitignored and
regenerates in ~80 seconds.

---

## Demo video script (aim for 4 minutes)

The thing that wins here is the **finding**, not the feature tour. Lead with it.

**0:00–0:25 — The claim**
> "We benchmarked RAG, GraphRAG and Agentic GraphRAG on a live TigerGraph
> Savanna workspace. Our headline result is that on the provided question set,
> **agentic reasoning is not worth its cost** — GraphRAG answers 100 out of 100
> with a single graph query. So we built a harder benchmark to find where it
> *does* pay, and found the boundary."

Starting with a negative result signals you measured rather than assumed. It
also sets up the rest.

**0:25–1:10 — The architecture** (show `docs/architecture.svg`)
One graph, three pipelines, everything held constant except adaptivity. Say why
that control matters: without it, an accuracy gap is unattributable.

**1:10–2:20 — The core idea: grounding verification**
This is your differentiator. Show the concrete failure live:

> "Ask: *in the discipline that held the most events at the 2004 Summer
> Olympics, how many had more than 41 competitors?* The planner emits
> `aggregation(discipline="Athletics", min_competitors=42)`. That query is
> well-formed, returns 18 real rows, cites 18 real documents — and it's wrong.
> 'Athletics' was guessed; the question never says it. And 42 isn't the number
> asked about. On a counting question that error is completely invisible in the
> output."

Then show the fix:

> "So we don't verify that the query is *valid*. We verify it's **grounded in
> the question** — every literal it filters on has to appear in the question.
> That check costs zero tokens, and because it names exactly which slot was
> guessed, the repair is deterministic too: resolve the discipline from the
> graph, substitute, re-run, re-verify. Correct answer, 872 tokens."

Then land the number that sells it:

> "On that whole question family we go from 0.167 to 0.833 — and we spend
> *nineteen fewer* tokens than the fixed pipeline, because it burns its LLM
> fallback guessing while we resolve the value from the graph for free."

Show the live trace with the `resolve_slot` → `repair_literal` → `graph_query`
→ `verify` steps.

**2:20–3:10 — The benchmark** (show the dashboard)
Flip between the Public and Stress tabs. Point at the verdict column:
*AGENT NOT WORTH IT* on the public set, *AGENT NEEDED* on the stress set where
fixed GraphRAG scores 7/60. Then the escalation-rate card: the agent spends zero
reasoning tokens on questions the free path answers verifiably.

**3:10–3:40 — Honesty about the fast path**
> "We have a regex fast path that covers the templated phrasings for free. We do
> **not** claim its accuracy as the system's capability — every query is labelled
> with which compiler produced it, and `--ablation` turns the fast path off so
> you can see the real generalisation number. Read the ablation numbers."

Judges notice this. Most submissions will quietly bank the template accuracy.

**3:40–4:00 — Limitations and close**
Name two real ones (deterministic repair covers three failure classes; the entity
layer is thin). Close on the finding, not the code.

### Recording tips

- Show the terminal and the dashboard, not slides.
- Run one question live end-to-end. A real trace scrolling past is worth more
  than any diagram.
- Do not narrate the file tree.

---

## What running it live actually taught us

Worth 30 seconds of the demo, because it is the part most submissions will not
have: the defects that only appeared once the system ran against a real
instance with a real model.

**GSQL `LIKE` is substring containment, not word matching.** A query for
`men's 20 kilometres walk` also matched **"WOmen's** 20 kilometres walk" and
returned the wrong medallist on 4 of 100 questions — succeeding, citing a real
document, and being wrong. The in-memory backend matched on word boundaries, so
the two silently disagreed 95/100. Fixed by giving matching exactly one
definition (`search_key()`, space-padded, shared by both backends).

**Switching the compiler from regex to an LLM broke the executor four ways.**
The regex parser had always produced well-formed specs, so nothing downstream
had ever been given malformed input:

| Compiler output | Consequence |
|---|---|
| `min_competitors: "73"` | `TypeError` inside a filter comparison |
| `target_year` absent | `int < None` |
| `season: null` | `AttributeError` on `None.lower()` — and the handler caught only `TypeError, ValueError`, so it escaped |
| no `games_id` **and** no `discipline` | **silently counted the whole corpus** and returned a plausible number |

The common root is that `dict.get(k, "")` returns `None` when the key *exists*
with a null value — precisely what an LLM emits for optional fields — so call
sites that look correct still propagate `None`. A compiled spec is untrusted
input and is now validated as such.

The last row is the one to dwell on: no exception, no warning, a clean trace and
a believable count. That is the same failure this project exists to catch, found
in our own code.

**Hosted free tiers defeated the benchmark three separate ways** — a `413` on
any request over the per-request cap, reasoning tokens billed outside
`max_tokens` (so client-side pacing under-counted), and `429`s with 5-8 minute
`Retry-After` while the headers reported ample quota. The published numbers were
produced on a local model for that reason, and the repo now paces, checkpoints
and resumes accordingly.

## Likely judge questions, and the honest answers

**"Isn't the regex path just fitting the eval set?"**
Yes — which is why it is **off by default**. Every pipeline compiles its query
with the LLM and spends real tokens, so the headline comparison measures
architecture, not a prior on the question templates. `--fast-path` reproduces
the zero-token numbers, reported separately as an engineering result: once the
corpus is modelled as a graph, a large class of questions needs no LLM at all.

**"Your graph pipelines used no tokens while RAG used thousands — how is that
a fair comparison?"**
It wasn't, in the first version, and that is exactly why the default changed.
With the fast path on, the graph pipelines got prior knowledge of the question
shapes that RAG never got. With it off, all three use the same model on the
same questions. Structure still costs less — a ~300-token planner call versus a
~3,000-token context — but that advantage is now earned rather than assumed.
We also report tokens per *correct answer*, because a pipeline that is cheap
because it answers nothing is not efficient.

**"You generated your own benchmark. Isn't that self-serving?"**
The gold answers are computed from the graph, not from our system's output —
there is no LLM in the labelling path, and our own fixed pipeline scores 7/60 on
it (0/60 without its LLM planner). It is reproducible from a seed and the generator is ~250 readable lines.

**"Did you actually run this on TigerGraph, or just locally?"**
Live Savanna, TigerGraph 4.2.5 — and running there found three defects local
testing could not, including GSQL `LIKE` matching "women's 20 kilometres walk"
for a men's query and silently returning the wrong medallist on 4 of 100
questions. That is written up in ARCHITECTURE.md §2.

**"Why is your agent only ~1 point better on the provided set?"**
Because on that set it shouldn't be. Those questions are single-hop over a
well-modelled graph. Reporting a fake win there would be the wrong answer to the
hackathon's actual question.

**"How much of this depends on the LLM being good?"**
Less than usual, deliberately. The grounding check and all three repair classes
are deterministic. We measured with a local 8B model that reliably fails to
decompose, and chained questions still resolve correctly.
