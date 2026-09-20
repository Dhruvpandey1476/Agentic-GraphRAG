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
| GitHub repository | **needs pushing** (see below) |
| TigerGraph connection | **needs your Savanna credentials** (see below) |
| Demo video | **needs recording** (script below) |
| Social post tagging @TigerGraph | optional, counts in your favour |

---

## Two things only you can do

### 1. Provision Savanna and connect (~15 min)

```bash
python -m scripts.setup_tigergraph --check    # prints exact steps if unconfigured
```

Then follow README → *Connecting TigerGraph*. Once `.env` has `TG_HOST` and
`TG_SECRET`:

```bash
python -m scripts.setup_tigergraph            # schema, queries, load, verify
```

Re-run the benchmarks afterwards so the dashboard shows
`graph backend: tigergraph` instead of `mock`. **Judges will look at this
field.** The dashboard prints a warning banner while it says `mock`.

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
> "We benchmarked RAG, GraphRAG and Agentic GraphRAG on the provided corpus.
> Our headline result is that on the provided question set, **agentic reasoning
> is not worth its cost** — GraphRAG already answers 99% of it with a single
> query. So we built a harder benchmark to find where it *does* pay. Here's the
> boundary we found."

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

Show the live trace with the `resolve_slot` → `repair_literal` → `graph_query`
→ `verify` steps.

**2:20–3:10 — The benchmark** (show the dashboard)
Flip between the Public and Stress tabs. Point at the verdict column:
*AGENT NOT WORTH IT* on the public set, *AGENT NEEDED* on the stress set where
fixed GraphRAG scores 0/60. Then the escalation-rate card: the agent spends zero
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

## Likely judge questions, and the honest answers

**"Isn't the regex path just fitting the eval set?"**
Yes — which is why it is labelled as a cache, reported separately as planner
provenance, and disableable with `--ablation`. The ablation numbers are the
system's real accuracy.

**"You generated your own benchmark. Isn't that self-serving?"**
The gold answers are computed from the graph, not from our system's output —
there is no LLM in the labelling path, and our own fixed pipeline scores 0/60 on
it. It is reproducible from a seed and the generator is ~250 readable lines.

**"Why is your agent only ~1 point better on the provided set?"**
Because on that set it shouldn't be. Those questions are single-hop over a
well-modelled graph. Reporting a fake win there would be the wrong answer to the
hackathon's actual question.

**"How much of this depends on the LLM being good?"**
Less than usual, deliberately. The grounding check and all three repair classes
are deterministic. We measured with a local 8B model that reliably fails to
decompose, and chained questions still resolve correctly.
