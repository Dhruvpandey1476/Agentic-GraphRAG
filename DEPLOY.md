# Deploying

Two things can be deployed, and they have very different requirements:

| | What it needs | Cost | Always on |
|---|---|---|---|
| **Benchmark dashboard** | static files only | free | yes |
| **Live query API** | TigerGraph + an LLM + an embedder | free tier possible | sleeps on free tiers |

Deploy the dashboard first. It is the page judges will actually open, it cannot
break, and it needs no backend at all.

---

## 1. Benchmark dashboard — static, free

`dashboard/index.html` reads the benchmark JSON in `outputs/` and renders it.
Nothing else is involved.

### GitHub Pages

`site/` is already in the repo, and `.github/workflows/pages.yml` publishes it
on every push to `main`. One-time setup:

**Settings → Pages → Source → "GitHub Actions"**

That is the whole thing. The first deploy runs within a minute; afterwards
check the **Actions** tab, or trigger it by hand with **Run workflow**.

> Do not look for a `/site` option under "Deploy from a branch" — that mode
> only offers the repo root or `/docs`, which is why this uses a workflow.

The site lands at `https://<user>.github.io/<repo>/`:

| Path | What it is |
|---|---|
| `/` | redirects to the dashboard |
| `/dashboard/index.html` | benchmark dashboard — fully working, no backend |
| `/dashboard/live.html` | live query page — inert until you give it an API |
| `/outputs/*.json` | the raw benchmark data it reads |

`site/` mirrors the layout Flask serves, so the dashboard's `../outputs/`
fetches and the nav links between the two pages work unchanged.

**Refreshing it after a new benchmark run:**

```bash
cp outputs/summary_v2_*.json outputs/results_v2_*.json \
   outputs/summary_hidden_final.json outputs/results_hidden_final.json site/outputs/
cp dashboard/*.html dashboard/*.css dashboard/*.js site/dashboard/
git add site && git commit -m "Refresh published dashboard" && git push
```

### If Actions is unavailable

Copy the site into `docs/` and use **Deploy from a branch → `main` / `/docs`**:

```bash
cp -r site/* docs/ && git add docs && git commit -m "Publish dashboard" && git push
```

Netlify and Vercel work the same way — point them at `site/` with no build
command.

---

## 2. Live query API

### What it actually needs

**TigerGraph.** Any Savanna workspace with the schema loaded
(`python -m scripts.setup_tigergraph`). Set `TG_HOST`, `TG_GRAPH`,
`TG_USERNAME`, `TG_PASSWORD`.

> **Savanna's starter tier idles workspaces out.** A stopped workspace answers
> `/api/ping` with a 500 and every query fails. This bit us mid-benchmark. If
> the demo has to survive unattended, use a tier that does not idle, or be
> ready to hit **Connect** before showing it.

**An LLM.** Ollama is the default for local runs and is *not* an option on a
PaaS — llama3 wants ~6 GB resident, which no free tier gives you. Use a hosted
provider instead; `config.py` already supports Groq, OpenRouter, Anthropic and
OpenAI:

```
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0
```

**An embedder — and this is the one that bites.** The `Chunk` vectors in the
graph were written by `nomic-embed-text` (768-dim). If the server embeds
questions with anything else, cosine similarity is being computed between two
unrelated vector spaces. **It does not raise.** Retrieval just returns noise —
during development this showed up as retrieval recall of 0.01 with no error
anywhere. Either:

- point `OLLAMA_BASE_URL` at a reachable Ollama instance, or
- re-embed the corpus with a hosted model and reload (`setup_tigergraph`), or
- accept that **RAG** is degraded while **GraphRAG and Agentic GraphRAG still
  work**, since those answer through GSQL queries rather than vector search.

That last option is a legitimate demo configuration — the two graph pipelines
are what the project is about — but say so rather than letting a judge think
RAG is losing because it is bad.

### Render

`render.yaml` is in the repo. Push, then **New → Blueprint**, point it at the
repo, and fill in the secrets it marks `sync: false`.

Or by hand: **New → Web Service**, build `pip install -r requirements.txt`,
start:

```
gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300
```

### Railway / Fly.io / Heroku

The `Procfile` covers all three:

```
web: gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300
```

### Why one worker and a 300-second timeout

`RealGraph.vector_search` pulls all 9,065 chunk vectors on first use (~60s) and
caches them **in process memory**. A second worker is a second copy of that
cache and a second cold start, not more throughput — so scale threads, not
workers. The long timeout is because the first request pays that fetch, and
each question runs three pipelines.

A free tier that sleeps will cold-start on the first hit. Warm it before a demo:

```bash
curl -s https://<your-app>/api/health
```

---

## 3. Pointing the static pages at the deployed API

Edit `site/config.js`:

```js
window.API_BASE = "https://agentic-graphrag.onrender.com";
```

and set `ALLOWED_ORIGIN` on the backend to the static site's origin:

```
ALLOWED_ORIGIN=https://<user>.github.io
```

Without it the browser blocks the call with a CORS error. Leave both unset when
Flask serves the pages itself — then everything is same-origin and no CORS is
involved.

---

## 4. Local — what the judges can reproduce

This is the configuration every number in `RESULTS.md` came from.

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in TG_* credentials
ollama serve &                # in another terminal
ollama pull llama3 && ollama pull nomic-embed-text

python -m scripts.setup_tigergraph    # schema, queries, load, verify
python server.py                      # http://localhost:5000
```

`/` redirects to the live page; the benchmark dashboard is at
`/dashboard/index.html`.

To regenerate the published numbers end to end:

```bash
bash verify_then_sweep.sh     # verifies the chunk fetch, then runs all four stages
```

It refuses to start if the chunk fetch is short or the workspace is stopped,
and `make_report` refuses to write a report whose sections disagree about the
backend, model or embedder.

---

## Pre-demo checklist

```bash
curl -s "$TG_HOST/api/ping"                       # 200, not 500 (workspace awake)
curl -s http://localhost:11434/api/tags           # 200 if using local Ollama
curl -s http://localhost:5000/api/health | python -m json.tool
```

`/api/health` should report `graph_backend: tigergraph` — **not `mock`**. The
mock backend is a real fallback that keeps the repo runnable offline, and it
will happily serve a demo while answering from an in-memory store instead of
TigerGraph. Check it before you present.

Then run one question through the UI so the chunk cache is warm. The first
query is slow; every one after it is not.
