"""
Ingestion for the hackathon dataset (corpus.jsonl, one JSON object per
line: doc_id, title, url, wikidata_qid, approx_tokens, text).

Two things happen per document, and only one of them can cost API calls:

1. Chunk + embed the full text -> Document/Chunk vertices with
   embeddings, for Pipeline 1 (RAG) and as fallback evidence for the
   agentic pipeline. This is the only step that touches an embedding
   API — and it doesn't have to: EMBEDDING_PROVIDER=tfidf (the default)
   or =ollama are both free.

2. If the doc carries an "[Infobox Olympic event]" block (2,162 of 2,951
   docs, and effectively all of what the eval questions probe), parse it
   deterministically into a structured record — no LLM — via
   src/ingestion/infobox.py. These become the OlympicEvent graph that
   exact aggregation/superlative/temporal/lookup queries run against.

Writes to whichever backend get_graph() selects: a live TigerGraph
instance if .env is configured, else the local in-memory store. Against
TigerGraph everything is written in batched upserts — one vertex per
HTTP round trip would be ~12,000 requests for this corpus.

Usage:
    python -m src.ingestion.build_graph                 # full corpus
    python -m src.ingestion.build_graph --limit 200      # smoke test
    python -m src.ingestion.build_graph --no-embed       # structured layer only
    python -m src.ingestion.build_graph --structured-only  # skip chunks entirely
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.embeddings import Embedder, to_storable
from src.tigergraph_client import get_graph, RealGraph
from src.ingestion.infobox import parse_olympic_event, build_games_chronology


def chunk_text(text, chunk_size=600, overlap=80):
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(corpus_path=None, limit=None, do_embed=True, structured_only=False,
          progress_every=500):
    corpus_path = corpus_path or config.CORPUS_PATH
    if not os.path.exists(corpus_path):
        print(f"No corpus found at {corpus_path}. See data/README.md.")
        return

    graph = get_graph()
    is_real = isinstance(graph, RealGraph)
    embedder = Embedder() if (do_embed and not structured_only) else None

    docs = list(load_jsonl(corpus_path))
    if limit:
        docs = docs[:limit]
    print(f"Loaded {len(docs)} documents from {corpus_path}")

    # ---------------- structured layer (always, free) ----------------
    olympic_events = []
    for doc in docs:
        event = parse_olympic_event(doc)
        if event:
            olympic_events.append(event)
    chronology = build_games_chronology(olympic_events)
    print(f"Parsed {len(olympic_events)} Olympic events / {len(chronology)} Games editions "
          f"deterministically (0 LLM calls)")

    # ---------------- chunk + embed layer ----------------
    n_chunks = 0
    t0 = time.time()
    if not structured_only:
        doc_chunks = {doc["doc_id"]: chunk_text(doc["text"]) for doc in docs}
        all_texts, index = [], []
        for doc_id, chunks in doc_chunks.items():
            for idx, c in enumerate(chunks):
                all_texts.append(c)
                index.append((doc_id, idx))
        n_chunks = len(all_texts)

        # TF-IDF's vector space depends on term frequencies across the WHOLE
        # corpus, so it must be fit before anything is embedded.
        if embedder is not None and embedder._client is None:
            print(f"Fitting TF-IDF vectorizer over {n_chunks} chunks...")
            embedder.fit_tfidf(all_texts)

        print(f"Embedding {n_chunks} chunks via {embedder.describe() if embedder else 'none'}...")
        if embedder is not None:
            vectors = []
            B = 512
            for i in range(0, n_chunks, B):
                vectors.extend(embedder.embed(all_texts[i:i + B]))
                if (i // B) % 4 == 0:
                    print(f"  embedded {min(i + B, n_chunks)}/{n_chunks} "
                          f"({time.time() - t0:.0f}s)")
        else:
            vectors = [[] for _ in range(n_chunks)]

        doc_rows = [(d["doc_id"], {"title": d.get("title", ""), "source": d.get("url", ""),
                                   "text": d["text"][:1500]}) for d in docs]
        chunk_rows = [
            (f"{doc_id}_c{idx}", {"doc_id": doc_id, "text": all_texts[i],
                                  "chunk_index": idx, "embedding": to_storable(vectors[i])})
            for i, (doc_id, idx) in enumerate(index)
        ]

        if is_real:
            sparse = chunk_rows and isinstance(chunk_rows[0][1]["embedding"], dict)
            if sparse:
                print("!! EMBEDDING_PROVIDER=tfidf produces SPARSE vectors, which cannot go "
                      "into a TigerGraph LIST<DOUBLE>. Chunks will be written WITHOUT "
                      "embeddings; set EMBEDDING_PROVIDER=ollama or openai for vector search "
                      "inside TigerGraph.")
                chunk_rows = [(cid, {**a, "embedding": []}) for cid, a in chunk_rows]
            print(f"Writing {len(doc_rows)} documents / {len(chunk_rows)} chunks to TigerGraph...")
            graph.load_documents_and_chunks(doc_rows, chunk_rows)
        else:
            for did, attrs in doc_rows:
                graph.upsert_vertex("Document", did, attrs)
            for cid, attrs in chunk_rows:
                graph.upsert_vertex("Chunk", cid, attrs)

    # ---------------- write the structured layer ----------------
    print(f"Writing {len(olympic_events)} OlympicEvent vertices...")
    graph.load_olympic_events(olympic_events, chronology)

    if hasattr(graph, "save"):
        graph.save()
        print(f"Saved local store -> {graph.store_path}")

    print(f"\nDone in {time.time() - t0:.1f}s. Backend: {graph.backend}")
    print(f"  {len(docs)} documents, {n_chunks} chunks")
    print(f"  {len(olympic_events)} Olympic events, {len(chronology)} Games editions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--structured-only", action="store_true",
                    help="skip the Document/Chunk layer entirely (fast, structured queries only)")
    args = ap.parse_args()
    build(limit=args.limit, do_embed=not args.no_embed, structured_only=args.structured_only)
