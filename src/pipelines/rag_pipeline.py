"""
Pipeline 1 — plain RAG: embed the question, vector-search the chunk
store, stuff top-k into the prompt, generate.

This is the baseline the other two are measured against, so it is built
to be a FAIR opponent, not a strawman:
  - same embedder and same chunk store as the other pipelines
  - a real LLM reads the retrieved context and answers
  - top-k is generous (k=8) rather than crippled
  - the prompt is a good RAG prompt, not a deliberately bad one

It is still expected to lose on aggregation/superlative/temporal
questions, and that loss is STRUCTURAL rather than a prompt-quality
artifact: no top-k window can hold "every biathlon event at the 2018
Winter Olympics" (a dozen-plus separate documents) to count across, and
the Games chronology needed for "immediately before <year>" is stated in
no single document. Demonstrating that this is a retrieval-shape limit
rather than a model limit is the whole point of the comparison — so the
baseline has to be given every fair chance to succeed first.

With no LLM configured, run() falls back to a clearly-labelled
`proxy_mode` extraction so the repo is still runnable offline. Proxy-mode
numbers are reported separately in the dashboard and never mixed with
real-LLM numbers.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.llm_client import UsageTracker, parse_json_safely
from src.embeddings import Embedder
from src.tigergraph_client import get_graph

ANSWER_SYSTEM = """You answer questions using ONLY the provided context passages.

Rules:
- Answer with the shortest exact string that answers the question (a name, a
  number, an event title). Do not write a sentence unless asked.
- If the context does not contain enough information, set answer to null and
  say why in "gap". Never guess.
- Cite the [chunk ids] you actually relied on.

Return ONLY valid JSON:
{"answer": str|null, "citations": [str], "confidence": float, "gap": str}"""

# Question shapes that CAN in principle be answered from a single retrieved
# chunk. Anything else needs cross-document reasoning no top-k window gives.
SINGLE_DOC_ANSWERABLE = ("lookup", "multi_hop")


def _proxy_extract(question: str, chunk_texts: list):
    """Zero-cost stand-in for 'an LLM reads top-k and answers', used only
    when no LLM is configured. Scans every retrieved chunk (as a real LLM
    would read its whole window), and returns None rather than guessing
    for question types no top-k window can answer at all."""
    if re.search(r"how many nations", question, re.I):
        for text in chunk_texts:
            m = re.search(r"nations:\s*(\d+)", text)
            if m:
                return m.group(1)
        return None
    if re.search(r"gold medal", question, re.I):
        # chunk_text() rejoins on whitespace, so raw chunks have no newlines
        # between infobox fields — an unbounded [^\n]+ would swallow every
        # later field. Bound the match at the next known field key.
        for text in chunk_texts:
            m = re.search(r"\bgold:\s*(.+?)(?=\s+goldNOC:|\s+silver:|\s+silverNOC:|\s+bronze|$)", text)
            if m:
                return m.group(1).strip()
        return None
    return None


def run(question: str, graph=None, embedder=None, llm=None, qtype=None, k=8) -> dict:
    graph = graph or get_graph()
    embedder = embedder or Embedder()
    usage = UsageTracker()

    q_emb = embedder.embed(question)
    top = graph.vector_search(q_emb, k=k)
    retrieved_doc_ids = list({c.get("doc_id") for _, _, c in top if c.get("doc_id")})
    chunk_ids = [cid for _, cid, _ in top]

    if llm is not None:
        context = "\n\n".join(f"[{cid}] {c.get('text','')}" for _, cid, c in top)
        result = llm.complete(
            ANSWER_SYSTEM, f"Question: {question}\n\nContext passages:\n{context}",
            max_tokens=400, json_mode=True, context_text=context,
        )
        usage.record("rag_answer", result)
        parsed = parse_json_safely(result.text, default={})
        answer = parsed.get("answer")
        confidence = parsed.get("confidence", 0.5)
        gap = parsed.get("gap", "")
        proxy = False
    else:
        answer = _proxy_extract(question, [c.get("text", "") for _, _, c in top]) \
            if qtype in SINGLE_DOC_ANSWERABLE and top else None
        confidence = 0.3 if answer else 0.0
        gap = "" if answer else "no LLM configured; top-k window cannot answer this question shape"
        proxy = True

    return {
        "pipeline": "rag",
        "question": question,
        "answer": answer,
        "citations": chunk_ids,
        "confidence": confidence,
        "gap": gap,
        "retrieved_chunks": chunk_ids,
        "matched_doc_ids": retrieved_doc_ids,
        "usage": usage.summary(),
        "steps": 1,
        "retrieval_backend": graph.backend,
        "embedder": embedder.describe(),
        "proxy_mode": proxy,
    }
