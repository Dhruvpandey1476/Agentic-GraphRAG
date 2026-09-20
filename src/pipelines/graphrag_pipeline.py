"""
Pipeline 2 — GraphRAG: a FIXED, non-adaptive sequence.

  1. Compile the question into ONE graph query (query_planner) and run it.
  2. If that returns an answer, accept it — even if the executor flagged
     the result as ambiguous. No second look.
  3. If it returned nothing, fall back to entity-link -> 1-hop traversal
     -> supporting-document retrieval -> generate.

The sequence never changes shape based on what comes back. That is
precisely what distinguishes it from Pipeline 3: the orchestrator in
src/agents/orchestrator.py re-decides its next move after every step,
can issue a SECOND query when the first is under-constrained, and can
refuse to stop while a gap remains. Holding everything else constant —
same planner, same graph, same executor, same model — isolates
*adaptivity* as the single independent variable in the benchmark.

That control matters for the hackathon's actual research question. If
Pipeline 2 and Pipeline 3 differed in retrieval quality as well as
adaptivity, any accuracy gap would be unattributable.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm_client import UsageTracker, parse_json_safely
from src.tigergraph_client import get_graph
from src.agents import structured_agent as sa
from src.agents import query_planner as qp

ENTITY_EXTRACT_SYSTEM = """Extract the key named entities/concepts in this
question that would be worth looking up in a knowledge graph.
Return ONLY JSON: {"entities": [str, ...]} (max 3, most important first)."""

ANSWER_SYSTEM = """You answer questions using the provided graph evidence.
Answer with the shortest exact string that answers the question. If the
evidence is insufficient, set answer to null.
Return ONLY valid JSON: {"answer": str|null, "citations": [str], "confidence": float}"""


def run(question: str, graph=None, llm=None, force_llm_planner=False) -> dict:
    graph = graph or get_graph()
    usage = UsageTracker()
    disciplines = graph.known_disciplines()

    # ---- Step A: compile one graph query and run it ----
    spec, planner_result = qp.plan(question, disciplines, llm=llm, force_llm=force_llm_planner)
    if planner_result is not None:
        usage.record("query_planner", planner_result)

    if spec.get("type") != "unstructured":
        result = sa.execute(spec, graph)
        if result["answer"] is not None:
            return {
                "pipeline": "graphrag",
                "question": question,
                "answer": result["answer"],
                # A fixed pipeline accepts an ambiguous hit as final. It has
                # no mechanism to do anything else — that limitation is the
                # experiment, not an oversight.
                "confidence": 0.5 if result.get("ambiguous") else 0.95,
                "citations": result["matched_doc_ids"],
                "matched_doc_ids": result["matched_doc_ids"],
                "query_spec": {k: v for k, v in spec.items() if k != "_llm_usage"},
                "planner": spec.get("_planner"),
                "ambiguous": bool(result.get("ambiguous")),
                "usage": usage.summary(),
                "steps": 2,
                "retrieval_backend": graph.backend,
            }

    # ---- Step B: generic fallback for anything the query layer missed ----
    if llm is None:
        return {"pipeline": "graphrag", "question": question, "answer": None,
                "citations": [], "matched_doc_ids": [], "confidence": 0.0,
                "planner": spec.get("_planner"),
                "note": "query did not resolve and no LLM configured for the fallback path",
                "usage": usage.summary(), "steps": 1, "retrieval_backend": graph.backend}

    r1 = llm.complete(ENTITY_EXTRACT_SYSTEM, question, max_tokens=150, json_mode=True)
    usage.record("entity_extract", r1)
    entity_names = parse_json_safely(r1.text, default={}).get("entities", []) or []

    seed_entities = []
    for name in entity_names:
        try:
            seed_entities.extend(graph.entity_lookup(name)[:1])
        except Exception:
            pass

    neighbor_entities = []
    for ent in seed_entities:
        eid = ent.get("entity_id") or ent.get("v_id")
        if eid:
            try:
                neighbor_entities.extend(graph.k_hop_neighborhood(eid, hops=1))
            except Exception:
                pass

    all_entities = {e.get("entity_id") or e.get("v_id"): e
                    for e in (seed_entities + neighbor_entities)}
    supporting_chunks = {}
    for eid in all_entities:
        try:
            for c in graph.entity_supporting_chunks(eid):
                supporting_chunks[c.get("chunk_id") or c.get("v_id")] = c
        except Exception:
            pass

    entity_block = "\n".join(f"- {e.get('name','?')}: {e.get('description','')}"
                             for e in all_entities.values())
    chunk_block = "\n\n".join(f"[{cid}] {c.get('text','')}"
                              for cid, c in supporting_chunks.items())
    context = f"Entities:\n{entity_block}\n\nSupporting text:\n{chunk_block}"

    r2 = llm.complete(ANSWER_SYSTEM, f"Question: {question}\n\n{context}",
                      max_tokens=400, json_mode=True, context_text=context)
    usage.record("graphrag_answer", r2)
    parsed = parse_json_safely(r2.text, default={})

    return {
        "pipeline": "graphrag",
        "question": question,
        "answer": parsed.get("answer"),
        "citations": parsed.get("citations", []),
        "matched_doc_ids": list({c.get("doc_id") for c in supporting_chunks.values() if c.get("doc_id")}),
        "confidence": parsed.get("confidence", 0.3),
        "planner": spec.get("_planner"),
        "seed_entities": list(all_entities.keys()),
        "retrieved_chunks": list(supporting_chunks.keys()),
        "usage": usage.summary(),
        "steps": 4,
        "retrieval_backend": graph.backend,
    }
