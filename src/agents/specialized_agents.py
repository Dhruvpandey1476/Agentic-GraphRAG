"""
Specialized agents. Each is a pure function: (state, graph, llm, **args)
-> List[EvidenceItem]. The orchestrator calls these by name; none of
them decide *when* they run — only *what happens* when invoked. This
separation (orchestrator = policy, agents = execution) is what makes
the "next action depends on evidence-so-far" requirement possible.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.agents.harness import EvidenceItem
from src.llm_client import parse_json_safely


def entity_linking(state, graph, llm, embedder, name: str):
    """Resolve a surface-form name to graph Entity vertices."""
    matches = graph.entity_lookup(name)
    items = []
    for m in matches[:3]:
        eid = m.get("entity_id") or m.get("v_id")
        if not eid or eid in state.visited_entities:
            continue
        state.linked_entities[eid] = m
        items.append(EvidenceItem(
            step=state.step_count, source_agent="entity_linking", kind="entity",
            ref_id=eid, content=f"{m.get('name','?')}: {m.get('description','')}",
            metadata={"entity_type": m.get("entity_type", "")},
        ))
    return items


def graph_traversal(state, graph, llm, embedder, entity_id: str, hops: int = 1):
    """Expand from a known entity to related entities."""
    if entity_id in state.visited_entities:
        return []
    state.visited_entities.add(entity_id)
    neighbors = graph.k_hop_neighborhood(entity_id, hops=hops)
    items = []
    for n in neighbors:
        eid = n.get("entity_id") or n.get("v_id")
        if not eid or eid in state.linked_entities:
            continue
        state.linked_entities[eid] = n
        items.append(EvidenceItem(
            step=state.step_count, source_agent="graph_traversal", kind="relation",
            ref_id=eid, content=f"related to seed: {n.get('name','?')} ({n.get('entity_type','')})",
        ))
    return items


def similarity_search(state, graph, llm, embedder, query: str, k: int = 5):
    """Vector search over chunks — used when graph linking finds nothing,
    or to widen recall alongside structured evidence. Goes through the
    backend's vector_search() so it works identically against TigerGraph's
    native vector index and the in-memory store."""
    q_emb = embedder.embed(query)
    try:
        top = graph.vector_search(q_emb, k=k)
    except Exception as e:
        print(f"[similarity_search] vector search failed: {e}")
        return []
    return [
        EvidenceItem(
            step=state.step_count, source_agent="similarity_search", kind="chunk",
            ref_id=cid, content=c.get("text", ""),
            metadata={"score": round(float(score), 4), "doc_id": c.get("doc_id", "")},
        )
        for score, cid, c in top
    ]


def document_retrieval(state, graph, llm, embedder, entity_id: str):
    """Pull the raw supporting text chunks for a linked entity."""
    chunks = graph.entity_supporting_chunks(entity_id)
    items = []
    for c in chunks:
        cid = c.get("chunk_id") or c.get("v_id")
        if not cid:
            continue
        items.append(EvidenceItem(
            step=state.step_count, source_agent="document_retrieval", kind="chunk",
            ref_id=cid, content=c.get("text", ""),
        ))
    return items


AGGREGATE_SYSTEM = """You merge and deduplicate evidence snippets gathered so
far into a concise briefing for downstream reasoning. Drop redundant items,
keep anything with a distinct fact. Return ONLY JSON:
{"summary": str, "distinct_facts": int}"""


def aggregation(state, graph, llm, embedder):
    """Compress accumulated evidence when it's getting large, so later
    reasoning steps aren't paying token cost for duplicated snippets."""
    block = state.evidence_text_block()
    result = llm.complete(AGGREGATE_SYSTEM, block, max_tokens=1200, json_mode=True)
    parsed = parse_json_safely(result.text, default={"summary": block, "distinct_facts": len(state.evidence)})
    item = EvidenceItem(
        step=state.step_count, source_agent="aggregation", kind="note",
        ref_id="aggregate", content=parsed.get("summary", ""),
    )
    return [item], result


MULTIHOP_SYSTEM = """Given the evidence collected so far, perform one hop of
explicit multi-step reasoning toward answering the question: state what can
now be concluded, and what is still missing. Return ONLY JSON:
{"reasoning": str, "still_missing": str}"""


def multihop_reasoning(state, graph, llm, embedder):
    prompt = f"Question: {state.question}\n\nEvidence:\n{state.evidence_text_block()}"
    result = llm.complete(MULTIHOP_SYSTEM, prompt, max_tokens=1200, json_mode=True)
    parsed = parse_json_safely(result.text, default={"reasoning": "", "still_missing": ""})
    item = EvidenceItem(
        step=state.step_count, source_agent="multihop_reasoning", kind="note",
        ref_id=f"reasoning_{state.step_count}", content=parsed.get("reasoning", ""),
        metadata={"still_missing": parsed.get("still_missing", "")},
    )
    return [item], result


EVAL_SYSTEM = """You judge whether the evidence collected so far is sufficient
to fully and confidently answer the question. Be strict: partial evidence is
NOT sufficient. Return ONLY JSON:
{"sufficient": bool, "gap": str, "confidence": float 0-1}"""


def evidence_evaluation(state, graph, llm, embedder):
    """Called before the orchestrator decides whether to stop or keep going."""
    prompt = f"Question: {state.question}\n\nEvidence so far:\n{state.evidence_text_block()}"
    result = llm.complete(EVAL_SYSTEM, prompt, max_tokens=1200, json_mode=True)
    parsed = parse_json_safely(result.text, default={"sufficient": False, "gap": "", "confidence": 0.3})
    return parsed, result


AGENT_REGISTRY = {
    "entity_linking": entity_linking,
    "graph_traversal": graph_traversal,
    "similarity_search": similarity_search,
    "document_retrieval": document_retrieval,
}
