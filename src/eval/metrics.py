"""
Scoring for the real dataset. Gold answers are short exact strings
(e.g. "5", "Chen Ding"), not prose — so exact_match is the primary,
free, zero-ambiguity metric (matches the benchmark's own apparent
grading style). LLM-as-judge is offered as a secondary, more lenient
metric for phrasing differences, but costs tokens and needs a key.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm_client import LLMClient, parse_json_safely
from src.ingestion.infobox import normalize


def exact_match(candidate, gold_answers: list) -> bool:
    if candidate is None:
        return False
    c = normalize(str(candidate))
    return any(c == normalize(str(g)) for g in gold_answers)


JUDGE_SYSTEM = """You are grading an AI system's answer against a reference
answer for a factual question. Score 0-1 on:
- correctness: does it match the key facts in the reference?
- completeness: does it cover everything the reference covers?
- grounding: is it plausible given the evidence used (not hallucinated)?
Return ONLY JSON: {"correctness": float, "completeness": float,
"grounding": float, "overall": float, "rationale": str}
overall is your holistic judgment, not necessarily the mean of the three."""


def judge_answer(question, candidate_answer, reference_answer, llm=None):
    llm = llm or LLMClient()
    prompt = (f"Question: {question}\n\n"
              f"Reference answer: {reference_answer}\n\n"
              f"Candidate answer: {candidate_answer}")
    r = llm.complete(JUDGE_SYSTEM, prompt, max_tokens=300, json_mode=True)
    parsed = parse_json_safely(r.text, default={
        "correctness": 0.0, "completeness": 0.0, "grounding": 0.0,
        "overall": 0.0, "rationale": "judge parse failure",
    })
    parsed["_judge_tokens"] = r.total_tokens
    return parsed


def retrieval_recall(matched_doc_ids, gold_doc_ids) -> float:
    """Fraction of gold_doc_ids that showed up among the pipeline's
    matched/retrieved/cited doc ids — a retrieval-quality signal
    independent of whether the final answer string was correct."""
    if not gold_doc_ids:
        return None
    gold = set(gold_doc_ids)
    got = set(matched_doc_ids or [])
    return len(gold & got) / len(gold)
