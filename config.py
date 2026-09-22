"""
Central configuration. Every value is read from environment variables
(see .env.example). Nothing is hardcoded, so the same code runs against
Savanna or Community Edition, and against Anthropic, OpenAI, Groq,
OpenRouter or a local Ollama, without touching source.
"""
import os
from dotenv import load_dotenv, dotenv_values

# load_dotenv() does NOT override variables already present in the
# environment — correct precedence for CI, but a real footgun locally: a
# shell variable left over from an earlier command silently beats .env.
# That is how a run ended up embedding questions with TF-IDF while the
# stored chunks were dense, which produces meaningless cosine scores rather
# than an error. Shadowing is legitimate, so it is reported, not overridden.
load_dotenv()

_SHADOWED = {
    k: (v, os.environ[k])
    for k, v in (dotenv_values() or {}).items()
    if k in os.environ and v is not None and os.environ[k] != v
}
if _SHADOWED:
    print("[config] shell environment overrides .env for: " + ", ".join(
        f"{k} (.env={dot!r} -> using {env!r})" for k, (dot, env) in _SHADOWED.items()))


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- TigerGraph
TG_HOST = os.getenv("TG_HOST", "")
TG_GRAPH = os.getenv("TG_GRAPH", "HackathonGraph")
TG_USERNAME = os.getenv("TG_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TG_PASSWORD", "")
TG_SECRET = os.getenv("TG_SECRET", "")
# Savanna serves GSQL on 443 behind the same hostname; Community Edition
# uses 14240. pyTigerGraph needs it explicitly when it isn't the default.
TG_GS_PORT = os.getenv("TG_GS_PORT", "")
TG_RESTPP_PORT = os.getenv("TG_RESTPP_PORT", "")
# Set to "1" to force MockGraph even when TG credentials exist — useful
# for reproducing benchmark numbers offline.
FORCE_MOCK_GRAPH = os.getenv("FORCE_MOCK_GRAPH", "") == "1"

# ---------------------------------------------------------------- LLM
# One of: anthropic | openai | groq | openrouter | ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq retired the Llama 3.x hosted models; gpt-oss-120b is the current
# strongest general model there. Check /v1/models if this 404s.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def provider_model(provider: str) -> str:
    """Model id for an OpenAI-compatible provider name."""
    return {
        "openai": OPENAI_MODEL,
        "groq": GROQ_MODEL,
        "openrouter": OPENROUTER_MODEL,
        "ollama": OLLAMA_MODEL,
    }.get(provider, OPENAI_MODEL)


# ---------------------------------------------------------------- Embeddings
# One of: tfidf | openai | ollama | tigergraph
#   tfidf      — zero-cost, corpus-fitted, no API key needed (default)
#   openai     — text-embedding-3-small
#   ollama     — nomic-embed-text, local and free
#   tigergraph — embed with whatever EMBEDDING_FALLBACK says, but store and
#                search vectors natively in TigerGraph's vector index
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "tfidf").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Vector search backend for the RAG pipeline: "local" (cosine in Python)
# or "tigergraph" (native TigerGraph vector index).
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "local").lower()

# ---------------------------------------------------------------- RAG
# Context budget for the RAG baseline, in tokens. Chunks are packed by
# descending similarity until the budget is reached, rather than always
# sending a fixed k. This is both better RAG practice (production RAG
# always has a budget) and a hard requirement on hosted free tiers: Groq's
# on-demand tier rejects any single request over 8,000 tokens outright with
# a 413, so a fixed k=8 (~8.5k tokens) could not run there at all.
# The regex question-parser is an OPT-IN cache, not the default.
#
# It resolves this dataset's templated phrasings for zero tokens, which
# makes GraphRAG and Agentic look free next to a RAG baseline that must
# always call an LLM. But it only costs zero because it encodes prior
# knowledge of what the questions look like — a prior RAG is not given. So
# using it by default compares a pre-tuned system against an untuned one
# and reports the gap as if it were architecture.
#
# Default OFF: every pipeline compiles its query with the LLM, so all three
# spend real tokens and the comparison measures architecture rather than
# eval-fitting. Set USE_REGEX_FAST_PATH=1 to measure the cached path.
USE_REGEX_FAST_PATH = os.getenv("USE_REGEX_FAST_PATH", "0") == "1"

RAG_TOP_K = _int("RAG_TOP_K", 8)
# Client-side pacing for hosted free tiers, in tokens per minute. 0 = off.
# Groq's on-demand tier allows 8,000 TPM and penalises bursts with
# Retry-After values of 20-30 MINUTES, so waiting for a 429 is far more
# expensive than never causing one. Set slightly under the real limit.
LLM_TOKENS_PER_MINUTE = _int("LLM_TOKENS_PER_MINUTE", 0)
RAG_MAX_CONTEXT_TOKENS = _int("RAG_MAX_CONTEXT_TOKENS", 4000)

# ---------------------------------------------------------------- Agent limits
MAX_AGENT_STEPS = _int("MAX_AGENT_STEPS", 8)
MAX_TOKENS_PER_RUN = _int("MAX_TOKENS_PER_RUN", 20000)

# ---------------------------------------------------------------- Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
EVAL_QUESTIONS_PATH = os.path.join(DATA_DIR, "eval_public.jsonl")
HIDDEN_QUESTIONS_PATH = os.path.join(DATA_DIR, "eval_hidden.jsonl")
