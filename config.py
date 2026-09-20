"""
Central configuration. Every value is read from environment variables
(see .env.example). Nothing is hardcoded, so the same code runs against
Savanna or Community Edition, and against Anthropic, OpenAI, Groq,
OpenRouter or a local Ollama, without touching source.
"""
import os
from dotenv import load_dotenv

load_dotenv()


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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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

# ---------------------------------------------------------------- Agent limits
MAX_AGENT_STEPS = _int("MAX_AGENT_STEPS", 8)
MAX_TOKENS_PER_RUN = _int("MAX_TOKENS_PER_RUN", 20000)

# ---------------------------------------------------------------- Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
EVAL_QUESTIONS_PATH = os.path.join(DATA_DIR, "eval_public.jsonl")
HIDDEN_QUESTIONS_PATH = os.path.join(DATA_DIR, "eval_hidden.jsonl")
