"""
Embeddings wrapper. Three tiers, in priority order:
  1. Real API embeddings (OpenAI) if a key is configured. Dense vectors.
  2. A corpus-fitted TF-IDF vectorizer — a standard, well-understood
     classical IR technique, fit once over every chunk in the corpus
     during ingestion and persisted to outputs/tfidf_vectorizer.pkl.
     This is the default zero-cost fallback. Vectors are SPARSE
     ({index: value} dicts, not dense arrays): the full vocabulary is
     tens of thousands of terms (no max_features cap — capping by raw
     frequency would throw away exactly the rare, distinguishing terms
     that matter most here, like "RS:X", a sailing class code that
     appears in only one document), and dense storage at that width
     would be gigabytes across ~9,000 chunks. Sparse storage costs only
     the ~30-150 nonzero terms an individual chunk actually contains.
  3. Raw hashed bag-of-words — only used before any vectorizer has been
     fit (e.g. embedding the very first chunk during ingestion, or if
     the persisted vectorizer file is missing). Dense vectors.

cosine_sim() below handles both dense arrays and sparse dicts (and any
mix of the two) transparently.
"""
import hashlib
import math
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TFIDF_PATH = os.path.join(config.OUTPUT_DIR, "tfidf_vectorizer.pkl")


def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Embedder:
    """Providers: openai | ollama | tfidf (default).

    `ollama` uses the OpenAI-compatible /v1/embeddings endpoint with
    nomic-embed-text — real dense semantic vectors, running locally, at
    zero API cost. That matters here: TF-IDF is lexical only, so a
    question phrased differently from the source text ("who took first
    place" vs "gold:") scores poorly. Dense local embeddings fix that
    without needing a paid key, which is what makes the RAG baseline a
    fair opponent rather than a strawman.
    """

    def __init__(self, provider=None):
        self.provider = (provider or config.EMBEDDING_PROVIDER).lower()
        self._client = None
        self._model = None
        self._tfidf = None

        if self.provider == "openai" and config.OPENAI_API_KEY:
            import openai
            self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            self._model = config.EMBEDDING_MODEL
        elif self.provider == "ollama":
            import openai
            self._client = openai.OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
            self._model = config.OLLAMA_EMBED_MODEL
        elif os.path.exists(TFIDF_PATH):
            with open(TFIDF_PATH, "rb") as f:
                self._tfidf = pickle.load(f)

    def describe(self) -> str:
        """Human-readable provenance string, recorded into results.json so
        a reader can tell which vector space produced a given number."""
        if self._client:
            return f"{self.provider}:{self._model}"
        if self._tfidf is not None:
            return "tfidf:corpus-fitted"
        return "hashed-bow:384"

    def fit_tfidf(self, texts: list, save: bool = True):
        """Fits a TF-IDF vectorizer over the full chunk corpus. Call once
        during ingestion, before embedding individual chunks, so every
        chunk (and every question embedded later) lands in the same
        vector space. Persists to outputs/tfidf_vectorizer.pkl so
        eval/query-time processes load the exact same fitted vectorizer
        instead of re-fitting (which would produce an incompatible
        vector space)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
            min_df=1,
            # Default token_pattern drops single-char tokens and splits on
            # ':'/'+'/'-', which mangles exactly the identifiers that matter
            # most here: "RS:X" (a sailing class), "+80 kg" (a weight
            # class), "4x100m" (a relay). This keeps those intact.
            token_pattern=r"(?u)\b[\w:+\-]+\b",
        )
        self._tfidf.fit(texts)
        if save:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            with open(TFIDF_PATH, "wb") as f:
                pickle.dump(self._tfidf, f)

    def embed(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if self._client:
            # Ollama's embeddings endpoint is much happier with modest
            # batches than one 9,000-item request, and OpenAI caps input
            # size per call anyway — chunk it rather than assume.
            vecs = []
            batch = 64 if self.provider == "ollama" else 256
            for i in range(0, len(texts), batch):
                resp = self._client.embeddings.create(model=self._model, input=texts[i:i + batch])
                vecs.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
        elif self._tfidf is not None:
            sparse = self._tfidf.transform(texts).tocoo()
            rows = {}
            for r, c, v in zip(sparse.row, sparse.col, sparse.data):
                rows.setdefault(r, {})[str(c)] = float(v)
            vecs = [rows.get(i, {}) for i in range(len(texts))]
        else:
            vecs = [_hash_embedding(t) for t in texts]

        return vecs[0] if single else vecs


def to_storable(embedding):
    """JSON-safe form of whatever embed() returned — a sparse dict is
    already JSON-safe as-is; a dense numpy array needs .tolist()."""
    return embedding if isinstance(embedding, dict) else embedding.tolist()


def cosine_sim(a, b) -> float:
    """Works for two dense arrays, two sparse dicts, or one of each —
    the storage format can differ (loaded-from-JSON sparse dicts have
    string keys; a freshly embedded query might too) without the caller
    needing to know which mode ingestion ran in."""
    if isinstance(a, dict) or isinstance(b, dict):
        da = a if isinstance(a, dict) else {str(i): float(x) for i, x in enumerate(a) if x}
        db = b if isinstance(b, dict) else {str(i): float(x) for i, x in enumerate(b) if x}
        common = da.keys() & db.keys()
        dot = sum(da[k] * db[k] for k in common)
        na = math.sqrt(sum(v * v for v in da.values()))
        nb = math.sqrt(sum(v * v for v in db.values()))
        return dot / (na * nb) if na and nb else 0.0

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
