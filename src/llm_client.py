"""
Unified LLM client so every pipeline/agent calls one interface
regardless of provider, and every call's token usage is recorded.

Supported providers (set LLM_PROVIDER in .env):
  anthropic   — native Anthropic SDK
  openai      — api.openai.com
  groq        — api.groq.com/openai/v1        (OpenAI-compatible)
  openrouter  — openrouter.ai/api/v1          (OpenAI-compatible)
  ollama      — localhost:11434/v1            (OpenAI-compatible, local, free)

Everything except Anthropic goes through one OpenAI-compatible code
path; only base_url / api_key / model differ. That matters for the
benchmark: token accounting has to be identical across providers or
the Agentic-Efficiency numbers aren't comparable.

Token accounting
----------------
Every provider returns prompt/completion token counts in its usage
object. Ollama's OpenAI-compatible endpoint does too. When a provider
omits them (rare), we fall back to a tiktoken estimate rather than
silently recording 0 — a 0 would quietly corrupt the efficiency
comparison that is 15% of the hackathon rubric.

We separately track *context tokens* (tokens of retrieved evidence
actually placed in the prompt), which the rubric asks for as a
distinct number from total LLM input tokens.
"""
import json
import time
import sys, os
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# OpenAI-compatible provider endpoints. Anthropic is handled separately.
OPENAI_COMPATIBLE = {
    "openai":     {"base_url": None,                              "key_attr": "OPENAI_API_KEY"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",  "key_attr": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",    "key_attr": "OPENROUTER_API_KEY"},
    "ollama":     {"base_url": None,                              "key_attr": None},  # base_url from config
}

_ENCODER = None


def estimate_tokens(text: str) -> int:
    """Only used when a provider fails to report usage. tiktoken's
    cl100k_base is a reasonable cross-model approximation; if tiktoken
    isn't installed we fall back to the standard ~4-chars-per-token
    heuristic. Never returns 0 for non-empty text."""
    global _ENCODER
    if not text:
        return 0
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = False
    if _ENCODER:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 4)


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    context_tokens: int = 0     # subset of input_tokens that is retrieved evidence
    estimated: bool = False     # True if token counts came from estimate_tokens()
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageTracker:
    """Accumulates per-call token/latency stats across a single pipeline
    run. `summary()` is what lands in outputs/results.json and drives the
    dashboard's token-efficiency comparison, so it carries per-call
    granularity (the rubric explicitly asks for 'tokens per operation'
    and 'time per operation'), not just a total."""

    def __init__(self):
        self.calls = []

    def record(self, tag: str, result: LLMResult):
        self.calls.append({
            "tag": tag,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "context_tokens": result.context_tokens,
            "total_tokens": result.total_tokens,
            "latency_s": round(result.latency_s, 3),
            "estimated": result.estimated,
        })

    def summary(self):
        return {
            "num_calls": len(self.calls),
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "output_tokens": sum(c["output_tokens"] for c in self.calls),
            "context_tokens": sum(c["context_tokens"] for c in self.calls),
            "total_tokens": sum(c["total_tokens"] for c in self.calls),
            "latency_s": round(sum(c["latency_s"] for c in self.calls), 3),
            "calls": self.calls,
        }


class LLMClient:
    """One `complete()` interface over five providers.

    `context_text` is optional and purely for accounting: pass the
    retrieved-evidence portion of the prompt and it's measured separately
    from instruction/question tokens, which is the "context tokens vs LLM
    input tokens" split the evaluation section asks for.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or config.LLM_PROVIDER).lower()

        if self.provider == "anthropic":
            import anthropic
            if not config.ANTHROPIC_API_KEY:
                raise ValueError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self._model = model or config.ANTHROPIC_MODEL
            return

        if self.provider not in OPENAI_COMPATIBLE:
            raise ValueError(
                f"Unknown LLM_PROVIDER: {self.provider!r}. "
                f"Expected one of: anthropic, {', '.join(OPENAI_COMPATIBLE)}"
            )

        import openai
        spec = OPENAI_COMPATIBLE[self.provider]

        if self.provider == "ollama":
            # Ollama needs no real key but the OpenAI SDK requires a
            # non-empty string, and its base_url is user-configurable
            # (people run it on a remote box or a non-default port).
            base_url = config.OLLAMA_BASE_URL
            api_key = "ollama"
            self._model = model or config.OLLAMA_MODEL
        else:
            base_url = spec["base_url"]
            api_key = getattr(config, spec["key_attr"], "")
            if not api_key:
                raise ValueError(
                    f"LLM_PROVIDER={self.provider} but {spec['key_attr']} is empty in .env"
                )
            self._model = model or config.provider_model(self.provider)

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, prompt: str, max_tokens: int = 1000,
                 json_mode: bool = False, context_text: str = "") -> LLMResult:
        start = time.time()

        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self._model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
        else:
            req = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            # Not every OpenAI-compatible backend honours response_format.
            # Groq and OpenAI do; OpenRouter honours it per-model; Ollama
            # accepts it for models that support structured output. Sending
            # it when unsupported is a hard 400, so we try-with and retry
            # without rather than losing the call.
            if json_mode:
                req["response_format"] = {"type": "json_object"}
            try:
                resp = self._client.chat.completions.create(**req)
            except Exception:
                req.pop("response_format", None)
                resp = self._client.chat.completions.create(**req)

            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None

        estimated = False
        if not in_tok:
            in_tok, estimated = estimate_tokens(system + prompt), True
        if not out_tok:
            out_tok, estimated = estimate_tokens(text), True

        result = LLMResult(
            text=(text or "").strip(),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=time.time() - start,
            context_tokens=estimate_tokens(context_text) if context_text else 0,
            estimated=estimated,
            model=self._model,
        )
        if json_mode:
            result.text = _strip_json_fences(result.text)
        return result


def make_llm(provider: Optional[str] = None, model: Optional[str] = None,
             quiet: bool = False) -> Optional["LLMClient"]:
    """Best-effort constructor: returns a client, or None with a printed
    reason if no provider is usable. Callers (the benchmark, the
    pipelines) degrade to their zero-LLM paths rather than crashing, so
    the repo stays runnable on a laptop with no keys at all."""
    try:
        return LLMClient(provider=provider, model=model)
    except Exception as e:
        if not quiet:
            print(f"[llm_client] No usable LLM ({e}).")
        return None


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    return text.strip()


def parse_json_safely(text: str, default=None):
    """Tolerant JSON extraction. Small/local models (Ollama especially)
    often wrap JSON in prose even when told not to, so if a plain parse
    fails we retry on the outermost {...} span before giving up."""
    cleaned = _strip_json_fences(text or "")
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass
    return default if default is not None else {}
