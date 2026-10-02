#!/usr/bin/env bash
# Same 20 questions, same pipelines, two models — the only variable is the model.
cd "$(dirname "$0")"; unset TG_HOST
export EMBEDDING_PROVIDER=ollama PYTHONUNBUFFERED=1
LLM_PROVIDER=groq GROQ_MODEL=openai/gpt-oss-120b LLM_TOKENS_PER_MINUTE=6000 \
  python -u -m src.eval.run_benchmark --limit 20 --pipelines graphrag,agentic_graphrag \
  --suffix _groqcheck > outputs/run_groqcheck.log 2>&1
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3 LLM_TOKENS_PER_MINUTE=0 \
  python -u -m src.eval.run_benchmark --limit 20 --pipelines graphrag,agentic_graphrag \
  --suffix _llamacheck > outputs/run_llamacheck.log 2>&1
echo CMP_DONE
