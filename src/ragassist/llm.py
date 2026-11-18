"""LLM backends. `fake` runs offline; `vertex` calls Gemini (Vertex AI, or the Gemini API when
GEMINI_API_KEY is set)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from ragassist.config import get_settings


@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int


class FakeLLM:
    """Deterministic stub: echoes the top context line. Lets CI exercise the full pipeline."""

    def generate(self, prompt: str, model: str = "fake") -> LLMResult:
        ctx = prompt.split("CONTEXT:", 1)[-1].strip().splitlines()
        answer = ctx[0] if ctx and ctx[0] else "I can't answer that from the provided sources."
        return LLMResult(answer, len(prompt) // 4, len(answer) // 4)


class VertexLLM:
    def __init__(self):
        from google import genai  # lazy import: optional dependency

        s = get_settings()
        api_key = os.getenv("GEMINI_API_KEY")  # free Google AI Studio key
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(vertexai=True, project=s.gcp_project, location=s.gcp_region)

    def generate(self, prompt: str, model: str) -> LLMResult:
        from ragassist.retrieval.embedders import _sleep, retry_delay

        for attempt in range(6):
            try:
                r = self.client.models.generate_content(model=model, contents=prompt)
                u = r.usage_metadata
                return LLMResult(r.text or "", u.prompt_token_count or 0, u.candidates_token_count or 0)
            except Exception as e:
                if attempt == 5:
                    raise
                _sleep(retry_delay(str(e), attempt))  # rate limits say how long to wait
        raise RuntimeError("unreachable")


def get_llm():
    return VertexLLM() if get_settings().llm_backend == "vertex" else FakeLLM()
