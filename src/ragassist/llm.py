"""LLM backends. `fake` runs offline; `vertex` calls Gemini on Vertex AI."""
from __future__ import annotations

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
        self.client = genai.Client(vertexai=True, project=s.gcp_project, location=s.gcp_region)

    def generate(self, prompt: str, model: str) -> LLMResult:
        r = self.client.models.generate_content(model=model, contents=prompt)
        u = r.usage_metadata
        return LLMResult(r.text or "", u.prompt_token_count or 0, u.candidates_token_count or 0)


def get_llm():
    return VertexLLM() if get_settings().llm_backend == "vertex" else FakeLLM()
