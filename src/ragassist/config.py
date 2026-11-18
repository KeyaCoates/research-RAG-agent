from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str):
    # read the environment when Settings() is created, not when this file is imported,
    # so command-line flags that set environment variables take effect
    return field(default_factory=lambda: os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    gcp_project: str = _env("GCP_PROJECT", "")
    gcp_region: str = _env("GCP_REGION", "us-central1")
    generation_model: str = _env("GENERATION_MODEL", "gemini-2.5-flash")
    router_model: str = _env("ROUTER_MODEL", "gemini-3.5-flash-lite")
    embedding_model: str = _env("EMBEDDING_MODEL", "text-embedding-005")
    docs_bucket: str = _env("DOCS_BUCKET", "")
    llm_backend: str = _env("LLM_BACKEND", "fake")  # "fake" | "vertex"


# USD per 1M tokens (input, output). Keep in sync with current Vertex AI pricing.
PRICING = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "fake": (0.0, 0.0),
}


def get_settings() -> Settings:
    return Settings()
