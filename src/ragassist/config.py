from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    gcp_project: str = os.getenv("GCP_PROJECT", "")
    gcp_region: str = os.getenv("GCP_REGION", "us-central1")
    generation_model: str = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
    router_model: str = os.getenv("ROUTER_MODEL", "gemini-2.5-flash-lite")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
    docs_bucket: str = os.getenv("DOCS_BUCKET", "")
    llm_backend: str = os.getenv("LLM_BACKEND", "fake")  # "fake" | "vertex"


# USD per 1M tokens (input, output). Keep in sync with current Vertex AI pricing.
PRICING = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "fake": (0.0, 0.0),
}


def get_settings() -> Settings:
    return Settings()
