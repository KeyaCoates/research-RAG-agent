"""Structure-aware chunking with parent-child links.

Documents are split on markdown-style headings first (sections = parents),
then sections are packed into child chunks of ~max_chars. Retrieval matches
children; generation can expand to the parent for fuller context.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    doc_id: str
    parent_id: str
    section: str
    text: str
    meta: dict = field(default_factory=dict)


def content_hash(text: str) -> str:
    """Stable hash used for incremental ingestion / dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("", text.strip())] if text.strip() else []
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(("", text[: matches[0].start()].strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            sections.append((m.group(2).strip(), body))
    return sections


def _pack(paragraphs: list[str], max_chars: int) -> list[str]:
    out, cur = [], ""
    for p in paragraphs:
        # hard-split oversized paragraphs
        while len(p) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:max_chars])
            p = p[max_chars:]
        if cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def chunk_document(doc_id: str, text: str, max_chars: int = 1000, meta: dict | None = None) -> list[Chunk]:
    meta = meta or {}
    chunks: list[Chunk] = []
    for s_idx, (title, body) in enumerate(split_sections(text)):
        parent_id = f"{doc_id}#s{s_idx}"
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for c_idx, piece in enumerate(_pack(paras, max_chars)):
            prefixed = f"{title}\n{piece}" if title else piece  # heading context helps retrieval
            chunks.append(
                Chunk(
                    id=f"{parent_id}c{c_idx}",
                    doc_id=doc_id,
                    parent_id=parent_id,
                    section=title,
                    text=prefixed,
                    meta={**meta, "hash": content_hash(piece)},
                )
            )
    return chunks
