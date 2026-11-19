"""Embeds the corpus once and saves the vectors to data/phak/index.npz (about 5 MB).
Usage (PowerShell, GEMINI_API_KEY set):  python scripts/build_index.py
Uses the embedding cache from your eval runs, so it should finish in seconds, not minutes.
"""
import os
import pathlib
import sys

from ragassist.retrieval.embedders import VertexEmbedder
from ragassist.service import load_chunks, save_index

corpus = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/phak/corpus")
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/phak/index.npz")
if not os.getenv("GEMINI_API_KEY"):
    sys.exit("Set GEMINI_API_KEY first.")
chunks = load_chunks(corpus)
print(f"embedding {len(chunks)} chunks (cached ones are free)...")
vecs = VertexEmbedder().embed([c.text for c in chunks])
save_index(corpus, vecs, out)
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
