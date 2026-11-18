"""Prints the Gemini model names your API key can use. Model names get retired, so check here.
Usage (PowerShell, GEMINI_API_KEY set):  python scripts/list_models.py
"""
import os
import sys

from google import genai

if not os.getenv("GEMINI_API_KEY"):
    sys.exit("Set GEMINI_API_KEY first.")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for m in client.models.list():
    actions = list(getattr(m, "supported_actions", None) or [])
    if "generateContent" in actions or "embedContent" in actions:
        print(m.name, actions)
