"""Central settings. Every value can be overridden with an environment variable of the same name."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "knowledge_base"
CHROMA_DIR = ROOT / "chroma_db"

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))
