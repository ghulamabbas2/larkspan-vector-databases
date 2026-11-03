"""Settings for every script in the course, read once from the .env file.

To change a model or a database address, edit .env. The code stays the same.
The API keys themselves are read straight from the environment by the
openai and anthropic libraries, so they don't appear here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Chat models. LLM_PROVIDER picks the one used by default: "anthropic" or "openai".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

# Embedding models. EMBED_PROVIDER: "openai", or "local" (free, runs on your machine).
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# The databases started by docker compose.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://larkspan:larkspan@localhost:5432/larkspan")

DATA_DIR = ROOT / "data" / "larkspan"
