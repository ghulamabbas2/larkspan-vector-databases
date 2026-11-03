"""Turn many texts into embeddings: in batches, and never pay for the same text twice.

    from vector_databases.embeddings import embed
    vectors = embed(["first text", "second text"])   # one row per text, same order

Every vector is saved in .cache/embeddings.sqlite. Next time, the same text with the
same model comes straight from that file, with no API call.
"""

import hashlib
import sqlite3
from functools import cache

import numpy as np
import openai

import settings

BATCH_SIZE = 500  # texts per request. The API allows up to 2,048 texts and 300,000 tokens.
CACHE_FILE = settings.ROOT / ".cache" / "embeddings.sqlite"
PRICE_PER_MILLION = 0.02  # text-embedding-3-small, dollars per 1M tokens. Checked 2026-10-31.

usage = {"requests": 0, "tokens": 0, "from_cache": 0}  # what the calls in this run used


def embed(texts: list[str], provider: str = settings.EMBED_PROVIDER, model: str | None = None) -> np.ndarray:
    # model: another model of the same provider, e.g. "text-embedding-3-large"; None means the one in .env
    model = model or (settings.OPENAI_EMBED_MODEL if provider == "openai" else settings.LOCAL_EMBED_MODEL)
    db = open_cache()
    keys = [cache_key(model, text) for text in texts]

    vectors = {}
    for key in set(keys):
        row = db.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
        if row:
            vectors[key] = np.frombuffer(row[0], dtype=np.float32)
    usage["from_cache"] += sum(key in vectors for key in keys)

    # Each different text that isn't cached yet, sent in batches
    missing = list({key: text for key, text in zip(keys, texts) if key not in vectors}.items())
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start + BATCH_SIZE]
        new = call_model([text for _, text in batch], provider, model)
        for (key, _), vector in zip(batch, new):
            vectors[key] = np.asarray(vector, dtype=np.float32)
            db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, vectors[key].tobytes()))
        db.commit()

    return np.array([vectors[key] for key in keys])


def call_model(texts: list[str], provider: str, model: str) -> list:
    usage["requests"] += 1
    if provider == "openai":
        response = openai.OpenAI().embeddings.create(model=model, input=texts)
        usage["tokens"] += response.usage.total_tokens
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
    return list(local_model(model).embed(texts))  # free: the model runs on your computer


@cache
def local_model(name: str):
    from fastembed import TextEmbedding  # downloads the model (about 67 MB) the first time
    return TextEmbedding(name)


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


def open_cache() -> sqlite3.Connection:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(CACHE_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB)")
    return db
