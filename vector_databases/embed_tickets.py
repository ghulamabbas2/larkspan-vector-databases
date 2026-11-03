"""Embed all 5,000 Larkspan tickets, and see what batching and the cache save.

    uv run python -m vector_databases.embed_tickets              # batched and cached
    uv run python -m vector_databases.embed_tickets one-by-one   # 100 tickets, one request each
"""

import sys
import time

import openai

import settings
from vector_databases import embeddings
from vector_databases.tickets import load_tickets, ticket_text

texts = [ticket_text(ticket) for ticket in load_tickets()]
start = time.perf_counter()

if sys.argv[1:] == ["one-by-one"]:
    client = openai.OpenAI()
    for text in texts[:100]:
        client.embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=text)
    seconds = time.perf_counter() - start
    print(f"100 tickets, one request each: {seconds:.1f} s")
    print(f"All {len(texts):,} at that speed: about {seconds / 100 * len(texts) / 60:.0f} minutes")
else:
    vectors = embeddings.embed(texts)
    seconds = time.perf_counter() - start
    used = embeddings.usage
    cost = used["tokens"] * embeddings.PRICE_PER_MILLION / 1_000_000
    print(f"{len(texts):,} tickets → array of shape {vectors.shape} in {seconds:.1f} s")
    print(f"Requests: {used['requests']}   From the cache: {used['from_cache']:,}")
    print(f"Tokens:   {used['tokens']:,}   Cost: ${cost:.4f}")
