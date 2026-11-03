"""Which embedding model finds the right tickets for Larkspan? OpenAI against a free local model.

eval/search_queries.json has 45 questions written in everyday words, each with the issue
(topic) it's about. For each question we take the 5 most similar tickets. It's a hit when
at least one of them is about the right issue. Each model is also tried with its vectors
cut short, to see how much a smaller vector loses.

    uv run python -m vector_databases.compare_models
"""

import json

import numpy as np

import settings
from vector_databases.embeddings import embed
from vector_databases.tickets import load_tickets, ticket_text

K = 5

tickets = load_tickets()
topics = json.loads((settings.ROOT / "eval" / "ticket_topics.json").read_text())["tickets"]
ticket_topics = np.array([topics[ticket["id"]] for ticket in tickets])
queries = json.loads((settings.ROOT / "eval" / "search_queries.json").read_text())


def shorten(vectors: np.ndarray, dims: int) -> np.ndarray:
    """Keep the first dims numbers, then scale each vector back to length 1."""
    cut = vectors[:, :dims]
    return cut / np.linalg.norm(cut, axis=1, keepdims=True)


def hits(query_vectors: np.ndarray, ticket_vectors: np.ndarray) -> int:
    scores = query_vectors @ ticket_vectors.T  # cosine similarity, because every vector has length 1
    top = np.argsort(-scores, axis=1)[:, :K]
    right = np.array([query["topic"] for query in queries])
    return int((ticket_topics[top] == right[:, None]).any(axis=1).sum())


print(f"{len(queries)} questions · a hit = the right issue in the top {K} of {len(tickets):,} tickets\n")
print(f"{'Model':26} {'Dimensions':>10} {'Hits':>9}")

for provider, short in [("openai", 256), ("local", 128)]:
    ticket_vectors = embed([ticket_text(ticket) for ticket in tickets], provider)
    query_vectors = embed([query["query"] for query in queries], provider)
    model = settings.OPENAI_EMBED_MODEL if provider == "openai" else settings.LOCAL_EMBED_MODEL
    full = ticket_vectors.shape[1]
    for dims in (full, short):
        found = hits(shorten(query_vectors, dims), shorten(ticket_vectors, dims))
        print(f"{model:26} {dims:>10} {found:>4}/{len(queries)}")
