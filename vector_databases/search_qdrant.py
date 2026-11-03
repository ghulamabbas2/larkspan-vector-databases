"""Search the tickets in Qdrant by meaning, and print how close each one is.

    uv run python -m vector_databases.search_qdrant

Run vector_databases.qdrant_store first, so the tickets are in Qdrant.
"""

from qdrant_client import QdrantClient

import settings
from vector_databases.embeddings import embed

client = QdrantClient(url=settings.QDRANT_URL)


def search(question: str, k: int = 3) -> None:
    vector = embed([question])[0]  # the same model as the tickets
    results = client.query_points("tickets", query=vector.tolist(), limit=k)

    print(f'\n"{question}"')
    for rank, point in enumerate(results.points, start=1):
        ticket = point.payload
        print(f"{rank}. {point.score:.2f}  {ticket['ticket_id']}  {ticket['priority']:<7} {ticket['subject']}")


search("I was charged twice this month")
search("I can't log in to my account")
