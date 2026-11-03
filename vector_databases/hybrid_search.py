"""Keyword search (BM25) and hybrid search over our tickets in Qdrant.

    uv run python -m vector_databases.hybrid_search

Run filter_search.py first. Hybrid search runs a vector search and a keyword search for
the same question, and merges the two lists with Reciprocal Rank Fusion (RRF).
"""

import uuid

from qdrant_client import models

from vector_databases.embeddings import embed
from vector_databases.filter_search import COLLECTION, client
from vector_databases.tickets import load_tickets, ticket_text
from vector_databases.tune_hnsw import wait_until_indexed

CANDIDATES = 100  # how many results each search hands to the fusion step (Qdrant's guide starts at 100)


def add_text_index() -> bool:
    """Store each ticket's subject and body in a `text` field, with a BM25 full-text index. Returns True if new."""
    if "text" in client.get_collection(COLLECTION).payload_schema:
        return False
    client.batch_update_points(COLLECTION, [
        models.SetPayloadOperation(set_payload=models.SetPayload(
            payload={"text": ticket_text(ticket)},
            points=[str(uuid.uuid5(uuid.NAMESPACE_URL, ticket["id"]))],  # the same ID ingest_tickets.py gave it
        ))
        for ticket in load_tickets()
    ])
    client.create_payload_index(COLLECTION, "text", wait=True, field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT, scoring=True))  # scoring=True: rank matches with BM25
    wait_until_indexed(COLLECTION)  # new payloads make Qdrant reorganise the collection; wait until it's done
    return True


def vector_search(question: str, limit: int = 10) -> list:
    return client.query_points(COLLECTION, query=embed([question])[0].tolist(), limit=limit,
                               with_payload=True).points


def keyword_search(question: str, limit: int = 10) -> list:
    return client.query_points(COLLECTION, query=models.TextQuery(text=question), using="text", limit=limit,
                               with_payload=True).points


def hybrid_search(question: str, limit: int = 10, weights: list | None = None) -> list:
    """Both searches in one request, merged by RRF. weights: e.g. [1, 3] gives the keyword list weight 3."""
    return client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=embed([question])[0].tolist(), limit=CANDIDATES),  # vector search
            models.Prefetch(query=models.TextQuery(text=question), using="text", limit=CANDIDATES),  # BM25
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=weights)),  # no weights: both lists count the same
        limit=limit,
        with_payload=True,
    ).points


if __name__ == "__main__":
    print(f"Text field and BM25 index: {'created' if add_text_index() else 'already there'}")

    question = "ERR-5031"
    for name, search in [("Vector", vector_search), ("Keyword", keyword_search), ("Hybrid", hybrid_search)]:
        results = search(question)
        print(f"\n{name} search for {question}: {sum(question in p.payload['text'] for p in results)} of 10 contain it")
        for point in results[:3]:
            print(f"  {point.score:6.3f}  {point.payload['subject']}")
