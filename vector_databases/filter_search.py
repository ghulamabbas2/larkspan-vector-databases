"""Filter a Qdrant search, and see what goes wrong when the filtered fields have no payload index.

    uv run python -m vector_databases.filter_search

Run the indexes section first: its tune_hnsw.py made Qdrant search our tickets with the
HNSW graph, the way a big collection is searched. The next lectures import the helpers below.
"""

from qdrant_client import models

from vector_databases.embeddings import embed
from vector_databases.recall_test import client
from vector_databases.tune_hnsw import wait_until_indexed

COLLECTION = "tickets"
FILTER_FIELDS = ["plan", "product_area", "priority", "status", "channel"]


def only(**fields) -> models.Filter:
    """A filter that keeps only the tickets whose payload has all these values, e.g. only(plan="Starter")."""
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))
                               for key, value in fields.items()])


def search(question: str, query_filter=None, limit: int = 10, **params) -> list:
    """The closest tickets to the question. params: search settings, such as exact=True."""
    return client.query_points(COLLECTION, query=embed([question])[0].tolist(), query_filter=query_filter,
                               limit=limit, search_params=models.SearchParams(**params), with_payload=True).points


def add_payload_indexes(schemas: dict) -> list:
    """Create the payload indexes that are missing, then rebuild the graph so it uses them. Returns the new fields."""
    existing = client.get_collection(COLLECTION).payload_schema
    new = [field for field in schemas if field not in existing]
    for field in new:
        client.create_payload_index(COLLECTION, field, field_schema=schemas[field], wait=True)
    if new:  # Qdrant's FAQ: after adding a payload index, bump ef_construct by 1 to rebuild the graph with it
        ef = client.get_collection(COLLECTION).config.hnsw_config.ef_construct
        client.update_collection(COLLECTION, hnsw_config=models.HnswConfigDiff(ef_construct=ef + 1))
        wait_until_indexed(COLLECTION)
    return new


def found(question: str, query_filter, **params) -> str:
    """How many of exact search's filtered top 10 a normal search returns."""
    truth = {point.id for point in search(question, query_filter, exact=True)}
    results = search(question, query_filter, **params)
    return f"{len(results)} results, {len(truth & {point.id for point in results})} of the {len(truth)} exact search finds"


if __name__ == "__main__":
    question = "we stopped using it, can we have our money back?"
    starter = only(plan="Starter")

    # 1. Post-filtering: search first, then drop the tickets that fail the rule
    top10 = search(question)
    print(f"Search first, then keep Starter tickets: {sum(p.payload['plan'] == 'Starter' for p in top10)} of 10 left")

    # 2. The filter inside the query, before we create the payload indexes
    print(f"Filter inside the query, before the indexes: {found(question, starter)}")

    # 3. Payload indexes for every field we filter on, then the same search again
    new = add_payload_indexes({field: models.PayloadSchemaType.KEYWORD for field in FILTER_FIELDS})
    print(f"\nPayload indexes created: {', '.join(new) or 'none (they already existed)'}")
    print(f"Filter inside the query, with the indexes: {found(question, starter)}")

    # 4. Two strict filters at once: ACORN also checks the neighbours of filtered-out points
    question, both = "we paid two times for the same month", only(plan="Starter", priority="urgent")
    print(f"\nStarter and urgent, normal search: {found(question, both)}")
    print(f"Starter and urgent, with ACORN:    {found(question, both, acorn=models.AcornSearchParams(enable=True))}")
