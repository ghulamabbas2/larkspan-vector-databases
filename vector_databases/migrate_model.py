"""Move the tickets to a better embedding model without downtime: a new collection, then an alias switch.

    uv run python -m vector_databases.migrate_model

Run the real-world search section first: the new collection copies the payload indexes and
the BM25 text index of `tickets`. The steps follow Qdrant's guide "Migrate to a New Embedding
Model with Zero Downtime". Safe to run again: each step skips what's already done.
"""

import uuid

from qdrant_client import models

import settings
from vector_databases.embeddings import embed
from vector_databases.filter_search import FILTER_FIELDS, client
from vector_databases.hybrid_search import CANDIDATES
from vector_databases.score_search import golden, reciprocal_rank, score
from vector_databases.tickets import load_tickets, ticket_text
from vector_databases.tune_hnsw import wait_until_indexed

ALIAS = "tickets_live"  # the name the app searches from now on
OLD, NEW = "tickets", "tickets_large"
MODELS = {OLD: settings.OPENAI_EMBED_MODEL, NEW: "text-embedding-3-large"}  # the model each collection was built with


def live_collection() -> str | None:
    """The collection the alias points at, or None if there's no alias yet."""
    return next((alias.collection_name for alias in client.get_aliases().aliases if alias.alias_name == ALIAS), None)


def point_alias_at(collection: str) -> None:
    """Move the alias to this collection. Both operations run as one step, so no search sees a gap."""
    operations = []
    if live_collection():
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS)))
    operations.append(models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=collection, alias_name=ALIAS)))
    client.update_collection_aliases(change_aliases_operations=operations)


def vector_search(question: str, collection: str = ALIAS, limit: int = 10) -> list:
    """Search one collection, or the alias, with the model that collection was built with."""
    model = MODELS[live_collection() if collection == ALIAS else collection]
    return client.query_points(collection, query=embed([question], model=model)[0].tolist(), limit=limit,
                               with_payload=True).points


def hybrid_search(question: str, collection: str, limit: int = 10) -> list:
    """The hybrid search from the real-world search section (vector + BM25, merged by RRF), on any collection."""
    return client.query_points(
        collection,
        prefetch=[models.Prefetch(query=embed([question], model=MODELS[collection])[0].tolist(), limit=CANDIDATES),
                  models.Prefetch(query=models.TextQuery(text=question), using="text", limit=CANDIDATES)],
        query=models.RrfQuery(rrf=models.Rrf()),
        limit=limit,
        with_payload=True,
    ).points


def create_new_collection() -> bool:
    """The new collection, sized for the new model, with the same payload indexes as the old one. True if new."""
    if client.collection_exists(NEW):
        return False
    size = len(embed([ticket_text(load_tickets()[0])], model=MODELS[NEW])[0])  # 3,072 numbers for 3-large
    client.create_collection(
        NEW,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(full_scan_threshold=10),  # searched with the graph, like tickets
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10),
    )
    # Qdrant's checklist: create payload indexes before loading the data
    for field, index in client.get_collection(OLD).payload_schema.items():
        client.create_payload_index(NEW, field, field_schema=index.params or index.data_type, wait=True)
    return True


def save_ticket(ticket: dict) -> None:
    """Dual write: while the migration runs, every new or changed ticket is saved in both collections."""
    payload = {"ticket_id": ticket["id"], "subject": ticket["subject"], "body": ticket["body"],
               **{field: ticket[field] for field in FILTER_FIELDS + ["customer_id"]}, "text": ticket_text(ticket)}
    for collection in (OLD, NEW):
        vector = embed([payload["text"]], model=MODELS[collection])[0]
        client.upsert(collection, [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, ticket["id"])),
                                                      vector=vector.tolist(), payload=payload)], wait=True)


def backfill(batch_size: int = 100) -> int:
    """Copy every point of the old collection into the new one, embedded with the new model."""
    offset, copied = None, 0
    while True:
        records, offset = client.scroll(OLD, limit=batch_size, offset=offset, with_payload=True, with_vectors=False)
        vectors = embed([record.payload["text"] for record in records], model=MODELS[NEW])
        client.upsert(NEW, [models.PointStruct(id=record.id, vector=vector.tolist(), payload=record.payload)
                            for record, vector in zip(records, vectors)],
                      update_mode=models.UpdateMode.INSERT_ONLY,  # never overwrite a point a dual write already saved
                      wait=True)
        copied += len(records)
        if offset is None:  # scroll returns no next offset after the last page
            return copied


def code_hits(question: str) -> str:
    """How many of the top 10 tickets, searched through the alias, contain the question's text."""
    return f"{sum(question in point.payload['text'] for point in vector_search(question))} of 10"


if __name__ == "__main__":
    question = "ERR-5031"

    # 1. The alias: the app searches tickets_live, which points at a collection
    if live_collection() is None:
        point_alias_at(OLD)
    print(f"{ALIAS} points at {live_collection()}. Vector search for {question}: {code_hits(question)} contain it")

    # 2. The new collection, with the old collection's payload indexes
    print(f"\n{NEW}: {'created' if create_new_collection() else 'already there'}")

    # 3. Dual write: a ticket saved during the migration goes into both collections
    save_ticket(load_tickets()[0])
    count = client.count(NEW).count
    print(f"Dual write of one ticket: {NEW} holds {count:,} point{'' if count == 1 else 's'}")

    # 4. Backfill: every old ticket, embedded again with the new model
    print(f"Backfill: {backfill():,} points read from {OLD}")
    wait_until_indexed(NEW)
    print(f"Points: {OLD} {client.count(OLD).count:,}, {NEW} {client.count(NEW).count:,}")

    # 5. Score both on the golden set before switching. The bar: MRR at least as good as now.
    setups = {f"{OLD}, vector": lambda q: vector_search(q, OLD), f"{NEW}, vector": lambda q: vector_search(q, NEW),
              f"{OLD}, hybrid": lambda q: hybrid_search(q, OLD), f"{NEW}, hybrid": lambda q: hybrid_search(q, NEW)}
    print(f"\n{len(golden)} golden questions          hit rate@10  precision@10    MRR")
    scores = {}
    for name, search in setups.items():
        scores[name] = score(search, golden)
        print(f"{name:<29} {scores[name][0]:>11.3f} {scores[name][1]:>13.3f} {scores[name][2]:>6.3f}")

    def rank(collection, item):
        return reciprocal_rank([p.payload["ticket_id"] for p in vector_search(item["question"], collection)], set(item["right"]))
    worse = [item["question"] for item in golden if rank(NEW, item) < rank(OLD, item)]
    print(f"Questions where the new model put the first right ticket lower: {len(worse)}"
          + (f" ({', '.join(repr(q) for q in worse[:3])})" if worse else ""))

    # 6. Switch the alias only if the new model clears the bar
    if scores[f"{NEW}, vector"][2] >= scores[f"{OLD}, vector"][2]:
        if live_collection() != NEW:
            point_alias_at(NEW)
        print(f"\nThe new model clears the bar. {ALIAS} points at {live_collection()}.")
    else:
        print(f"\nThe new model doesn't clear the bar, so {ALIAS} stays on {live_collection()}.")
    print(f"Vector search for {question} through {ALIAS}: {code_hits(question)} contain it")
