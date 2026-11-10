"""Move the tickets to a better embedding model without downtime: a new collection, then an alias switch.

    uv run python -m vector_databases.migrate_model

Run the real-world search section first. The steps follow Qdrant's guide "Migrate to a New
Embedding Model with Zero Downtime". Safe to run again.
"""

from qdrant_client import models

import settings
from vector_databases.embeddings import embed
from vector_databases.filter_search import client
from vector_databases.score_search import golden, score
from vector_databases.tune_hnsw import wait_until_indexed

ALIAS = "tickets_live"  # the name the app searches from now on
OLD, NEW = "tickets", "tickets_large"
MODELS = {OLD: settings.OPENAI_EMBED_MODEL, NEW: "text-embedding-3-large"}  # the model each collection was built with


def live_collection():
    """The collection the alias points at, or None if there's no alias yet."""
    for alias in client.get_aliases().aliases:
        if alias.alias_name == ALIAS:
            return alias.collection_name
    return None


def point_alias_at(collection):
    """Point the alias at this collection. If it already points somewhere else, it moves in one step."""
    change = models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=collection, alias_name=ALIAS))
    client.update_collection_aliases(change_aliases_operations=[change])


def search(question, collection, hybrid=False):
    """The top 10 tickets, embedded with the model of the collection (behind the alias, if we search the alias)."""
    model = MODELS[live_collection() if collection == ALIAS else collection]
    vector = embed([question], model=model)[0].tolist()
    if not hybrid:
        return client.query_points(collection, query=vector, limit=10, with_payload=True).points
    # hybrid search from the real-world search section: vector + keyword (BM25), merged by RRF
    return client.query_points(
        collection,
        prefetch=[models.Prefetch(query=vector, limit=100),
                  models.Prefetch(query=models.TextQuery(text=question), using="text", limit=100)],
        query=models.RrfQuery(rrf=models.Rrf()), limit=10, with_payload=True).points

def create_new_collection():
    """A collection sized for the new model, with the old collection's payload indexes. True if it's new."""
    if client.collection_exists(NEW):
        return False
    size = len(embed(["size check"], model=MODELS[NEW])[0])  # 3,072 numbers for 3-large
    client.create_collection(NEW, vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
                             hnsw_config=models.HnswConfigDiff(full_scan_threshold=10),  # use the graph, like tickets
                             optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10))
    for field, index in client.get_collection(OLD).payload_schema.items():  # filters and BM25 keep working
        client.create_payload_index(NEW, field, field_schema=index.params or index.data_type)
    return True


def backfill(batch_size=100):
    """Copy every ticket into the new collection, embedded again with the new model. Returns how many."""
    offset, copied = None, 0
    while True:
        records, offset = client.scroll(OLD, limit=batch_size, offset=offset, with_payload=True)
        vectors = embed([record.payload["text"] for record in records], model=MODELS[NEW])
        client.upsert(NEW, [models.PointStruct(id=record.id, vector=vector.tolist(), payload=record.payload)
                            for record, vector in zip(records, vectors)])
        copied += len(records)
        if offset is None:  # no next page: every ticket is copied
            return copied


def code_hits(question):
    """How many of the top 10 tickets, searched through the alias, contain the question's text."""
    return sum(question in point.payload["text"] for point in search(question, ALIAS))

if __name__ == "__main__":
    question = "ERR-5031"

    # 1. The app searches the alias, which points at the old collection
    if live_collection() is None:
        point_alias_at(OLD)
    print(f"{ALIAS} points at {live_collection()}. Search for {question}: {code_hits(question)} of 10 contain it")

    # 2. A new collection for the new model, and 3. the backfill
    if create_new_collection():
        backfill()
        wait_until_indexed(NEW)
    print(f"{NEW}: {client.count(NEW).count:,} tickets, embedded with {MODELS[NEW]}")

    # 3. Score both collections on the golden set, before switching
    print(f"\n{'':<22} {'hit rate@10':>11} {'precision@10':>12} {'MRR':>6}")
    mrr = {}
    for name in (OLD, NEW):
        for hybrid in (False, True):
            label = f"{name}, {'hybrid' if hybrid else 'vector'}"
            hit_rate, precision, mrr[label] = score(lambda q: search(q, name, hybrid), golden)
            print(f"{label:<22} {hit_rate:>11.3f} {precision:>12.3f} {mrr[label]:>6.3f}")

    # 4. Switch the alias only if the new model is at least as good (MRR, vector search)
    if mrr[f"{NEW}, vector"] >= mrr[f"{OLD}, vector"]:
        point_alias_at(NEW)
    print(f"\n{ALIAS} points at {live_collection()}. Search for {question}: {code_hits(question)} of 10 contain it")