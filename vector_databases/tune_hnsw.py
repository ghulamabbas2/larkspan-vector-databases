"""Switch on the HNSW index for our tickets, then see how its settings trade accuracy for speed.

    uv run python -m vector_databases.tune_hnsw

Run vector_databases.ingest_tickets first, so the tickets are in Qdrant.
The quantization and benchmark lectures import the two helpers below.
"""

import time

from qdrant_client import models

from vector_databases.embeddings import embed
from vector_databases.recall_test import client, recall_at_k
from vector_databases.tickets import load_tickets, ticket_text


def wait_until_indexed(collection: str) -> float:
    """Wait until every vector is in the HNSW graph. Returns the seconds it took."""
    start = time.perf_counter()
    while True:
        info = client.get_collection(collection)
        if info.status == models.CollectionStatus.GREEN and info.indexed_vectors_count >= info.points_count:  # Qdrant says this count is approximate, and it can go above points_count
            return time.perf_counter() - start
        time.sleep(0.2)


def practice_collection(name: str, hnsw=None, quantization=None, memory=None) -> float:
    """A fresh copy of the 5,000 ticket vectors with the given settings. Returns seconds to build."""
    vectors = embed([ticket_text(ticket) for ticket in load_tickets()])  # from the cache
    if client.collection_exists(name):
        client.delete_collection(name)
    start = time.perf_counter()
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE, memory=memory),
        hnsw_config=(hnsw or models.HnswConfigDiff()).model_copy(update={"full_scan_threshold": 10}),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10),  # index it, small as it is
        quantization_config=quantization,
    )
    client.upload_points(name, [models.PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vectors)], wait=True)
    wait_until_indexed(name)
    return time.perf_counter() - start


if __name__ == "__main__":
    # 1. Our tickets are too small for Qdrant to build the index by itself. Lower both thresholds.
    print(f"Before: {client.get_collection('tickets').indexed_vectors_count:,} vectors in the HNSW index")
    client.update_collection(
        "tickets",
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10),  # build the graph (in KB)
        hnsw_config=models.HnswConfigDiff(full_scan_threshold=10),  # and search with it, instead of a full scan
    )
    seconds = wait_until_indexed("tickets")
    print(f"After:  {client.get_collection('tickets').indexed_vectors_count:,} vectors in the HNSW index ({seconds:.1f} s)")

    # 2. A deliberately weak graph next to the default one: fewer links, less effort while building
    weak = models.HnswConfigDiff(m=4, ef_construct=16)
    print(f"\nWeak graph (m=4, ef_construct=16) built in {practice_collection('tickets_weak', hnsw=weak):.1f} s")

    # 3. Recall@10 as the search looks harder (hnsw_ef)
    print("\nhnsw_ef   default graph   weak graph")
    for ef in (10, 16, 32, 64, 128):
        print(f"{ef:>7}   {recall_at_k('tickets', hnsw_ef=ef):>13.3f}   {recall_at_k('tickets_weak', hnsw_ef=ef):>10.3f}")
    client.delete_collection("tickets_weak")
