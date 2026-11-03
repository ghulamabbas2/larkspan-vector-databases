"""Load the Larkspan tickets into Qdrant. Safe to run as often as you like.

    uv run python -m vector_databases.ingest_tickets

The same steps as qdrant_store.py, with two changes so a second run never stores a
ticket twice: the collection is only created if it's missing (never deleted), and
each point's ID is made from the ticket's own ID, so it's the same on every run.
"""

import time
import uuid

from qdrant_client import QdrantClient, models

import settings
from vector_databases.embeddings import embed
from vector_databases.tickets import load_tickets, ticket_text

COLLECTION = "tickets"
PAYLOAD_FIELDS = ["plan", "product_area", "priority", "status", "channel", "customer_id"]

start = time.perf_counter()
tickets = load_tickets()
vectors = embed([ticket_text(ticket) for ticket in tickets])  # from the cache if we made them before

client = QdrantClient(url=settings.QDRANT_URL)
if not client.collection_exists(COLLECTION):  # create it once; later runs reuse it
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )

points = [
    models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, ticket["id"])),  # same ticket → same ID, on every run
        vector=vector.tolist(),
        payload={"ticket_id": ticket["id"], "subject": ticket["subject"], "body": ticket["body"],
                 **{field: ticket[field] for field in PAYLOAD_FIELDS}},
    )
    for ticket, vector in zip(tickets, vectors)
]
client.upload_points(COLLECTION, points, wait=True)  # an existing ID is replaced, a new one is added

print(f"Loaded {len(points):,} tickets in {time.perf_counter() - start:.1f} s")
print(f"The collection now holds {client.count(COLLECTION).count:,} points")
