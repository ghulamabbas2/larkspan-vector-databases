"""Store all 5,000 Larkspan tickets in Qdrant: each one as a point with a vector and a payload.

    uv run python -m vector_databases.qdrant_store

We run Qdrant as a server (we started it with docker compose in setup). We make the
vectors with our embed() and send them to Qdrant. Every run starts fresh: the
collection is deleted and built again.
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
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)  # start fresh
client.create_collection(
    COLLECTION,
    vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
)

points = [
    models.PointStruct(
        id=str(uuid.uuid4()),  # a new random ID: Qdrant IDs must be a whole number or a UUID
        vector=vector.tolist(),
        payload={"ticket_id": ticket["id"], "subject": ticket["subject"], "body": ticket["body"],
                 **{field: ticket[field] for field in PAYLOAD_FIELDS}},
    )
    for ticket, vector in zip(tickets, vectors)
]
client.upload_points(COLLECTION, points, wait=True)  # sent in batches; returns once all are stored

print(f"Stored {client.count(COLLECTION).count:,} tickets in {time.perf_counter() - start:.1f} s")
print(f"Each vector has {len(vectors[0])} numbers. Look inside: http://localhost:6333/dashboard")
