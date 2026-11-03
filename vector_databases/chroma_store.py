"""Store all 5,000 Larkspan tickets in Chroma: the text, the vector and the metadata.

    uv run python -m vector_databases.chroma_store

We give Chroma the text and our embedding function. Chroma calls it to make the
vectors (they come from the cache if we made them before). The database is saved in
the chroma_data/ folder. Every run starts fresh: the folder is deleted and built again.
"""

import shutil
import time

import chromadb

import settings
from vector_databases.chroma_embedding import LarkspanEmbedding
from vector_databases.tickets import load_tickets, ticket_text

CHROMA_PATH = settings.ROOT / "chroma_data"
METADATA_FIELDS = ["plan", "product_area", "priority", "status", "channel", "customer_id"]
BATCH_SIZE = 1000  # tickets per add() call. Chroma limits how many records one call can take.

shutil.rmtree(CHROMA_PATH, ignore_errors=True)  # start fresh: delete the old database folder
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.create_collection(
    "tickets",
    embedding_function=LarkspanEmbedding(),  # Chroma uses it for the tickets and for every question
    configuration={"hnsw": {"space": "cosine"}},
)

start = time.perf_counter()
tickets = load_tickets()

for first in range(0, len(tickets), BATCH_SIZE):
    batch = tickets[first:first + BATCH_SIZE]
    collection.add(
        ids=[ticket["id"] for ticket in batch],
        documents=[ticket_text(ticket) for ticket in batch],  # Chroma embeds these for us
        metadatas=[{field: ticket[field] for field in METADATA_FIELDS} for ticket in batch],
    )

vector = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
print(f"Stored {collection.count():,} tickets in {time.perf_counter() - start:.1f} s")
print(f"Each vector has {len(vector)} numbers. Saved in the chroma_data/ folder.")
