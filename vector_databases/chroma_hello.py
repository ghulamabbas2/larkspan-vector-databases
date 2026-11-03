"""Chroma in a few lines: store four sentences, then search them by meaning.

    uv run python -m vector_databases.chroma_hello

Chroma turns the sentences into vectors itself, with its built-in model
(downloaded the first time, about 80 MB).
"""

import chromadb

client = chromadb.EphemeralClient()  # kept in memory: gone when the script ends
collection = client.create_collection("hello", configuration={"hnsw": {"space": "cosine"}})

collection.add(
    ids=["1", "2", "3", "4"],
    documents=[
        "The cat sleeps on the sofa",
        "A puppy plays in the garden",
        "I ate a banana for lunch",
        "The bus was late this morning",
    ],
)

results = collection.query(query_texts=["my kitten is napping"], n_results=4)
for text, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"{distance:.3f}  {text}")
