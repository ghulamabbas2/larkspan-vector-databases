"""Search the tickets in Chroma by meaning: the top k, how close each one is, and filters.

    uv run python -m vector_databases.search_chroma

Run vector_databases.chroma_store first, so the tickets are in Chroma.
"""

import chromadb

import settings
import vector_databases.chroma_embedding  # Chroma needs the class to rebuild our embedding function

client = chromadb.PersistentClient(path=settings.ROOT / "chroma_data")
collection = client.get_collection("tickets")  # comes back with the embedding function it was made with


def search(question: str, k: int = 3, where: dict | None = None) -> None:
    results = collection.query(query_texts=[question], n_results=k, where=where)  # Chroma embeds the question

    print(f'\n"{question}"   filter: {where}')
    for rank in range(len(results["ids"][0])):
        score = 1 - results["distances"][0][rank]  # cosine distance → cosine similarity
        meta = results["metadatas"][0][rank]
        subject = results["documents"][0][rank].split("\n")[0]
        print(f"{rank + 1}. {score:.2f}  {meta['priority']:<7} {meta['plan']:<10} {meta['status']:<8} {subject}")


search("I can't log in to my account", k=5)
search("I was charged twice this month", where={"priority": "urgent"})
search("webhook keeps failing",
       where={"$and": [{"plan": "Enterprise"}, {"status": {"$in": ["open", "pending"]}}]})
