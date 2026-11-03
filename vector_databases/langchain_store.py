"""Our tickets in Qdrant through LangChain's vector store interface, then the same calls on another store.

    uv run python -m vector_databases.langchain_store

Run migrate_model.py first. Embeddings come from our cached embed(), wrapped in LangChain's
Embeddings interface, with the model the live collection uses. Safe to run again.
"""

import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from vector_databases.embeddings import embed
from vector_databases.filter_search import client
from vector_databases.migrate_model import ALIAS, MODELS, live_collection
from vector_databases.tickets import load_tickets, ticket_text

COLLECTION = "tickets_langchain"


class LarkspanEmbeddings(Embeddings):
    """LangChain's interface for an embedding model, answered by our embed() and its cache."""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts, model=self.model).tolist()

    def embed_query(self, text: str) -> list[float]:
        return embed([text], model=self.model)[0].tolist()


def to_document(ticket: dict) -> Document:
    """A ticket as a LangChain Document: the text to search, plus metadata to show and filter on."""
    return Document(page_content=ticket_text(ticket),
                    metadata={"ticket_id": ticket["id"], "plan": ticket["plan"], "priority": ticket["priority"]})


def show(results: list) -> None:
    for document, score in results:
        print(f"  {score:.3f}  {document.metadata.get('ticket_id', '?')}  {document.metadata.get('plan', '?'):<10}  "
              f"{document.page_content.splitlines()[0]}")


if __name__ == "__main__":
    embeddings = LarkspanEmbeddings(MODELS[live_collection()])
    tickets = load_tickets()
    documents = [to_document(ticket) for ticket in tickets]
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, ticket["id"])) for ticket in tickets]  # same ticket, same ID

    # 1. A Qdrant collection in LangChain's layout, set up the way LangChain's docs show
    if not client.collection_exists(COLLECTION):
        size = len(embeddings.embed_query("sample text"))
        client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE))
    qdrant = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    qdrant.add_documents(documents, ids=ids)
    print(f"{COLLECTION}: {client.count(COLLECTION).count:,} points")

    question = "we paid two times for the same month"
    print(f"\nQdrant through LangChain: {question!r}")
    show(qdrant.similarity_search_with_score(question, k=3))
    starter = models.Filter(must=[models.FieldCondition(key="metadata.plan", match=models.MatchValue(value="Starter"))])
    print("Only Starter tickets:")
    show(qdrant.similarity_search_with_score(question, k=3, filter=starter))

    # 2. The same calls on another vector store: LangChain's in-memory one
    memory = InMemoryVectorStore(embedding=embeddings)
    memory.add_documents(documents, ids=ids)
    print("\nIn-memory store, same question:")
    show(memory.similarity_search_with_score(question, k=3))
    print("Only Starter tickets (this store takes a Python function as its filter):")
    show(memory.similarity_search_with_score(question, k=3, filter=lambda doc: doc.metadata["plan"] == "Starter"))
    same = {d.metadata["ticket_id"] for d in qdrant.similarity_search(question, k=10)} & \
           {d.metadata["ticket_id"] for d in memory.similarity_search(question, k=10)}
    print(f"Tickets both stores put in their top 10: {len(same)} of 10")

    # 3. Our own collection, which wasn't made by LangChain: the text is found, the metadata isn't
    ours = QdrantVectorStore(client=client, collection_name=ALIAS, embedding=embeddings, content_payload_key="text")
    document, score = ours.similarity_search_with_score(question, k=1)[0]
    print(f"\nOur {ALIAS} collection through LangChain: {document.page_content.splitlines()[0]!r}")
    print(f"  its metadata: {document.metadata}")
