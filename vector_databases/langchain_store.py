"""Our tickets in Qdrant through LangChain, then the same calls on another store.

    uv run python -m vector_databases.langchain_store

Run migrate_model.py first: we embed with the model of the collection behind tickets_live.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from vector_databases.embeddings import embed
from vector_databases.filter_search import client
from vector_databases.migrate_model import MODELS, live_collection
from vector_databases.tickets import load_tickets, ticket_text

COLLECTION = "tickets_langchain"

class LarkspanEmbeddings(Embeddings):
    """LangChain's interface for an embedding model, answered by our cached embed()."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return embed(texts, model=self.model).tolist()

    def embed_query(self, text):
        return embed([text], model=self.model)[0].tolist()


def to_document(ticket):
    """A ticket as a LangChain Document: the text to search, plus metadata."""
    return Document(page_content=ticket_text(ticket),
                    metadata={"ticket_id": ticket["id"], "plan": ticket["plan"], "priority": ticket["priority"]})


def show(store_name, results):
    print(f"\n{store_name}:")
    for document, score in results:
        subject = document.page_content.splitlines()[0]
        print(f"  {score:.3f}  {document.metadata['ticket_id']}  {document.metadata['plan']:<10}  {subject}")


if __name__ == "__main__":
    embeddings = LarkspanEmbeddings(MODELS[live_collection()])
    documents = [to_document(ticket) for ticket in load_tickets()]
    question = "we paid two times for the same month"
    print(f"Question: {question!r}")

    # 1. Qdrant through LangChain: we create the collection, LangChain stores and searches
    client.delete_collection(COLLECTION)  # start fresh on every run
    size = len(embeddings.embed_query("size check"))
    vectors = models.VectorParams(size=size, distance=models.Distance.COSINE)
    client.create_collection(COLLECTION, vectors_config=vectors)

    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    store.add_documents(documents)
    show("Qdrant", store.similarity_search_with_score(question, k=3))


    # 2. Another store: change the line that creates it, keep the rest
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(documents)
    show("In-memory store", store.similarity_search_with_score(question, k=3))