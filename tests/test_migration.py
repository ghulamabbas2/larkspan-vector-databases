import importlib
import py_compile
import uuid

import numpy as np
import pytest
import qdrant_client
from qdrant_client import QdrantClient, models

import settings
from vector_databases import embeddings
from vector_databases.tickets import load_tickets, ticket_text

SCRIPTS = ["migrate_model", "langchain_store"]
calls = []  # (model, number of texts) for every fake embed call


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_compile(name):
    py_compile.compile(str(settings.ROOT / "vector_databases" / f"{name}.py"), doraise=True)


def fake_embed(texts, provider=None, model=None):
    """Small vectors made from the text; the new model gives 4 numbers, the old one 3."""
    calls.append((model, len(texts)))
    size = 4 if model == "text-embedding-3-large" else 3
    return np.array([[len(text) % 7 + 1.0, text.count(" ") + 1.0] + [1.0] * (size - 2) for text in texts],
                    dtype=np.float32)


@pytest.fixture
def migrate(monkeypatch):
    """20 tickets in an in-memory Qdrant `tickets` collection, and migrate_model wired to it."""
    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda url=None, **kwargs: client)
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    calls.clear()
    client.create_collection("tickets", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    client.upsert("tickets", [
        models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, t["id"])), vector=fake_embed([ticket_text(t)])[0].tolist(),
                           payload={"ticket_id": t["id"], "text": ticket_text(t)})
        for t in load_tickets()[:20]])
    from vector_databases import filter_search, hybrid_search, migrate_model, recall_test, score_search
    for module in (recall_test, filter_search, hybrid_search, score_search):
        importlib.reload(module)
    return importlib.reload(migrate_model), client


def test_the_alias_moves_in_one_step(migrate):
    m, client = migrate
    assert m.live_collection() is None
    m.point_alias_at("tickets")
    assert m.live_collection() == "tickets"
    m.create_new_collection()
    m.point_alias_at("tickets_large")
    assert m.live_collection() == "tickets_large"
    assert len(client.get_aliases().aliases) == 1


def test_the_new_collection_is_sized_for_the_new_model(migrate):
    m, client = migrate
    assert m.create_new_collection() is True and m.create_new_collection() is False  # the second run skips it
    assert client.get_collection("tickets_large").config.params.vectors.size == 4


def test_backfill_copies_every_point_and_keeps_newer_writes(migrate):
    m, client = migrate
    m.create_new_collection()
    first = str(uuid.uuid5(uuid.NAMESPACE_URL, load_tickets()[0]["id"]))
    client.upsert("tickets_large", [models.PointStruct(id=first, vector=[1.0, 1.0, 1.0, 1.0], payload={"text": "newer"})])
    assert m.backfill(batch_size=7) == 20
    assert client.count("tickets_large").count == 20
    assert client.retrieve("tickets_large", [first])[0].payload == {"text": "newer"}  # insert-only left it alone


def test_dual_write_saves_the_ticket_in_both_with_each_model(migrate):
    m, client = migrate
    m.create_new_collection()
    calls.clear()
    m.save_ticket(load_tickets()[0])
    assert [model for model, _ in calls] == [settings.OPENAI_EMBED_MODEL, "text-embedding-3-large"]
    assert client.count("tickets_large").count == 1 and client.count("tickets").count == 20


def test_search_through_the_alias_uses_the_model_of_the_collection_behind_it(migrate):
    m, _ = migrate
    m.create_new_collection()
    m.backfill()
    for collection, model in (("tickets", settings.OPENAI_EMBED_MODEL), ("tickets_large", "text-embedding-3-large")):
        m.point_alias_at(collection)
        calls.clear()
        assert m.vector_search("billing question", limit=3)
        assert calls == [(model, 1)]


def test_langchain_wrapper_works_with_a_langchain_store(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    from langchain_core.vectorstores import InMemoryVectorStore

    from vector_databases import langchain_store
    store = InMemoryVectorStore(embedding=langchain_store.LarkspanEmbeddings("text-embedding-3-large"))
    tickets = load_tickets()[:10]
    store.add_documents([langchain_store.to_document(t) for t in tickets], ids=[t["id"] for t in tickets])
    results = store.similarity_search("billing", k=3)
    assert len(results) == 3 and all("ticket_id" in d.metadata for d in results)
    assert len(store.embeddings.embed_query("x")) == 4  # the model name reached embed()
