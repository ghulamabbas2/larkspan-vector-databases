import py_compile
import runpy
import uuid

import numpy as np
import pytest
import qdrant_client
from qdrant_client import QdrantClient, models

import settings
from vector_databases import embeddings, tickets

SCRIPTS = ["qdrant_store", "search_qdrant", "ingest_tickets"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_compile(name):
    py_compile.compile(str(settings.ROOT / "vector_databases" / f"{name}.py"), doraise=True)


def fake_embed(texts, provider=None):
    """Small vectors made from the text, so tests need no model and no API."""
    return np.array([[len(text) % 7 + 1.0, text.count(" ") + 1.0, 1.0] for text in texts], dtype=np.float32)


@pytest.fixture
def qdrant(monkeypatch):
    """Qdrant's in-memory mode instead of the server; every script in the test shares it."""
    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda url=None, **kwargs: client)
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    return client


def run(name):
    runpy.run_module(f"vector_databases.{name}", run_name="__main__")


def payload_of(client, ticket_id):
    hits = client.scroll("tickets", scroll_filter=models.Filter(must=[
        models.FieldCondition(key="ticket_id", match=models.MatchValue(value=ticket_id))]), limit=10)[0]
    return [hit.payload for hit in hits]


def test_store_fills_the_collection_and_starts_fresh(qdrant):
    run("qdrant_store")
    run("qdrant_store")   # starts fresh, so still 5,000
    assert qdrant.count("tickets").count == 5000
    assert qdrant.get_collection("tickets").config.params.vectors.distance == models.Distance.COSINE
    assert payload_of(qdrant, "TCK-10001") == [{
        "ticket_id": "TCK-10001", "subject": "How to create canned replies?",
        "body": "How do I share saved replies with the whole team?", "plan": "Growth",
        "product_area": "Helpdesk & SLA", "priority": "high", "status": "solved", "channel": "chat",
        "customer_id": "cus_043"}]


def test_ingest_can_run_again_and_again(qdrant):
    for _ in range(3):
        run("ingest_tickets")
    assert qdrant.count("tickets").count == 5000


def test_ingest_after_the_store_script_keeps_both_sets_of_ids(qdrant):
    # What lecture 4 shows: the random IDs from qdrant_store don't match the stable ones
    run("qdrant_store")
    run("ingest_tickets")
    assert qdrant.count("tickets").count == 10000
    assert len(payload_of(qdrant, "TCK-10001")) == 2


def test_a_ticket_always_gets_the_same_id(qdrant):
    run("ingest_tickets")
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "TCK-10001"))
    assert expected == "6b9d8b84-658a-5c48-bc5b-4e59ef183185"
    assert qdrant.retrieve("tickets", ids=[expected])[0].payload["ticket_id"] == "TCK-10001"


def test_an_edited_ticket_is_replaced_not_added(qdrant, monkeypatch):
    run("ingest_tickets")
    edited = tickets.load_tickets()
    edited[0] = {**edited[0], "status": "open", "subject": "Canned replies for the whole team"}
    monkeypatch.setattr(tickets, "load_tickets", lambda: edited)
    run("ingest_tickets")
    assert qdrant.count("tickets").count == 5000
    [payload] = payload_of(qdrant, "TCK-10001")
    assert payload["status"] == "open" and payload["subject"] == "Canned replies for the whole team"
    assert payload["plan"] == "Growth"   # the full payload is sent again, so nothing is lost


def test_search_prints_best_first_with_the_score(qdrant, monkeypatch, capsys):
    """Three points with hand-made vectors; every question becomes [1, 0], so the order is known."""
    monkeypatch.setattr(embeddings, "embed", lambda texts, provider=None: np.array([[1.0, 0.0]] * len(texts)))
    qdrant.create_collection("tickets", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    qdrant.upsert("tickets", points=[
        models.PointStruct(id=1, vector=[1.0, 0.0], payload={"ticket_id": "A", "priority": "urgent", "subject": "Login fails"}),
        models.PointStruct(id=2, vector=[0.8, 0.6], payload={"ticket_id": "B", "priority": "high", "subject": "Charged twice"}),
        models.PointStruct(id=3, vector=[0.0, 1.0], payload={"ticket_id": "C", "priority": "low", "subject": "Export broken"}),
    ])
    run("search_qdrant")
    first_search = capsys.readouterr().out.strip().split("\n\n")[0].splitlines()[1:]
    assert [line.split()[:3] for line in first_search] == [["1.", "1.00", "A"], ["2.", "0.80", "B"], ["3.", "0.00", "C"]]
