import importlib
import json
import py_compile
import runpy

import numpy as np
import pytest
import qdrant_client
from qdrant_client import QdrantClient

import settings
from vector_databases import embeddings

SCRIPTS = ["filter_search", "tenant_search", "hybrid_search", "score_search"]
GOLDEN = json.loads((settings.ROOT / "eval" / "golden_set.json").read_text())


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_compile(name):
    py_compile.compile(str(settings.ROOT / "vector_databases" / f"{name}.py"), doraise=True)


def fake_embed(texts, provider=None):
    """Small vectors made from the text, so tests need no model and no API."""
    return np.array([[len(text) % 7 + 1.0, text.count(" ") + 1.0, 1.0] for text in texts], dtype=np.float32)


@pytest.fixture
def tenant_search(monkeypatch):
    """The tickets in Qdrant's in-memory mode, and tenant_search wired to it."""
    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda url=None, **kwargs: client)
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    runpy.run_module("vector_databases.ingest_tickets", run_name="__main__")
    from vector_databases import filter_search, recall_test, tenant_search
    importlib.reload(recall_test), importlib.reload(filter_search)
    return importlib.reload(tenant_search)


def test_search_for_customer_only_returns_that_customers_tickets(tenant_search):
    for customer in ("cus_026", "cus_033"):
        results = tenant_search.search_for_customer(customer, "every webhook delivery fails", limit=50)
        assert results and {point.payload["customer_id"] for point in results} == {customer}


def test_only_builds_one_condition_per_field():
    from vector_databases.filter_search import only
    conditions = only(plan="Starter", priority="urgent").must
    assert [(c.key, c.match.value) for c in conditions] == [("plan", "Starter"), ("priority", "urgent")]


def test_precision_and_reciprocal_rank():
    from vector_databases.score_search import precision_at_k, reciprocal_rank
    found = ["a", "x", "b", "y", "z", "q", "w", "e", "r", "t"]
    assert precision_at_k(found, {"a", "b"}) == pytest.approx(0.2)
    assert reciprocal_rank(found, {"b"}) == pytest.approx(1 / 3)  # first right result in third place
    assert reciprocal_rank(found, {"nothing"}) == 0.0


def test_golden_set_matches_the_tickets():
    tickets = {json.loads(line)["id"]: json.loads(line) for line in (settings.DATA_DIR / "tickets.jsonl").open()}
    assert len(GOLDEN) == 63 and {q["kind"] for q in GOLDEN} == {"everyday", "error code"}
    for item in GOLDEN:
        assert item["right"] and all(ticket in tickets for ticket in item["right"])
        if item["kind"] == "error code":
            code = item["question"].split()[-1]
            assert all(code in tickets[t]["subject"] + " " + tickets[t]["body"] for t in item["right"])
