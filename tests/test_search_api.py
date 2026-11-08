import py_compile
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import settings
from vector_databases import search_api

TICKET = dict(ticket_id="TCK-10001", subject="Export fails", body="ERR-5031 when I export.", customer_id="cus_004",
              plan="Enterprise", priority="normal", product_area="Reports & Dashboards", status="closed")


def test_script_compiles():
    py_compile.compile(str(settings.ROOT / "vector_databases" / "search_api.py"), doraise=True)


@pytest.fixture
def api(monkeypatch):
    """The API with a fake search, so no Qdrant or OpenAI is needed. Returns (client, calls)."""
    calls = []

    def fake_search(question, limit=10, query_filter=None):
        calls.append((question, limit, query_filter))
        return [search_api.Ticket(**TICKET, score=0.5)]

    monkeypatch.setattr(search_api, "search_tickets", fake_search)
    return TestClient(search_api.app), calls


def test_search_returns_tickets(api):
    client, calls = api
    response = client.get("/search", params={"q": "ERR-5031"})
    assert response.status_code == 200
    assert response.json() == [{**TICKET, "score": 0.5}]
    assert calls == [("ERR-5031", 10, None)]  # no filter when none is asked for


@pytest.mark.parametrize("params", [{"q": ""}, {}, {"q": "refund", "limit": 0}, {"q": "refund", "limit": 51},
                                    {"q": "refund", "plan": "Basic"}, {"q": "refund", "priority": "critical"},
                                    {"q": "x" * 501}])
def test_bad_requests_are_rejected_before_searching(api, params):
    client, calls = api
    assert client.get("/search", params=params).status_code == 422
    assert calls == []


def test_filters_become_one_qdrant_filter(api):
    client, calls = api
    client.get("/search", params={"q": "refund", "plan": "Starter", "customer_id": "cus_026", "limit": 3})
    question, limit, query_filter = calls[0]
    assert (question, limit) == ("refund", 3)
    assert {(c.key, c.match.value) for c in query_filter.must} == {("plan", "Starter"), ("customer_id", "cus_026")}


def test_the_filter_goes_into_both_searches(monkeypatch):
    """search_tickets puts the filter on the vector prefetch and on the BM25 prefetch."""
    seen = {}

    def fake_query_points(collection, **kwargs):
        seen.update(kwargs, collection=collection)
        return SimpleNamespace(points=[SimpleNamespace(payload=TICKET, score=0.5)])

    monkeypatch.setattr(search_api, "client", SimpleNamespace(query_points=fake_query_points))
    monkeypatch.setattr(search_api, "live_collection", lambda: "tickets_large")
    monkeypatch.setattr(search_api, "embed", lambda texts, model=None: np.ones((len(texts), 4), dtype=np.float32))
    query_filter = search_api.only(plan="Starter")
    results = search_api.search_tickets("refund", 5, query_filter)
    assert seen["collection"] == "tickets_live" and seen["limit"] == 5
    assert [prefetch.filter for prefetch in seen["prefetch"]] == [query_filter, query_filter]
    assert results[0].ticket_id == "TCK-10001"


def test_the_search_page_is_served_but_not_listed(api):
    client, calls = api
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert 'fetch("/search?"' in page.text and ".innerHTML =" not in page.text  # results go in as text
    assert "/" not in client.get("/openapi.json").json()["paths"]  # a page for people, not an endpoint on /docs
    assert calls == []


@pytest.mark.live
def test_the_real_api_finds_err_5031():
    """Needs Qdrant with tickets_live (production section) and an OpenAI key or a cached vector."""
    results = TestClient(search_api.app).get("/search", params={"q": "ERR-5031"}).json()
    assert sum("ERR-5031" in t["subject"] + t["body"] for t in results) == 10
