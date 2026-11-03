import importlib
import py_compile
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import qdrant_client
from qdrant_client import QdrantClient, models

import settings
from vector_databases import embeddings

SCRIPTS = ["recall_test", "tune_hnsw", "quantize", "benchmark"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_compile(name):
    py_compile.compile(str(settings.ROOT / "vector_databases" / f"{name}.py"), doraise=True)


def fake_embed(texts, provider=None):
    """Small vectors made from the text, so tests need no model and no API."""
    return np.array([[len(text) % 7 + 1.0, text.count(" ") + 1.0, 1.0] for text in texts], dtype=np.float32)


@pytest.fixture
def qdrant(monkeypatch):
    """Qdrant's in-memory mode (always exact search) instead of the server, and a fresh recall_test."""
    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda url=None, **kwargs: client)
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    from vector_databases import recall_test
    return client, importlib.reload(recall_test)


def collection(client, name, vectors):
    client.create_collection(name, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    client.upsert(name, [models.PointStruct(id=i, vector=v) for i, v in enumerate(vectors)])


def test_recall_is_one_when_both_searches_are_exact(qdrant, capsys):
    client, recall_test = qdrant
    runpy.run_module("vector_databases.ingest_tickets", run_name="__main__")
    assert recall_test.recall_at_k("tickets") == 1.0
    runpy.run_path(str(settings.ROOT / "vector_databases" / "recall_test.py"), run_name="__main__")
    assert "recall@10 over 45 questions: 1.000" in capsys.readouterr().out


def test_ground_truth_keeps_every_ticket_that_ties_for_last_place(qdrant):
    client, recall_test = qdrant
    vectors = [[1.0, float(i), 0.0] for i in range(9)] + [[1.0, 20.0, 0.0]] * 2 + [[0.0, 0.0, 1.0]]
    collection(client, "ties", vectors)
    truth = recall_test.ground_truth("ties", np.array([1.0, 0.0, 0.0]), k=10)
    assert truth == set(range(11))  # 9 clear winners, then two identical vectors sharing 10th place


def test_recall_counts_what_the_fast_search_missed(qdrant, monkeypatch):
    client, recall_test = qdrant
    collection(client, "small", [[1.0, float(i), 0.0] for i in range(30)])  # scores fall with i: no ties
    monkeypatch.setattr(recall_test, "question_vectors", np.array([[1.0, 0.0, 0.0]]))
    exact_search = client.query_points

    def drops_the_best(name, query, limit, search_params):  # a "fast" search that always misses one
        points = exact_search(name, query=query, limit=limit + 1).points
        return SimpleNamespace(points=points if search_params.exact else points[1:])

    monkeypatch.setattr(recall_test.client, "query_points", drops_the_best)
    assert recall_test.recall_at_k("small") == pytest.approx(0.9)


def test_practice_collection_is_always_indexed(qdrant, monkeypatch):
    client, _ = qdrant
    from vector_databases import tune_hnsw
    tune_hnsw = importlib.reload(tune_hnsw)
    monkeypatch.setattr(tune_hnsw, "wait_until_indexed", lambda name: 0.0)  # local mode never builds a graph
    created = {}
    create = client.create_collection
    monkeypatch.setattr(client, "create_collection", lambda name, **kw: created.update(kw) or create(name, **kw))
    tune_hnsw.practice_collection("weak", hnsw=models.HnswConfigDiff(m=4, ef_construct=16), memory=models.Memory.COLD)
    assert created["hnsw_config"] == models.HnswConfigDiff(m=4, ef_construct=16, full_scan_threshold=10)
    assert created["optimizers_config"].indexing_threshold == 10
    assert created["vectors_config"].memory == models.Memory.COLD
    assert client.count("weak").count == 5000
