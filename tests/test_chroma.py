import py_compile
import runpy

import chromadb
import numpy as np
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

import settings
from vector_databases import embeddings
from vector_databases.chroma_embedding import LarkspanEmbedding

SCRIPTS = ["chroma_hello", "chroma_embedding", "chroma_store", "search_chroma"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_compile(name):
    py_compile.compile(str(settings.ROOT / "vector_databases" / f"{name}.py"), doraise=True)


def fake_embed(texts, provider=None):
    """Small length-1 vectors made from the text, so tests need no model and no API."""
    vectors = np.array([[len(text) % 7 + 1.0, text.count(" ") + 1.0, 1.0] for text in texts], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_store_fills_the_collection_and_starts_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    for _ in range(2):   # the second run must not add the tickets twice
        SharedSystemClient.clear_system_cache()   # each real run is a new process
        runpy.run_module("vector_databases.chroma_store", run_name="__main__")
    collection = chromadb.PersistentClient(path=tmp_path / "chroma_data").get_collection("tickets")
    assert collection.count() == 5000
    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert isinstance(collection.configuration["embedding_function"], LarkspanEmbedding)   # saved and rebuilt
    first = collection.get(ids=["TCK-10001"])["metadatas"][0]
    assert first == {"plan": "Growth", "product_area": "Helpdesk & SLA", "priority": "high",
                     "status": "solved", "channel": "chat", "customer_id": "cus_043"}
    assert len(list((tmp_path / "chroma_data").iterdir())) == 2   # chroma.sqlite3 + one index folder


@pytest.fixture
def small_collection(monkeypatch):
    """Three tickets with hand-made vectors: the closest to [1, 0] is A, then B, then C.
    Every question becomes [1, 0], so the order is known."""
    monkeypatch.setattr(embeddings, "embed", lambda texts, provider=None: np.array([[1.0, 0.0]] * len(texts)))
    collection = chromadb.EphemeralClient().get_or_create_collection(
        "test-search", embedding_function=LarkspanEmbedding(), configuration={"hnsw": {"space": "cosine"}})
    collection.add(
        ids=["A", "B", "C"],
        embeddings=[[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
        documents=["Login fails\nbody", "Charged twice\nbody", "Export broken\nbody"],
        metadatas=[{"priority": "urgent", "plan": "Enterprise", "status": "open"},
                   {"priority": "high", "plan": "Starter", "status": "pending"},
                   {"priority": "urgent", "plan": "Growth", "status": "closed"}],
    )
    yield collection
    chromadb.EphemeralClient().delete_collection("test-search")


def search_function(collection):
    """search() from search_chroma.py, run against a test collection."""
    source = (settings.ROOT / "vector_databases" / "search_chroma.py").read_text()
    namespace = {"collection": collection}
    exec(source[source.index("def search"):source.index('search("')], namespace)   # the function only
    return namespace["search"]


def test_search_prints_best_first_with_similarity(small_collection, capsys):
    search_function(small_collection)("anything", k=3)
    lines = capsys.readouterr().out.strip().splitlines()[1:]
    assert [line.split()[0] for line in lines] == ["1.", "2.", "3."]
    assert lines[0].split()[1] == "1.00" and lines[0].endswith("Login fails")   # same direction: similarity 1
    assert lines[1].split()[1] == "0.80"                                       # cosine of [1,0] and [0.8,0.6]
    assert lines[2].split()[1] == "0.00"                                       # right angle: similarity 0


def test_search_filters(small_collection, capsys):
    search = search_function(small_collection)
    search("anything", where={"priority": "urgent"})
    assert "Charged twice" not in capsys.readouterr().out
    search("anything", where={"$and": [{"priority": "urgent"}, {"status": {"$in": ["open", "pending"]}}]})
    out = capsys.readouterr().out
    assert "Login fails" in out and "Export broken" not in out


# What Chroma does, as the lectures describe it. If an upgrade changes any of these, the slides need updating.
def test_two_fields_in_one_filter_need_and(small_collection):
    with pytest.raises(ValueError, match="Expected where to have exactly one operator"):
        small_collection.query(query_embeddings=[[1.0, 0.0]], where={"priority": "urgent", "plan": "Growth"})


def test_cosine_distance_is_1_minus_similarity(small_collection):
    results = small_collection.query(query_embeddings=[[1.0, 0.0]], n_results=3)
    assert np.allclose(results["distances"][0], [0.0, 0.2, 1.0], atol=1e-5)


def test_default_distance_is_l2():
    collection = chromadb.EphemeralClient().get_or_create_collection("test-default", embedding_function=None)
    assert collection.configuration["hnsw"]["space"] == "l2"


def test_add_keeps_the_old_record_for_an_existing_id(small_collection):
    small_collection.add(ids=["A"], embeddings=[[0.0, 1.0]], documents=["new text"])
    assert small_collection.get(ids=["A"])["documents"] == ["Login fails\nbody"]


def test_reopened_collection_embeds_questions_with_our_model(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    client = chromadb.PersistentClient(path=tmp_path)
    client.create_collection("test-reopen", embedding_function=LarkspanEmbedding("local"))
    SharedSystemClient.clear_system_cache()
    reopened = chromadb.PersistentClient(path=tmp_path).get_collection("test-reopen")
    saved = reopened.configuration["embedding_function"]
    assert isinstance(saved, LarkspanEmbedding) and saved.provider == "local"
    reopened.add(ids=["a"], documents=["Charged twice"])
    assert np.allclose(reopened.get(ids=["a"], include=["embeddings"])["embeddings"][0], fake_embed(["Charged twice"])[0])
