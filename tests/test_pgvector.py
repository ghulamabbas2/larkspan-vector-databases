import importlib
import py_compile
import runpy

import numpy as np
import psycopg
import pytest

import settings
from vector_databases import embeddings

SCRIPTS = ["pg_store", "pg_search", "pg_indexes", "compare_databases"]
TEST_SCHEMA = "larkspan_test"  # the tests' own copy of the table, so the real `tickets` is never touched


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_compile(name):
    py_compile.compile(str(settings.ROOT / "vector_databases" / f"{name}.py"), doraise=True)


def fake_embed(texts, provider=None, model=None):
    """Small vectors made from the text, so tests need no model and no API."""
    return np.array([[len(text) % 7 + 1.0, text.count(" ") + 1.0, 1.0] for text in texts], dtype=np.float32)


@pytest.fixture
def postgres(monkeypatch):
    """Our Postgres, with every script working inside a separate schema that is dropped afterwards."""
    try:
        admin = psycopg.connect(settings.DATABASE_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("The course's Postgres isn't reachable (docker compose up -d)")
    admin.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
    admin.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    sep = "&" if "?" in settings.DATABASE_URL else "?"
    monkeypatch.setattr(settings, "DATABASE_URL", f"{settings.DATABASE_URL}{sep}options=-csearch_path%3D{TEST_SCHEMA},public")
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    yield
    admin.execute(f"DROP SCHEMA {TEST_SCHEMA} CASCADE")
    admin.close()


def test_store_can_run_again_and_again(postgres, capsys):
    for _ in range(2):
        runpy.run_module("vector_databases.pg_store", run_name="__main__")
    assert capsys.readouterr().out.count("The table now holds 5,000 rows") == 2


def test_the_table_is_in_the_test_schema(postgres):
    runpy.run_module("vector_databases.pg_store", run_name="__main__")
    from vector_databases.pg_store import connect
    schema = connect().execute("SELECT table_schema FROM information_schema.tables WHERE table_name = 'tickets' "
                               "AND table_schema = %s", (TEST_SCHEMA,)).fetchone()
    assert schema == (TEST_SCHEMA,)


def test_search_returns_the_closest_first(postgres):
    runpy.run_module("vector_databases.pg_store", run_name="__main__")
    from vector_databases import pg_search
    results = importlib.reload(pg_search).search("we paid two times", limit=5)  # reload: connect with the test settings
    similarities = [similarity for _, _, similarity in results]
    assert len(results) == 5 and similarities == sorted(similarities, reverse=True)
    assert all(-1.0 <= s <= 1.0 + 1e-6 for s in similarities)  # cosine similarity = 1 - cosine distance
