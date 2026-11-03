import py_compile
import subprocess
import sys

import numpy as np
import pytest

import settings
from vector_databases import embeddings

SCRIPTS = ["first_embedding", "tickets", "embeddings", "embed_tickets"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_compile(name):
    py_compile.compile(str(settings.ROOT / "vector_databases" / f"{name}.py"), doraise=True)


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    """embed() with a temporary cache and a fake model that records each batch it gets."""
    monkeypatch.setattr(embeddings, "CACHE_FILE", tmp_path / "embeddings.sqlite")
    monkeypatch.setattr(embeddings, "BATCH_SIZE", 2)
    batches = []

    def fake_call(texts, provider, model):
        batches.append(texts)
        return [[float(len(text)), 1.0] for text in texts]

    monkeypatch.setattr(embeddings, "call_model", fake_call)
    return batches


def test_embed_batches_keeps_order_and_skips_repeats(fake_model):
    vectors = embeddings.embed(["a", "bbb", "a", "cc", "dddd"], "openai")
    assert [len(batch) for batch in fake_model] == [2, 2]       # 4 different texts, 2 per batch
    assert vectors[:, 0].tolist() == [1, 3, 1, 2, 4]           # one row per text, same order


def test_embed_uses_the_cache_the_second_time(fake_model):
    first = embeddings.embed(["a", "bbb"], "openai")
    second = embeddings.embed(["bbb", "a"], "openai")
    assert len(fake_model) == 1                                 # no new call
    assert np.array_equal(second, first[::-1])


def test_cache_keeps_models_apart(fake_model):
    embeddings.embed(["a"], "openai")
    embeddings.embed(["a"], "local")
    assert len(fake_model) == 2


@pytest.mark.live
def test_first_embedding_runs():
    result = subprocess.run([sys.executable, "-m", "vector_databases.first_embedding"],
                            cwd=settings.ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "Dimensions: 1536" in result.stdout
