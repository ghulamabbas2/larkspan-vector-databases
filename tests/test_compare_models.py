import json
import py_compile

import numpy as np

import settings


def test_compare_models_compiles():
    py_compile.compile(str(settings.ROOT / "vector_databases" / "compare_models.py"), doraise=True)


def test_every_issue_has_a_search_query():
    queries = json.loads((settings.ROOT / "eval" / "search_queries.json").read_text())
    topics = json.loads((settings.ROOT / "eval" / "ticket_topics.json").read_text())["topics"]
    assert sorted(query["topic"] for query in queries) == sorted(topics)


def test_shorten_keeps_the_first_numbers_at_length_1():
    source = (settings.ROOT / "vector_databases" / "compare_models.py").read_text()
    namespace = {"np": np}
    exec(source[source.index("def shorten"):source.index("def hits")], namespace)   # the function only, no API call
    short = namespace["shorten"](np.array([[3.0, 4.0, 12.0]]), 2)
    assert np.allclose(short, [[0.6, 0.8]])
