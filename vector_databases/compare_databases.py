"""The same benchmark on pgvector and on Qdrant: recall, search time, and the time each database reports.

    uv run python -m vector_databases.compare_databases

Run pg_indexes.py first: it leaves the tickets table with an HNSW index. The Qdrant side is a
fresh practice copy of the 5,000 ticket vectors (deleted at the end). Both use their default
HNSW settings, on the same computer, with the same 45 questions. Takes about a minute.
"""

import json
import time
import urllib.request

import numpy as np
from qdrant_client import QdrantClient

import settings
from vector_databases.pg_indexes import conn, exact, plan, question_vectors, recall
from vector_databases.recall_test import recall_at_k
from vector_databases.tune_hnsw import practice_collection

ROUNDS = 5  # each question is asked this many times, so the numbers are steadier
COLLECTION = "compare_qdrant"
rest = QdrantClient(url=settings.QDRANT_URL)  # the client we used all course
grpc = QdrantClient(url=settings.QDRANT_URL, prefer_grpc=True)  # the same server, through gRPC


def timed(search) -> list:
    """Milliseconds for each search: one untimed pass first, then ROUNDS passes over the 45 questions."""
    for vector in question_vectors:
        search(vector)
    times = []
    for _ in range(ROUNDS):
        for vector in question_vectors:
            start = time.perf_counter()
            search(vector)
            times.append((time.perf_counter() - start) * 1000)
    return times


def pg_search(vector):
    conn.execute("SELECT id, embedding <=> %s AS distance FROM tickets ORDER BY distance LIMIT 10", (vector,)).fetchall()


def pg_own_time(vector) -> float:
    """The time Postgres reports for running the search (EXPLAIN ANALYZE's execution time), in ms."""
    report = conn.execute("EXPLAIN (ANALYZE, FORMAT JSON) SELECT id, embedding <=> %s AS distance FROM tickets "
                          "ORDER BY distance LIMIT 10", (vector,)).fetchone()[0]
    return report[0]["Execution Time"]


def qdrant_own_time(vector) -> float:
    """The time Qdrant reports for processing the request (the 'time' field of its REST reply), in ms."""
    body = json.dumps({"query": vector.tolist(), "limit": 10}).encode()
    request = urllib.request.Request(f"{settings.QDRANT_URL}/collections/{COLLECTION}/points/query", data=body,
                                     headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request).read())["time"] * 1000


def row(label: str, recall_10: float, times: list, own: list | None) -> None:
    own_p50 = f"{np.percentile(own, 50):.2f} ms" if own else "not reported"
    print(f"{label:<34} {recall_10:>9.3f} {np.percentile(times, 50):>8.2f} {np.percentile(times, 95):>8.2f}   {own_p50:>12}")


if __name__ == "__main__":
    assert "Index Scan" in plan(), "No HNSW index on tickets: run vector_databases.pg_indexes first"
    practice_collection(COLLECTION)  # a fresh Qdrant copy with the default HNSW settings, fully indexed
    truths = [exact(vector) for vector in question_vectors]

    print(f"{'Setup':<34} {'recall@10':>9} {'p50 ms':>8} {'p95 ms':>8}   {'own time p50':>12}")
    for ef in (40, 100):  # 40 is pgvector's default
        conn.execute(f"SET hnsw.ef_search = {ef}")  # for this connection, so each search is one query
        row(f"pgvector HNSW, ef_search {ef}", recall(truths, f"hnsw.ef_search = {ef}"),
            timed(pg_search), [pg_own_time(vector) for vector in question_vectors])
    conn.execute("RESET hnsw.ef_search")

    qdrant_recall = recall_at_k(COLLECTION)
    row("Qdrant HNSW, default, REST", qdrant_recall,
        timed(lambda vector: rest.query_points(COLLECTION, query=vector.tolist(), limit=10)),
        [qdrant_own_time(vector) for vector in question_vectors])
    row("Qdrant HNSW, default, gRPC", qdrant_recall,
        timed(lambda vector: grpc.query_points(COLLECTION, query=vector.tolist(), limit=10)), None)
    print(f"(p50 and p95 of {ROUNDS * len(question_vectors)} searches each, timed from Python;"
          " own time = what the database reports for its own work)")

    rest.delete_collection(COLLECTION)
