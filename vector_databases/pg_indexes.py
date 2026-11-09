"""Indexes in pgvector, measured on our tickets: HNSW, IVFFlat, the filter trap and halfvec.

    uv run python -m vector_databases.pg_indexes

Run pg_store.py first. It ends with one HNSW index on the table, for the next lecture.
"""

import json

import settings
from vector_databases.embeddings import embed
from vector_databases.pg_store import connect

conn = connect()
questions = [q["query"] for q in json.loads((settings.ROOT / "eval" / "search_queries.json").read_text())]
question_vectors = embed(questions)  # the 45 everyday questions, from the cache
HALF = f"halfvec({len(question_vectors[0])})"  # halfvec(1536)
HNSW = "CREATE INDEX tickets_hnsw ON tickets USING hnsw (embedding vector_cosine_ops)"


def nearest(vector, where="TRUE", params=(), half=False):
    """The IDs of the 10 closest tickets."""
    distance = f"embedding::{HALF} <=> %s::{HALF}" if half else "embedding <=> %s"
    rows = conn.execute(f"SELECT id FROM tickets WHERE {where} ORDER BY {distance} LIMIT 10",
                        (*params, vector)).fetchall()
    return {row[0] for row in rows}


def exact(vector, where="TRUE", params=()):
    """The true top 10: with the index switched off, Postgres compares every row.
    Some tickets have the same text, so a ticket exactly as close as the 10th counts too."""
    conn.execute("SET enable_indexscan = off")
    rows = conn.execute(f"SELECT id, embedding <=> %s AS distance FROM tickets WHERE {where} "
                        "ORDER BY distance LIMIT 20", (vector, *params)).fetchall()
    conn.execute("RESET enable_indexscan")
    if len(rows) <= 10:
        return {row[0] for row in rows}
    tenth = rows[9][1]
    return {id_ for id_, distance in rows if distance <= tenth + 1e-6}


def recall(truths, half=False):
    """How many of the true top 10 the index finds, on average (1.000 = all of them).
    On our small table Postgres often reads every row instead, so the index is forced here."""
    conn.execute("SET enable_seqscan = off")
    found = 0
    for vector, truth in zip(question_vectors, truths):
        found += len(nearest(vector, half=half) & truth)
    conn.execute("RESET enable_seqscan")
    return found / (10 * len(truths))


def plan():
    """How Postgres will read the table for a search, from its EXPLAIN plan."""
    rows = conn.execute("EXPLAIN (COSTS FALSE) SELECT id FROM tickets ORDER BY embedding <=> %s LIMIT 10",
                        (question_vectors[0],)).fetchall()
    for row in rows:  # one row per line of the plan
        if "Scan" in row[0]:
            return row[0].replace("->", "").strip()  # the line without its arrow, e.g. "Seq Scan on tickets"


def size(index):
    """How much disk space an index takes, as text like "37 MB"."""
    return conn.execute("SELECT pg_size_pretty(pg_relation_size(%s))", (index,)).fetchone()[0]


if __name__ == "__main__":
    for name in ("tickets_hnsw", "tickets_ivfflat", "tickets_hnsw_half"):
        conn.execute(f"DROP INDEX IF EXISTS {name}")  # start with no index
    truths = [exact(vector) for vector in question_vectors]

    # 1. No index: Postgres reads every row, which is exact search
    print("No index:  ", plan())

    # 2. HNSW with pgvector's defaults. ef_search = candidates kept while searching (default 40)
    conn.execute(HNSW)
    print("HNSW index:", plan(), size("tickets_hnsw"))
    for ef in (10, 20, 40, 100):
        conn.execute(f"SET hnsw.ef_search = {ef}")
        print(f"  ef_search {ef}: recall@10 {recall(truths):.3f}")
    conn.execute("RESET hnsw.ef_search")

    # 3. IVFFlat with 5 lists (5,000 rows / 1000). probes = lists searched (default 1)
    conn.execute("DROP INDEX tickets_hnsw")
    conn.execute("CREATE INDEX tickets_ivfflat ON tickets USING ivfflat (embedding vector_cosine_ops) WITH (lists = 5)")
    print("\nIVFFlat index:", size("tickets_ivfflat"))
    for probes in (1, 2, 3):
        conn.execute(f"SET ivfflat.probes = {probes}")
        print(f"  probes {probes}: recall@10 {recall(truths):.3f}")
    conn.execute("RESET ivfflat.probes")
    conn.execute("DROP INDEX tickets_ivfflat")

    # 4. The filter trap: with the index in use, the filter runs after the search
    conn.execute(HNSW)
    vector = embed(["we stopped using it, can we have our money back?"])[0]
    truth = exact(vector, where="plan = %s", params=("Starter",))
    print("\nStarter customers only")
    conn.execute("SET enable_seqscan = off")  # use the index, as on a big table
    for scan in ("off", "strict_order", "relaxed_order"):  # off is the default
        conn.execute(f"SET hnsw.iterative_scan = {scan}")
        found = nearest(vector, where="plan = %s", params=("Starter",))
        print(f"  iterative scan {scan}: {len(found)} results, {len(found & truth)} of them in the true top 10")
    conn.execute("RESET hnsw.iterative_scan")
    conn.execute("RESET enable_seqscan")

    # 5. halfvec: the same HNSW index with each number in 2 bytes instead of 4
    conn.execute(f"CREATE INDEX tickets_hnsw_half ON tickets USING hnsw ((embedding::{HALF}) halfvec_cosine_ops)")
    print(f"\nHNSW on vector:  {size('tickets_hnsw')}, recall@10 {recall(truths):.3f}")
    print(f"HNSW on halfvec: {size('tickets_hnsw_half')}, recall@10 {recall(truths, half=True):.3f}")
    conn.execute("DROP INDEX tickets_hnsw_half")  # keep one HNSW index for the next lecture
