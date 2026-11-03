"""Indexes and filters in pgvector: HNSW, IVFFlat, filtered searches and halfvec, measured on our tickets.

    uv run python -m vector_databases.pg_indexes

Run pg_store.py first. Takes about 20 seconds. Every run starts by removing the indexes it
makes, and it ends with one HNSW index on the table, ready for the next lecture.
"""

import json
import time

import settings
from vector_databases.embeddings import embed
from vector_databases.pg_store import connect

K = 10
INDEXES = ["tickets_hnsw", "tickets_ivfflat", "tickets_hnsw_half"]
conn = connect()
questions = [q["query"] for q in json.loads((settings.ROOT / "eval" / "search_queries.json").read_text())]
question_vectors = embed(questions)  # the 45 everyday questions, from the cache
DIMS = len(question_vectors[0])


def nearest(vector, *options, where="TRUE", params=(), column="embedding", cast="") -> list:
    """IDs of the K closest tickets. options: settings for this search only, e.g. "hnsw.ef_search = 100"."""
    with conn.transaction():  # SET LOCAL lasts until the end of this transaction
        for option in options:
            conn.execute(f"SET LOCAL {option}")
        rows = conn.execute(f"SELECT id, {column} <=> %s{cast} AS distance FROM tickets WHERE {where} "
                            f"ORDER BY distance LIMIT {K}", (vector, *params)).fetchall()
    return [row[0] for row in rows]


def exact(vector, where="TRUE", params=()) -> set:
    """The true top K, with the index switched off, plus any ticket that ties with the K-th."""
    with conn.transaction():
        conn.execute("SET LOCAL enable_indexscan = off")  # pgvector's README: how to get exact results to compare with
        rows = conn.execute(f"SELECT id, embedding <=> %s AS distance FROM tickets WHERE {where} "
                            f"ORDER BY distance LIMIT {K + 10}", (vector, *params)).fetchall()
    if len(rows) <= K:
        return {id_ for id_, _ in rows}
    cutoff = rows[K - 1][1]
    return {id_ for id_, distance in rows if distance <= cutoff + 1e-6}  # some tickets have the same text


def recall(truths: list, *options, column="embedding", cast="") -> float:
    """The share of the true top K that the index returns, over the 45 questions.
    On our small table Postgres sometimes reads every row instead, so the index is forced here."""
    found = sum(len(set(nearest(vector, "enable_seqscan = off", *options, column=column, cast=cast)) & truth)
                for vector, truth in zip(question_vectors, truths))
    return found / (K * len(question_vectors))


def plan(*options, where="TRUE", params=()) -> str:
    """How Postgres will run a search: the scan line of its EXPLAIN plan."""
    with conn.transaction():
        for option in options:
            conn.execute(f"SET LOCAL {option}")
        lines = [row[0] for row in conn.execute(f"EXPLAIN SELECT id, embedding <=> %s AS distance FROM tickets "
                                                f"WHERE {where} ORDER BY distance LIMIT {K}",
                                                (question_vectors[0], *params))]
    return next(line.split("(cost")[0].strip(" ->") for line in lines if "Scan" in line)


def own_choice(*options) -> str:
    """What Postgres chooses by itself for this setting: the index, or reading every row."""
    return "the index" if "Index Scan" in plan(*options) else "every row"


def build(sql: str) -> float:
    """Create an index and return the seconds it took."""
    start = time.perf_counter()
    conn.execute(sql)
    return time.perf_counter() - start


def size(index: str) -> str:
    return conn.execute("SELECT pg_size_pretty(pg_relation_size(%s))", (index,)).fetchone()[0]


def drop(*names: str) -> None:
    for name in names:
        conn.execute(f"DROP INDEX IF EXISTS {name}")


HNSW = "CREATE INDEX tickets_hnsw ON tickets USING hnsw (embedding vector_cosine_ops)"
FILTERS = [  # (rule, values, question)
    ("plan = %s", ("Starter",), "we stopped using it, can we have our money back?"),
    ("plan = %s AND priority = %s", ("Starter", "urgent"), "we paid two times for the same month"),
    ("customer_id = %s", ("cus_026",), "the email to set a new passcode is dead when I click it"),
]

if __name__ == "__main__":
    drop(*INDEXES)  # start with no index, so every run measures the same thing
    truths = [exact(vector) for vector in question_vectors]

    # 1. No index: Postgres reads every row, which is exact search
    print(f"No index:   {plan()}")

    # 2. HNSW with pgvector's defaults (m 16, ef_construction 64)
    seconds = build(HNSW)
    print(f"HNSW index: {plan()} (built in {seconds:.1f} s, {size('tickets_hnsw')})")
    print("\nhnsw.ef_search   recall@10   Postgres by itself reads")
    for ef in (10, 20, 40, 100):  # 40 is the default
        option = f"hnsw.ef_search = {ef}"
        print(f"{ef:>14}   {recall(truths, option):>9.3f}   {own_choice(option)}")

    # 3. IVFFlat: 5,000 rows / 1000 = 5 lists, the README's starting point
    drop("tickets_hnsw")
    seconds = build("CREATE INDEX tickets_ivfflat ON tickets USING ivfflat (embedding vector_cosine_ops) WITH (lists = 5)")
    print(f"\nIVFFlat, 5 lists: built in {seconds:.1f} s, {size('tickets_ivfflat')}")
    print("probes   recall@10   Postgres by itself reads")
    for probes in (1, 2, 3):  # 1 is the default; the README suggests the square root of lists (about 2) to start
        option = f"ivfflat.probes = {probes}"
        print(f"{probes:>6}   {recall(truths, option):>9.3f}   {own_choice(option)}")
    drop("tickets_ivfflat")

    # 4. Filters with the HNSW index. On our small table Postgres chooses to read every row
    #    for a filtered search, so we switch that off to see what the index does on a big table.
    seconds = build(HNSW)  # a new graph: each build is a little different, so its recall can differ too
    print(f"\nHNSW index built again ({seconds:.1f} s), recall@10 at ef_search 40: {recall(truths):.3f}")
    rule, values, _ = FILTERS[0]
    print(f"Filtered search, Postgres's own choice: {plan(where=rule, params=values)}")
    print(f"With enable_seqscan = off:              {plan('enable_seqscan = off', where=rule, params=values)}")
    print(f"\n{'Filter':<26} {'index only':>16} {'strict_order':>16} {'relaxed_order':>16}")
    for rule, values, question in FILTERS:
        vector = embed([question])[0]
        truth = exact(vector, rule, values)
        cells = []
        for scan in (None, "strict_order", "relaxed_order"):
            options = ["enable_seqscan = off"] + ([f"hnsw.iterative_scan = {scan}"] if scan else [])
            found = nearest(vector, *options, where=rule, params=values)
            cells.append(f"{len(found)} ({len(set(found) & truth)} right)")
        print(f"{' and '.join(values):<26} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
    print("(results returned, and how many of them are in exact search's top 10)")

    # 5. halfvec: the same index with each number stored in 2 bytes instead of 4
    seconds = build(f"CREATE INDEX tickets_hnsw_half ON tickets USING hnsw ((embedding::halfvec({DIMS})) halfvec_cosine_ops)")
    half = recall(truths, column=f"embedding::halfvec({DIMS})", cast=f"::halfvec({DIMS})")
    print(f"\nHNSW on halfvec: {size('tickets_hnsw_half')}, recall@10 {half:.3f} (built in {seconds:.1f} s)")
    print(f"HNSW on vector:  {size('tickets_hnsw')} (the index from step 4)")
    drop("tickets_hnsw_half")  # the table keeps one HNSW index on the full vectors
