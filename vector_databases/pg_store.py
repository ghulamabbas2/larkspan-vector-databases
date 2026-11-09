"""Store the Larkspan tickets in Postgres with pgvector. Safe to run as often as you like.

    uv run python -m vector_databases.pg_store

Postgres runs in Docker next to Qdrant (docker compose up -d). Each row of the
tickets table holds one ticket: its fields, its text and its embedding.
The next scripts in this section import connect() from here.
"""

import time

import psycopg
from pgvector.psycopg import register_vector

import settings
from vector_databases.embeddings import embed
from vector_databases.tickets import load_tickets, ticket_text

FIELDS = ["plan", "product_area", "priority", "status", "channel", "customer_id"]


def connect() -> psycopg.Connection:
    """A connection to our Postgres, with pgvector switched on and its vector type registered."""
    conn = psycopg.connect(settings.DATABASE_URL, autocommit=True)  # autocommit: each statement is saved at once
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # once per database; later runs skip it
    register_vector(conn)  # lets psycopg send and receive NumPy vectors
    return conn


if __name__ == "__main__":
    start = time.perf_counter()
    tickets = load_tickets()
    vectors = embed([ticket_text(ticket) for ticket in tickets])  # from the cache if we made them before

    conn = connect()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS tickets (
            id text PRIMARY KEY,
            plan text, product_area text, priority text, status text, channel text, customer_id text,
            subject text, body text,
            embedding vector({len(vectors[0])})
        )""")

    rows = [(ticket["id"], *(ticket[field] for field in FIELDS), ticket["subject"], ticket["body"], vector)
            for ticket, vector in zip(tickets, vectors)]
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO tickets (id, plan, product_area, priority, status, channel, customer_id, subject, body, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                plan = EXCLUDED.plan, product_area = EXCLUDED.product_area, priority = EXCLUDED.priority,
                status = EXCLUDED.status, channel = EXCLUDED.channel, customer_id = EXCLUDED.customer_id,
                subject = EXCLUDED.subject, body = EXCLUDED.body, embedding = EXCLUDED.embedding
            """, rows)  # a ticket that's already there is updated, a new one is added

    print(f"Loaded {len(rows):,} tickets in {time.perf_counter() - start:.1f} s")
    print(f"The table now holds {conn.execute('SELECT count(*) FROM tickets').fetchone()[0]:,} rows")
