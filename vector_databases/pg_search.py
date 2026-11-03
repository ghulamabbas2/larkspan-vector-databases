"""Search the tickets in Postgres: the closest tickets to a question, by cosine distance.

    uv run python -m vector_databases.pg_search

Run pg_store.py first. Without an index, Postgres compares the question with every
row, so this is exact search.
"""

from vector_databases.embeddings import embed
from vector_databases.pg_store import connect

conn = connect()


def search(question: str, limit: int = 10) -> list:
    """(id, subject, cosine similarity) of the closest tickets, closest first."""
    rows = conn.execute("""
        SELECT id, subject, embedding <=> %(question)s AS distance
        FROM tickets
        ORDER BY distance
        LIMIT %(limit)s
        """, {"question": embed([question])[0], "limit": limit}).fetchall()
    return [(id_, subject, 1 - distance) for id_, subject, distance in rows]  # cosine similarity = 1 - distance


if __name__ == "__main__":
    question = "the email to set a new passcode is dead when I click it"
    print(f"Question: {question!r}\n")
    for id_, subject, similarity in search(question, limit=5):
        print(f"{similarity:.3f}  {id_}  {subject}")
