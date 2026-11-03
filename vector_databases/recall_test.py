"""How much does a fast search miss? Compare it with exact search on 45 real questions.

    uv run python -m vector_databases.recall_test

Recall@10: of the 10 tickets exact search ranks closest, how many the normal search
also returns, averaged over the questions. 1.000 means it missed nothing.
The later lectures in this section import recall_at_k() from here.
"""

import json

from qdrant_client import QdrantClient, models

import settings
from vector_databases.embeddings import embed

K = 10
client = QdrantClient(url=settings.QDRANT_URL)
questions = [q["query"] for q in json.loads((settings.ROOT / "eval" / "search_queries.json").read_text())]
question_vectors = embed(questions)  # the same model as the tickets, from the cache after the first run


def ground_truth(collection: str, vector, k: int = K) -> set:
    """The IDs exact search ranks in the top k, plus any ticket that ties with the k-th score."""
    exact = client.query_points(collection, query=vector.tolist(), limit=k + 10,
                                search_params=models.SearchParams(exact=True)).points
    cutoff = exact[k - 1].score
    return {point.id for point in exact if point.score >= cutoff - 1e-6}  # some tickets have the same text


def recall_at_k(collection: str, k: int = K, **search) -> float:
    """The share of the true top k that a normal search returns. search: settings such as hnsw_ef=64."""
    found = 0
    for vector in question_vectors:
        truth = ground_truth(collection, vector, k)
        results = client.query_points(collection, query=vector.tolist(), limit=k,
                                      search_params=models.SearchParams(**search)).points
        found += len({point.id for point in results} & truth)
    return found / (k * len(question_vectors))


if __name__ == "__main__":
    info = client.get_collection("tickets")
    print(f"tickets: {info.points_count:,} points, {info.indexed_vectors_count:,} of them in an HNSW index")
    print(f"recall@{K} over {len(questions)} questions: {recall_at_k('tickets'):.3f}")
