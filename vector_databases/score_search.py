"""Score six ways of searching on the golden set (our 63 test questions with their right tickets).

    uv run python -m vector_databases.score_search

Run hybrid_search.py first, so the tickets have their keyword index.
"""

import json
from statistics import mean

import settings
from vector_databases.hybrid_search import hybrid_search, keyword_search, vector_search

golden = json.loads((settings.ROOT / "eval" / "golden_set.json").read_text())

def precision_at_k(found, right, k=10):
    """How many of the top 10 are right, as a share: 4 right → 0.4."""
    right_ones = [ticket for ticket in found[:k] if ticket in right]
    return len(right_ones) / k


def reciprocal_rank(found, right, k=10):
    """1 ÷ the place of the first right ticket: 5th → 0.2. None right → 0."""
    for place, ticket in enumerate(found[:k], start=1):
        if ticket in right:
            return 1 / place
    return 0.0


def score(search, questions):
    """Run one search on every question, then average: hit rate, precision@10 and MRR."""
    hits, precisions, ranks = [], [], []
    for q in questions:
        found = [point.payload["ticket_id"] for point in search(q["question"])]
        right = set(q["right"])
        rank = reciprocal_rank(found, right)
        hits.append(rank > 0)  # a right ticket somewhere in the top 10
        precisions.append(precision_at_k(found, right))
        ranks.append(rank)
    return mean(hits), mean(precisions), mean(ranks)


SETUPS = {
    "Vector only": vector_search,
    "Keyword only": keyword_search,
    "Hybrid, equal weights": hybrid_search,
    "Hybrid, keyword weight 2": lambda question: hybrid_search(question, weights=[1, 2]),
    "Hybrid, keyword weight 3": lambda question: hybrid_search(question, weights=[1, 3]),
    "Hybrid, vector weight 2": lambda question: hybrid_search(question, weights=[2, 1]),
}

if __name__ == "__main__":
    everyday = [q for q in golden if q["kind"] == "everyday"]
    codes = [q for q in golden if q["kind"] == "error code"]
    for title, questions in [("All 63", golden), ("everyday (45)", everyday), ("error code (18)", codes)]:
        print(f"\n{title} questions        hit rate  precision    MRR")
        for name, search in SETUPS.items():
            hit, precision, mrr = score(search, questions)
            print(f"{name:<25}    {hit:.3f}      {precision:.3f}  {mrr:.3f}")