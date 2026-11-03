"""Score each way of searching on the golden set: hit rate@10, precision@10 and MRR.

    uv run python -m vector_databases.score_search

Run hybrid_search.py first, so the tickets have their BM25 index.
eval/golden_set.json lists 63 questions with the tickets that count as right answers.
At the end it picks a winner on half the questions and checks it on the other half, as Qdrant advises.
"""

import json
from statistics import mean

import settings
from vector_databases.hybrid_search import hybrid_search, keyword_search, vector_search

K = 10
golden = json.loads((settings.ROOT / "eval" / "golden_set.json").read_text())


def precision_at_k(found: list, right: set, k: int = K) -> float:
    """The share of the top k results that are right."""
    return sum(ticket in right for ticket in found[:k]) / k


def reciprocal_rank(found: list, right: set, k: int = K) -> float:
    """1 / the position of the first right result in the top k, or 0 if there is none."""
    return next((1 / position for position, ticket in enumerate(found[:k], start=1) if ticket in right), 0.0)


def score(search, questions: list) -> tuple:
    """Hit rate@k, precision@k and MRR of one way of searching, over these questions."""
    hits, precisions, ranks = [], [], []
    for item in questions:
        found, right = [point.payload["ticket_id"] for point in search(item["question"])], set(item["right"])
        precisions.append(precision_at_k(found, right))
        ranks.append(reciprocal_rank(found, right))
        hits.append(ranks[-1] > 0)  # a right ticket somewhere in the top k
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
    groups = {"All 63": golden}
    for kind in ("everyday", "error code"):
        groups[f"{kind} ({sum(q['kind'] == kind for q in golden)})"] = [q for q in golden if q["kind"] == kind]
    for name, questions in groups.items():
        print(f"\n{name} questions        hit rate@{K}  precision@{K}    MRR")
        for setup, search in SETUPS.items():
            hit, precision, mrr = score(search, questions)
            print(f"{setup:<25} {hit:>11.3f} {precision:>13.3f} {mrr:>6.3f}")

    # Qdrant's advice: pick the winner on one half of the questions, then measure it on the other half
    pick, check = golden[0::2], golden[1::2]
    print(f"\nPick by MRR on {len(pick)} questions, then check on the other {len(check)}:")
    for setup, search in SETUPS.items():
        print(f"{setup:<25} picking half {score(search, pick)[2]:.3f}   checking half {score(search, check)[2]:.3f}")
