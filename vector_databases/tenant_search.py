"""Search one customer's tickets only, and check that no other customer's ticket ever comes back.

    uv run python -m vector_databases.tenant_search

Each Larkspan customer is a tenant: its tickets must only ever be shown to that customer.
Run filter_search.py first (this script reuses its helpers).
"""

import json

from qdrant_client import models

import settings
from vector_databases.filter_search import add_payload_indexes, only, search

def search_for_customer(customer_id: str, question: str, limit: int = 10) -> list:
    """The customer's own tickets closest to the question. The filter is always added here, never by the caller."""
    return search(question, only(customer_id=customer_id), limit)

if __name__ == "__main__":
    # 1. One collection for all 50 customers, with a tenant index on customer_id
    tenant_index = models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True)
    new = add_payload_indexes({"customer_id": tenant_index})
    print(f"Tenant index on customer_id: {'created' if new else 'already there'}")

    # 2. The leak: the smallest customer asks a question, and we forget the filter
    customer, question = "cus_026", "every webhook delivery to our endpoint fails"
    leaked = [p.payload["customer_id"] for p in search(question) if p.payload["customer_id"] != customer]
    print(f"\nNo filter: {len(leaked)} of 10 results belong to other customers ({', '.join(sorted(set(leaked)))})")
    own = search_for_customer(customer, question)
    print(f"search_for_customer: {sum(p.payload['customer_id'] == customer for p in own)} of {len(own)} results are {customer}'s")

    # 3. Every customer asks all 45 questions: count results that belong to someone else
    customers = [c["id"] for c in json.loads((settings.DATA_DIR / "customers.json").read_text())]
    questions = [q["query"] for q in json.loads((settings.ROOT / "eval" / "search_queries.json").read_text())]
    wrong = sum(p.payload["customer_id"] != c for c in customers for q in questions for p in search_for_customer(c, q))
    print(f"\nChecked {len(customers) * len(questions):,} searches: {wrong} results from another customer")