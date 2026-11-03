"""The Larkspan ticket search API: hybrid search over all 5,000 tickets, with optional filters.

    uv run fastapi dev

Then open http://127.0.0.1:8000 for the search page, or /docs to try each endpoint.
Run the production section first: the API searches the alias tickets_live
(text-embedding-3-large behind it) with the hybrid search we measured there, so the alias,
the BM25 text index and the payload indexes must exist.
"""

from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import models

from vector_databases.embeddings import embed
from vector_databases.filter_search import client, only
from vector_databases.hybrid_search import CANDIDATES
from vector_databases.migrate_model import ALIAS, MODELS, live_collection

app = FastAPI(title="Larkspan ticket search")


class Ticket(BaseModel):
    """One search result: what the support team sees, plus the RRF score that ranked it."""
    ticket_id: str
    subject: str
    body: str
    customer_id: str
    plan: str
    priority: str
    product_area: str
    status: str
    score: float


def search_tickets(question: str, limit: int = 10, query_filter: models.Filter | None = None) -> list[Ticket]:
    """Hybrid search on the live collection: vector and BM25 lists, each filtered, merged by RRF."""
    vector = embed([question], model=MODELS[live_collection()])[0].tolist()  # the model of the collection behind the alias
    points = client.query_points(
        ALIAS,
        prefetch=[
            models.Prefetch(query=vector, filter=query_filter, limit=CANDIDATES),  # vector search
            models.Prefetch(query=models.TextQuery(text=question), using="text", filter=query_filter,
                            limit=CANDIDATES),  # BM25 keyword search
        ],
        query=models.RrfQuery(rrf=models.Rrf()),
        limit=limit,
        with_payload=True,
    ).points
    return [Ticket(**{field: point.payload[field] for field in Ticket.model_fields if field != "score"},
                   score=point.score) for point in points]


@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=1, max_length=500, description="The question, in everyday words or an error code")],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    plan: Literal["Starter", "Growth", "Enterprise"] | None = None,
    priority: Literal["low", "normal", "high", "urgent"] | None = None,
    customer_id: str | None = None,
) -> list[Ticket]:
    fields = {key: value for key, value in {"plan": plan, "priority": priority, "customer_id": customer_id}.items()
              if value is not None}
    return search_tickets(q, limit, only(**fields) if fields else None)


@app.get("/health")
def health() -> dict:
    """Which collection the alias points at, and how many tickets it holds."""
    return {"alias": ALIAS, "collection": live_collection(), "tickets": client.count(ALIAS).count}


@app.get("/", include_in_schema=False)  # a page for people, not an endpoint for apps, so it stays off /docs
def page() -> FileResponse:
    """The search page. Its JavaScript calls /search from the browser."""
    return FileResponse(Path(__file__).parent / "search_page.html")
