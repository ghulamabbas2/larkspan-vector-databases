"""Benchmark the search API end to end, then write the results to reports/search_benchmark.md.

    uv run python -m vector_databases.benchmark_report

Start the API first, in another terminal: uv run fastapi dev
Every question goes through the API over HTTP, the way an app would send it. Takes about a minute.
"""

import json
import time
from statistics import mean

import httpx
import numpy as np
import openai

import settings
from vector_databases.migrate_model import MODELS, live_collection
from vector_databases.score_search import golden, precision_at_k, reciprocal_rank
from vector_databases.search_api import search_tickets

API = "http://127.0.0.1:8000"
ROUNDS = 5  # each question is asked this many times for the timings, so the numbers are steadier
REPORT = settings.ROOT / "reports" / "search_benchmark.md"
api = httpx.Client(base_url=API, timeout=30)


def ask(question: str, **filters) -> list[dict]:
    response = api.get("/search", params={"q": question, **filters})
    response.raise_for_status()  # stop with a clear error on anything but 200 OK
    return response.json()


def scores(item: dict) -> tuple:
    """Ask one golden question once. Returns (hit, precision@10, reciprocal rank) for its answer."""
    found, right = [ticket["ticket_id"] for ticket in ask(item["question"])], set(item["right"])
    rank = reciprocal_rank(found, right)
    return rank > 0, precision_at_k(found, right), rank


def wrong_results(field: str, values: list, questions: list) -> tuple:
    """Ask every question with each filter value. Returns (searches, results, results that break the filter)."""
    searches = results = wrong = 0
    for value in values:
        for question in questions:
            tickets = ask(question, **{field: value})
            searches, results = searches + 1, results + len(tickets)
            wrong += sum(ticket[field] != value for ticket in tickets)
    return searches, results, wrong


def milliseconds(run, questions: list) -> list:
    """Time each call: one untimed pass first, then ROUNDS passes over the questions."""
    for question in questions:
        run(question)
    times = []
    for _ in range(ROUNDS):
        for question in questions:
            start = time.perf_counter()
            run(question)
            times.append((time.perf_counter() - start) * 1000)
    return times


def new_question_ms(questions: list) -> list:
    """How long OpenAI takes to embed a question that isn't in our cache (called directly, so the cache is skipped)."""
    times = []
    for question in questions:
        start = time.perf_counter()
        openai.OpenAI().embeddings.create(model=MODELS[live_collection()], input=[question])
        times.append((time.perf_counter() - start) * 1000)
    return times


if __name__ == "__main__":
    health = api.get("/health").json()  # fails here if the API isn't running
    lines = [f"# Larkspan search benchmark\n",
             f"The API searches `{health['alias']}`, which points at `{health['collection']}` "
             f"({health['tickets']:,} tickets, {MODELS[health['collection']]}), with hybrid search.\n",
             f"## 1. Quality on the golden set (top 10 of each answer)\n",
             "| Questions | hit rate@10 | precision@10 | MRR |", "|---|---|---|---|"]
    answers = [(item["kind"], scores(item)) for item in golden]  # each question asked once, so the rows agree
    groups = {f"All {len(golden)}": [s for _, s in answers]}
    for kind in ("everyday", "error code"):
        groups[f"{kind} ({sum(k == kind for k, _ in answers)})"] = [s for k, s in answers if k == kind]
    for name, rows in groups.items():
        hit, precision, mrr = (mean(column) for column in zip(*rows))
        lines.append(f"| {name} | {hit:.3f} | {precision:.3f} | {mrr:.3f} |")

    lines += ["\n## 2. Filters: no result may break the filter\n",
              "| Filter | searches | results | results that break it |", "|---|---|---|---|"]
    everyday = [q["question"] for q in golden if q["kind"] == "everyday"]
    customers = [c["id"] for c in json.loads((settings.DATA_DIR / "customers.json").read_text())]
    checks = {"customer_id (all 50 customers)": ("customer_id", customers, everyday[:5]),
              "plan (each plan)": ("plan", ["Starter", "Growth", "Enterprise"], everyday),
              "priority (each priority)": ("priority", ["low", "normal", "high", "urgent"], everyday)}
    for name, (field, values, questions) in checks.items():
        searches, results, wrong = wrong_results(field, values, questions)
        lines.append(f"| {name} | {searches:,} | {results:,} | {wrong} |")

    questions = [q["question"] for q in golden]
    through_api = milliseconds(ask, questions)
    in_process = milliseconds(search_tickets, questions)
    new = new_question_ms(everyday[:10])
    lines += [f"\n## 3. Speed, one search at a time ({len(questions)} questions x {ROUNDS} rounds)\n",
              "| What we timed | p50 ms | p95 ms |", "|---|---|---|",
              f"| Through the API (HTTP), question vector from the cache | {np.percentile(through_api, 50):.1f} "
              f"| {np.percentile(through_api, 95):.1f} |",
              f"| The same search in Python, without the API | {np.percentile(in_process, 50):.1f} "
              f"| {np.percentile(in_process, 95):.1f} |",
              f"| Embedding a new question with OpenAI ({len(new)} questions, once each) | {np.percentile(new, 50):.1f} "
              f"| {np.percentile(new, 95):.1f} |"]

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {REPORT.relative_to(settings.ROOT)}")
