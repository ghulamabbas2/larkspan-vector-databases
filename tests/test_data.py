import json
import subprocess
import sys

import settings


def load_tickets() -> list[dict]:
    with (settings.DATA_DIR / "tickets.jsonl").open() as f:
        return [json.loads(line) for line in f]


def test_dataset_sizes():
    customers = json.loads((settings.DATA_DIR / "customers.json").read_text())
    assert len(customers) == 50
    tickets = load_tickets()
    assert len(tickets) == 5000
    assert len({t["id"] for t in tickets}) == 5000


def test_every_ticket_has_a_topic():
    truth = json.loads((settings.ROOT / "eval" / "ticket_topics.json").read_text())
    assert {t["id"] for t in load_tickets()} == set(truth["tickets"])


def test_generator_is_deterministic():
    before = (settings.DATA_DIR / "tickets.jsonl").read_bytes()
    subprocess.run([sys.executable, "tools/generate_data.py"], cwd=settings.ROOT, check=True, capture_output=True)
    assert (settings.DATA_DIR / "tickets.jsonl").read_bytes() == before
