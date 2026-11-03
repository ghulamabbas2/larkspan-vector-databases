"""Load the Larkspan support tickets, and the text of each one that we embed."""

import json

import settings


def load_tickets() -> list[dict]:
    with (settings.DATA_DIR / "tickets.jsonl").open() as f:
        return [json.loads(line) for line in f]


def ticket_text(ticket: dict) -> str:
    return f"{ticket['subject']}\n{ticket['body']}"
