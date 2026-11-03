"""Build the golden set: test questions, each with the tickets that count as right answers.

    uv run python tools/make_golden_set.py

Writes eval/golden_set.json. The right answers come from how the tickets were generated
(eval/ticket_topics.json), not from people marking them, so we build the file instead of
typing it. Two kinds of question:
  - "everyday": the 45 questions from eval/search_queries.json. Right = every ticket about that issue.
  - "error code": each code on its own and in a sentence. Right = every ticket that contains the code.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, EVAL = ROOT / "data" / "larkspan", ROOT / "eval"

tickets = [json.loads(line) for line in (DATA / "tickets.jsonl").open()]
topic_of = json.loads((EVAL / "ticket_topics.json").read_text())["tickets"]
questions = json.loads((EVAL / "search_queries.json").read_text())
codes = json.loads((DATA / "error_codes.json").read_text())

golden = []
for q in questions:
    right = [t["id"] for t in tickets if topic_of[t["id"]] == q["topic"]]
    golden.append({"question": q["query"], "kind": "everyday", "right": right})
for code in codes:
    right = [t["id"] for t in tickets if code in t["subject"] + " " + t["body"]]
    golden.append({"question": code, "kind": "error code", "right": right})
    golden.append({"question": f"customer reports {code}", "kind": "error code", "right": right})

(EVAL / "golden_set.json").write_text(json.dumps(golden, indent=1) + "\n")
print(f"{len(golden)} questions → eval/golden_set.json")
