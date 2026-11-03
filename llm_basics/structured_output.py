"""Get the answer as a Python object with fixed fields, instead of free text.

    uv run python -m llm_basics.structured_output             # the provider set in .env
    uv run python -m llm_basics.structured_output openai      # or pick one: anthropic / openai
"""

import json
import sys
from typing import Literal

import anthropic
import openai
from pydantic import BaseModel, Field

import settings

ProductArea = Literal[
    "Account & Login", "API & Webhooks", "Automations", "Billing", "Boards & Views",
    "Helpdesk & SLA", "Import & Export", "Integrations", "Mobile App", "Notifications & Email",
    "Performance", "Permissions & Roles", "Projects & Tasks", "Reports & Dashboards", "Security & Privacy",
]


class Triage(BaseModel):
    product_area: ProductArea
    priority: Literal["low", "normal", "high", "urgent"]
    summary: str = Field(description="One short sentence")
    error_code: str | None = Field(description="Like ERR-4012, if the ticket mentions one")


with (settings.DATA_DIR / "tickets.jsonl").open() as f:
    ticket = next(t for t in map(json.loads, f) if t["id"] == "TCK-10015")
prompt = f"Triage this support ticket.\n\nSubject: {ticket['subject']}\n\n{ticket['body']}"

provider = sys.argv[1] if len(sys.argv) > 1 else settings.LLM_PROVIDER

if provider == "anthropic":
    response = anthropic.Anthropic().messages.parse(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "disabled"},
        output_format=Triage,
    )
    triage = response.parsed_output
else:
    response = openai.OpenAI().responses.parse(
        model=settings.OPENAI_MODEL,
        max_output_tokens=500,
        input=prompt,
        reasoning={"effort": "none"},
        text_format=Triage,
    )
    triage = response.output_parsed

print(repr(triage))
print(f"Larkspan's own label: {ticket['product_area']}, {ticket['priority']}")

if triage.priority in ("high", "urgent"):
    print(f"\nPage the on-call engineer: {triage.summary}")
