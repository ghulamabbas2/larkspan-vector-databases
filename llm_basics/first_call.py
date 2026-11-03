"""Send one question to Claude or GPT, and see the answer, the tokens and the cost.

    uv run python -m llm_basics.first_call             # the provider set in .env
    uv run python -m llm_basics.first_call openai      # or pick one: anthropic / openai
"""

import sys

import anthropic
import openai

import settings  # loads .env, so both libraries find their API key

# Dollars per 1 million tokens: (input, output). Checked 2026-10-31.
# These are the prices for normal-length prompts; very long prompts cost more.
# If you change a model in .env, add its prices here.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-6-luna": (0.10, 0.50),
}

provider = sys.argv[1] if len(sys.argv) > 1 else settings.LLM_PROVIDER
system = "You are a support lead at Larkspan. Answer in one short sentence."
question = "Why do support teams tag tickets by product area?"

if provider == "anthropic":
    model = settings.ANTHROPIC_MODEL
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": question}],
        thinking={"type": "disabled"},
    )
    answer = next(block.text for block in response.content if block.type == "text")
    stopped = response.stop_reason  # "end_turn", or "max_tokens" if the answer was cut off
else:
    model = settings.OPENAI_MODEL
    client = openai.OpenAI()
    response = client.responses.create(
        model=model,
        max_output_tokens=300,
        instructions=system,
        input=question,
        reasoning={"effort": "none"},
    )
    answer = response.output_text
    stopped = response.status  # "completed", or "incomplete" if the answer was cut off

tokens_in = response.usage.input_tokens
tokens_out = response.usage.output_tokens
price_in, price_out = PRICES[model]
cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000

print(f"[{model}] {answer}\n")
print(f"Stopped: {stopped}")
print(f"Tokens: {tokens_in} in, {tokens_out} out")
print(f"Cost:   ${cost:.6f}  (about {1 / cost:,.0f} calls like this per dollar)")
