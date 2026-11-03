"""Print the answer while the model is still writing it.

    uv run python -m llm_basics.streaming             # the provider set in .env
    uv run python -m llm_basics.streaming openai      # or pick one: anthropic / openai
"""

import sys

import anthropic
import openai

import settings

provider = sys.argv[1] if len(sys.argv) > 1 else settings.LLM_PROVIDER
question = "Give three common reasons people get locked out of a work app, one line each."

if provider == "anthropic":
    with anthropic.Anthropic().messages.stream(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": question}],
        thinking={"type": "disabled"},
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
else:
    with openai.OpenAI().responses.stream(
        model=settings.OPENAI_MODEL,
        max_output_tokens=300,
        input=question,
        reasoning={"effort": "none"},
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
print()
