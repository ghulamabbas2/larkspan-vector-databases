"""Turn one sentence into an embedding and look at what comes back.

    uv run python -m vector_databases.first_embedding
"""

import numpy as np
import openai

import settings  # loads .env, so the openai library finds its API key

client = openai.OpenAI()
response = client.embeddings.create(
    model=settings.OPENAI_EMBED_MODEL,
    input="We were charged twice for the same month",
)
vector = response.data[0].embedding

print(f"Model:      {response.model}")
print(f"Dimensions: {len(vector)}")
print(f"First 6:    {[round(x, 4) for x in vector[:6]]}")
print(f"Length:     {np.linalg.norm(vector):.4f}")
print(f"Tokens:     {response.usage.total_tokens}")
