"""Our embed() as a Chroma embedding function.

We attach it to the collection. Chroma then calls it for the tickets we add and for
every question we search with, so both always go through the same model. Chroma also
saves its name and settings with the collection, and rebuilds it when we reopen it.
"""

from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

import settings
from vector_databases import embeddings


@register_embedding_function  # lets Chroma find this class again by its name
class LarkspanEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, provider: str = settings.EMBED_PROVIDER):
        self.provider = provider  # "openai", or "local" (free)

    def __call__(self, input: Documents) -> Embeddings:
        return list(embeddings.embed(list(input), self.provider))  # batched and cached

    @staticmethod
    def name() -> str:
        return "larkspan"

    def get_config(self) -> dict[str, Any]:
        return {"provider": self.provider}  # what Chroma saves with the collection

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "LarkspanEmbedding":
        return LarkspanEmbedding(config["provider"])  # how Chroma rebuilds it on reopen
