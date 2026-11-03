"""Compress the ticket vectors with quantization: how much memory it saves, and what it costs.

    uv run python -m vector_databases.quantize

Each step builds a practice copy of the 5,000 ticket vectors (deleted at the end).
The originals stay on disk; only the small copies are kept in RAM.
"""

import json
import urllib.request

from qdrant_client import models

import settings
from vector_databases.recall_test import client, question_vectors, recall_at_k
from vector_databases.tune_hnsw import practice_collection

PRACTICE = "tickets_quantized"


def vector_ram_mb(collection: str) -> tuple[float, float]:
    """RAM used by the original vectors and by the small copies, in MB, from Qdrant's memory report."""
    url = f"{settings.QDRANT_URL}/collections/{collection}/memory"  # no Python method for this yet
    report = json.load(urllib.request.urlopen(url))["result"]["vectors"][0]
    ram = lambda part: (part["ram_bytes"] + part["cached_bytes"]) / 1e6
    return ram(report["storage"]), ram(report["quantized"]) if "quantized" in report else 0.0


def measure(name: str, quantization) -> None:
    practice_collection(PRACTICE, quantization=quantization, memory=models.Memory.COLD)  # originals on disk
    originals, small = vector_ram_mb(PRACTICE)
    print(f"\n{name}: small copy {small:.2f} MB in RAM, originals {originals:.2f} MB in RAM")

    first = question_vectors[0].tolist()
    small_only = client.query_points(PRACTICE, query=first, limit=1, search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(rescore=False))).points[0]
    exact = client.query_points(PRACTICE, query=first, limit=1, search_params=models.SearchParams(exact=True)).points[0]
    print(f"  top score for the first question: small copy alone {small_only.score:.4f}, originals {exact.score:.4f}")

    for label, params in [("no rescoring", models.QuantizationSearchParams(rescore=False)),
                          ("rescoring", models.QuantizationSearchParams(rescore=True)),
                          ("rescoring, 4x candidates", models.QuantizationSearchParams(rescore=True, oversampling=4.0))]:
        print(f"  recall@10, {label:<24} {recall_at_k(PRACTICE, quantization=params):.3f}")


if __name__ == "__main__":
    print(f"Originals: 5,000 vectors x 1,536 numbers x 4 bytes = {5000 * 1536 * 4 / 1e6:.2f} MB")
    measure("Scalar (int8)", models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, memory=models.Memory.PINNED)))
    measure("Binary (1 bit)", models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(memory=models.Memory.PINNED)))
    client.delete_collection(PRACTICE)
