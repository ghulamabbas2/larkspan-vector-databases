"""Compare every setting from this section on our tickets: recall, latency and memory.

    uv run python -m vector_databases.benchmark

Builds five practice copies of the 5,000 ticket vectors (deleted at the end) and asks
each one the 45 questions five times. Takes under a minute.
"""

import time

import numpy as np
from qdrant_client import models

from vector_databases.quantize import vector_ram_mb
from vector_databases.recall_test import client, question_vectors, recall_at_k
from vector_databases.tune_hnsw import practice_collection

ROUNDS = 5  # each question is asked this many times, so the latency numbers are steadier


def latency_ms(collection: str, **search) -> tuple[float, float]:
    """p50 and p95 time for one search, in milliseconds."""
    times = []
    for _ in range(ROUNDS):
        for vector in question_vectors:
            start = time.perf_counter()
            client.query_points(collection, query=vector.tolist(), limit=10, search_params=models.SearchParams(**search))
            times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


cold, pinned = models.Memory.COLD, models.Memory.PINNED
collections = {
    "bench_default": dict(),
    "bench_weak": dict(hnsw=models.HnswConfigDiff(m=4, ef_construct=16)),
    "bench_scalar": dict(memory=cold, quantization=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, memory=pinned))),
    "bench_binary": dict(memory=cold, quantization=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(memory=pinned))),
    "bench_turbo": dict(memory=cold, quantization=models.TurboQuantization(  # Qdrant's suggested default
        turbo=models.TurboQuantQuantizationConfig(bits=models.TurboQuantBitSize.BITS4, memory=pinned))),
}
ram = {}
for name, config in collections.items():
    practice_collection(name, **config)
    ram[name] = sum(vector_ram_mb(name))  # measured before any search reads the originals

no_rescore = models.QuantizationSearchParams(rescore=False)
rescore_x4 = models.QuantizationSearchParams(rescore=True, oversampling=4.0)
settings_to_test = [
    ("Exact search", "bench_default", dict(exact=True)),
    ("HNSW, default graph", "bench_default", dict()),
    ("HNSW, weak graph, hnsw_ef 10", "bench_weak", dict(hnsw_ef=10)),
    ("Scalar, no rescoring", "bench_scalar", dict(quantization=no_rescore)),
    ("Scalar, rescoring x4", "bench_scalar", dict(quantization=rescore_x4)),
    ("Binary, no rescoring", "bench_binary", dict(quantization=no_rescore)),
    ("Binary, rescoring x4", "bench_binary", dict(quantization=rescore_x4)),
    ("TurboQuant 4-bit, no rescoring", "bench_turbo", dict(quantization=no_rescore)),
    ("TurboQuant 4-bit, rescoring x4", "bench_turbo", dict(quantization=rescore_x4)),
]

for name in collections:  # one untimed pass each, so the first timed searches aren't slower
    latency_ms(name)

print(f"{'Setting':<30} {'recall@10':>9} {'p50 ms':>7} {'p95 ms':>7} {'vectors in RAM':>15}")
for label, name, search in settings_to_test:
    p50, p95 = latency_ms(name, **search)
    print(f"{label:<30} {recall_at_k(name, **search):>9.3f} {p50:>7.1f} {p95:>7.1f} {ram[name]:>12.2f} MB")

for name in collections:
    client.delete_collection(name)
