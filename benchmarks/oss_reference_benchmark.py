from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voyager_index import TRITON_AVAILABLE, ColPaliConfig, ColPaliEngine, fast_colbert_scores
from voyager_index.server import create_app


def benchmark_maxsim(device: str, queries: int, docs: int, tokens: int, dim: int) -> dict:
    torch.manual_seed(7)
    q = torch.randn((queries, tokens, dim), device=device, dtype=torch.float32)
    d = torch.randn((docs, tokens, dim), device=device, dtype=torch.float32)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    scores = fast_colbert_scores(q, d)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "benchmark": "maxsim",
        "device": device,
        "queries": queries,
        "docs": docs,
        "tokens": tokens,
        "dim": dim,
        "elapsed_ms": elapsed_ms,
        "scores_shape": list(scores.shape),
    }


def _reference_maxsim(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    similarities = torch.einsum("ash,bth->abst", queries_embeddings, documents_embeddings)
    if documents_mask is not None:
        similarities = similarities.masked_fill(
            ~documents_mask.to(dtype=torch.bool, device=similarities.device)[None, :, None, :],
            float("-inf"),
        )
    max_sim = similarities.max(dim=-1).values
    if queries_mask is not None:
        max_sim = max_sim * queries_mask.to(dtype=max_sim.dtype, device=max_sim.device)[:, None, :]
    return torch.nan_to_num(max_sim, neginf=0.0).sum(dim=-1)


def _timed_score(
    fn,
    *,
    synchronize_device: str | None = None,
) -> tuple[torch.Tensor, float]:
    if synchronize_device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    scores = fn()
    if synchronize_device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return scores, elapsed_ms


def _benchmark_runs(
    fn,
    *,
    synchronize_device: str | None = None,
    warmup_runs: int = 1,
    measure_runs: int = 5,
) -> tuple[torch.Tensor, dict]:
    last_scores = None
    warmup_elapsed = []
    for _ in range(max(warmup_runs, 0)):
        last_scores, elapsed_ms = _timed_score(fn, synchronize_device=synchronize_device)
        warmup_elapsed.append(elapsed_ms)

    measured = []
    for _ in range(max(measure_runs, 1)):
        last_scores, elapsed_ms = _timed_score(fn, synchronize_device=synchronize_device)
        measured.append(elapsed_ms)

    assert last_scores is not None
    return last_scores, {
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "first_warmup_ms": warmup_elapsed[0] if warmup_elapsed else None,
        "median_ms": float(statistics.median(measured)),
        "min_ms": float(min(measured)),
        "max_ms": float(max(measured)),
        "runs_ms": measured,
    }


def benchmark_maxsim_tensors(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    *,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    processor_score_fn=None,
) -> dict:
    q_cpu = queries_embeddings.detach().to("cpu", dtype=torch.float32)
    d_cpu = documents_embeddings.detach().to("cpu", dtype=torch.float32)
    qm_cpu = queries_mask.detach().to("cpu", dtype=torch.float32) if queries_mask is not None else None
    dm_cpu = documents_mask.detach().to("cpu", dtype=torch.float32) if documents_mask is not None else None

    reference_scores, reference_bench = _benchmark_runs(
        lambda: _reference_maxsim(q_cpu, d_cpu, queries_mask=qm_cpu, documents_mask=dm_cpu),
        warmup_runs=0,
        measure_runs=3,
    )
    cpu_scores, cpu_bench = _benchmark_runs(
        lambda: fast_colbert_scores(q_cpu, d_cpu, queries_mask=qm_cpu, documents_mask=dm_cpu),
        warmup_runs=1,
        measure_runs=5,
    )

    result = {
        "benchmark": "maxsim_real_tensors",
        "queries": int(q_cpu.shape[0]),
        "docs": int(d_cpu.shape[0]),
        "query_tokens": int(q_cpu.shape[1]),
        "doc_tokens": int(d_cpu.shape[1]),
        "dim": int(q_cpu.shape[2]),
        "reference_cpu": {
            "elapsed_ms": reference_bench["median_ms"],
            "benchmark": reference_bench,
            "scores_shape": list(reference_scores.shape),
        },
        "fast_colbert_cpu": {
            "elapsed_ms": cpu_bench["median_ms"],
            "benchmark": cpu_bench,
            "max_abs_delta_vs_reference": float(torch.max(torch.abs(cpu_scores - reference_scores)).item()),
            "parity": bool(torch.allclose(cpu_scores, reference_scores, atol=1e-5, rtol=1e-5)),
        },
        "triton_cuda": {
            "status": "skipped",
            "reason": "cuda_or_triton_unavailable",
        },
    }

    if processor_score_fn is not None:
        processor_scores, processor_bench = _benchmark_runs(
            lambda: processor_score_fn(queries_embeddings, documents_embeddings),
            synchronize_device="cuda" if queries_embeddings.device.type == "cuda" else None,
            warmup_runs=1,
            measure_runs=5,
        )
        processor_scores_cpu = processor_scores.detach().to("cpu", dtype=torch.float32)
        result["processor"] = {
            "elapsed_ms": processor_bench["median_ms"],
            "benchmark": processor_bench,
            "scores_shape": list(processor_scores_cpu.shape),
            "max_abs_delta_vs_reference": float(torch.max(torch.abs(processor_scores_cpu - reference_scores)).item()),
        }

    if (
        TRITON_AVAILABLE
        and torch.cuda.is_available()
        and queries_embeddings.device.type == "cuda"
        and documents_embeddings.device.type == "cuda"
    ):
        qm_cuda = (
            queries_mask.to(device=queries_embeddings.device, dtype=torch.float32) if queries_mask is not None else None
        )
        dm_cuda = (
            documents_mask.to(device=documents_embeddings.device, dtype=torch.float32)
            if documents_mask is not None
            else None
        )
        triton_scores, triton_bench = _benchmark_runs(
            lambda: fast_colbert_scores(
                queries_embeddings,
                documents_embeddings,
                queries_mask=qm_cuda,
                documents_mask=dm_cuda,
            ),
            synchronize_device="cuda",
            warmup_runs=1,
            measure_runs=5,
        )
        triton_scores_cpu = triton_scores.detach().to("cpu", dtype=torch.float32)
        result["triton_cuda"] = {
            "status": "passed",
            "elapsed_ms": triton_bench["median_ms"],
            "benchmark": triton_bench,
            "max_abs_delta_vs_reference": float(torch.max(torch.abs(triton_scores_cpu - reference_scores)).item()),
            "parity": bool(torch.allclose(triton_scores_cpu, reference_scores, atol=5e-4, rtol=1e-3)),
        }

    return result


def benchmark_reference_api_ingest(index_path: Path, points: int) -> dict:
    app = create_app(index_path=str(index_path))
    with TestClient(app) as client:
        client.post("/collections/dense", json={"dimension": 16, "kind": "dense"})
        start = time.perf_counter()
        client.post(
            "/collections/dense/points",
            json={
                "points": [
                    {
                        "id": f"doc-{idx}",
                        "vector": [1.0 if i == (idx % 16) else 0.0 for i in range(16)],
                        "payload": {"text": f"document {idx}"},
                    }
                    for idx in range(points)
                ]
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "benchmark": "reference_api_ingest",
        "points": points,
        "elapsed_ms": elapsed_ms,
    }


def benchmark_reference_api_search(index_path: Path, top_k: int) -> dict:
    app = create_app(index_path=str(index_path))
    with TestClient(app) as client:
        client.post("/collections/dense", json={"dimension": 16, "kind": "dense"})
        client.post(
            "/collections/dense/points",
            json={
                "points": [
                    {
                        "id": f"doc-{idx}",
                        "vector": [1.0 if i == (idx % 16) else 0.0 for i in range(16)],
                        "payload": {"text": f"document {idx}"},
                    }
                    for idx in range(32)
                ]
            },
        )
        start = time.perf_counter()
        response = client.post(
            "/collections/dense/search",
            json={"vector": [1.0] + [0.0] * 15, "top_k": top_k},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.raise_for_status()
        payload = response.json()

    return {
        "benchmark": "reference_api_search",
        "top_k": top_k,
        "elapsed_ms": elapsed_ms,
        "first_result": payload["results"][0]["id"],
        "result_count": payload["total"],
    }


def benchmark_multimodal_search(index_path: Path, top_k: int) -> dict:
    engine = ColPaliEngine(
        index_path / "colpali",
        config=ColPaliConfig(embed_dim=16, device="cpu", use_quantization=False),
        device="cpu",
        load_if_exists=False,
    )
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(32, 8, 16)).astype(np.float32)
    query = embeddings[0]
    engine.add_documents(embeddings, doc_ids=[f"page-{idx}" for idx in range(32)])
    start = time.perf_counter()
    results = engine.search(query_embedding=query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "benchmark": "multimodal_search",
        "top_k": top_k,
        "elapsed_ms": elapsed_ms,
        "first_result": results[0].doc_id,
        "result_count": len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small reproducible OSS benchmarks.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--docs", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--points", type=int, default=64)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        results = {
            "maxsim": benchmark_maxsim(args.device, args.queries, args.docs, args.tokens, args.dim),
            "reference_api_ingest": benchmark_reference_api_ingest(Path(tmpdir) / "ingest", args.points),
            "reference_api_search": benchmark_reference_api_search(Path(tmpdir) / "search", args.top_k),
            "multimodal_search": benchmark_multimodal_search(Path(tmpdir) / "mm", args.top_k),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
