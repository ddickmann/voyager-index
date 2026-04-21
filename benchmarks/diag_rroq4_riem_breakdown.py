"""diag_rroq4_riem_breakdown.py — per-stage timing breakdown of the
rroq4_riem dispatch path on CPU and GPU.

Why this script exists
----------------------
The Phase-7 BEIR sweep recorded **wrapper-included** p95 latency:
the timer started at ``query_np`` and stopped after top-k IDs were
returned. That stack contains a lot more than the kernel call:

  query_np
    -> encode_query_for_rroq4_riem  (FWHT + q_group_sums)
    -> q @ centroids.T              (qc_table GEMM, K=8192)
    -> 6x torch.index_select        (gather padded payload by candidates)
    -> np.ascontiguousarray x9      (force contiguous fp32/uint8/i32 copies)
    -> latence_shard_engine.rroq4_riem_score_batch  (the actual kernel)
    -> np.argpartition + sort        (top-k on the python side)

The microbenches in ``tests/test_rroq4_riem_kernel.py::*microbench`` time
**only** the kernel call. To know whether it's worth optimizing the kernel
or the wrapper first, we need the per-stage split. That's what this script
emits to ``reports/diag_rroq4_riem_breakdown.json``.

Run::

    python benchmarks/diag_rroq4_riem_breakdown.py
    python benchmarks/diag_rroq4_riem_breakdown.py --device gpu
    python benchmarks/diag_rroq4_riem_breakdown.py --n-docs 4000 --n-q-tok 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from colsearch._internal.inference.quantization.rroq4_riem import (
    Rroq4RiemConfig,
    encode_query_for_rroq4_riem,
    encode_rroq4_riem,
    get_cached_fwht_rotator,
)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def _build_payload(
    n_docs: int,
    n_d_tok: int,
    dim: int,
    K: int,
    group_size: int,
    device: str,
    seed: int = 0,
):
    """Encode a synthetic corpus and return tensors shaped like a real
    BEIR-style payload (the same shape ``_build_rroq4_riem_payload`` in
    ``beir_benchmark.py`` produces, with ``T == n_d_tok`` for all docs)."""
    rng = np.random.default_rng(seed)
    n_total = n_docs * n_d_tok
    tokens = rng.standard_normal((n_total, dim)).astype(np.float32)

    cfg = Rroq4RiemConfig(
        K=K, group_size=group_size,
        fit_sample_cap=min(n_total, 60_000),
        encode_chunk=min(n_total, 60_000),
        seed=seed,
    )
    enc = encode_rroq4_riem(tokens, cfg)

    n_groups = enc.mins.shape[1]
    n_bytes = dim // 2

    codes = enc.codes_packed.reshape(n_docs, n_d_tok, n_bytes)
    mins = enc.mins.astype(np.float16).reshape(n_docs, n_d_tok, n_groups)
    deltas = enc.deltas.astype(np.float16).reshape(n_docs, n_d_tok, n_groups)
    cid = enc.centroid_id.astype(np.int32).reshape(n_docs, n_d_tok)
    cosn = enc.cos_norm.astype(np.float16).reshape(n_docs, n_d_tok)
    sinn = enc.sin_norm.astype(np.float16).reshape(n_docs, n_d_tok)
    mask = np.ones((n_docs, n_d_tok), dtype=np.float32)

    rotator = get_cached_fwht_rotator(dim=dim, seed=enc.fwht_seed)
    return {
        "cid": torch.from_numpy(cid).to(device),
        "cosn": torch.from_numpy(cosn).to(device),
        "sinn": torch.from_numpy(sinn).to(device),
        "codes": torch.from_numpy(codes).to(device),
        "mins": torch.from_numpy(mins).to(device),
        "deltas": torch.from_numpy(deltas).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "centroids": torch.from_numpy(enc.centroids).to(device),
        "centroids_np": np.ascontiguousarray(enc.centroids, dtype=np.float32),
        "rotator": rotator,
        "fwht_seed": enc.fwht_seed,
        "group_size": group_size,
        "n_groups": n_groups,
        "K": K,
        "dim": dim,
    }


def _stage_timer():
    """Return (mark, dump) — call ``mark(name)`` between stages and
    ``dump()`` to get the per-stage delta in milliseconds. We synchronize
    CUDA before each mark so GPU stage times are accurate."""
    on_cuda = torch.cuda.is_available()
    samples: dict[str, float] = {}
    last = [None]

    def _now():
        if on_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def mark(name: str | None):
        now = _now()
        if last[0] is not None and name is not None:
            samples.setdefault(name, 0.0)
            samples[name] += (now - last[0]) * 1000.0
        last[0] = now if name is not None else None

    def reset():
        samples.clear()
        last[0] = _now()

    def snapshot():
        out = dict(samples)
        samples.clear()
        last[0] = None
        return out

    return mark, reset, snapshot


def _profile_one_query_cpu(
    payload, query_np: np.ndarray, n_candidates: int, top_k: int,
    rng: np.random.Generator,
):
    """Time each stage of one CPU query against the synthetic payload."""
    import latence_shard_engine as eng
    rroq4_fn = eng.rroq4_riem_score_batch

    n_docs = payload["cid"].shape[0]
    cand_idx_np = rng.choice(n_docs, size=n_candidates, replace=False).astype(np.int64)
    cand_idx = torch.from_numpy(cand_idx_np)

    centroids_np = payload["centroids_np"]
    fwht_seed = payload["fwht_seed"]
    group_size = payload["group_size"]
    rotator = payload["rotator"]

    stages: dict[str, float] = {}

    t = time.perf_counter()
    q_inputs = encode_query_for_rroq4_riem(
        query_np, centroids_np, fwht_seed=fwht_seed,
        group_size=group_size, rotator=rotator,
    )
    stages["encode_query"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    q_rot = torch.from_numpy(q_inputs["q_rot"][None, :, :])
    q_gs = torch.from_numpy(q_inputs["q_group_sums"][None, :, :])
    qc_table = torch.from_numpy(q_inputs["qc_table"][None, :, :])
    stages["from_numpy_q"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    cid_b = payload["cid"].index_select(0, cand_idx)
    cosn_b = payload["cosn"].index_select(0, cand_idx)
    sinn_b = payload["sinn"].index_select(0, cand_idx)
    codes_b = payload["codes"].index_select(0, cand_idx)
    mins_b = payload["mins"].index_select(0, cand_idx)
    deltas_b = payload["deltas"].index_select(0, cand_idx)
    mask_b = payload["mask"].index_select(0, cand_idx)
    stages["index_select_x7"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    qrot_np = np.ascontiguousarray(q_rot.detach().cpu().numpy(), dtype=np.float32)
    qgs_np = np.ascontiguousarray(q_gs.detach().cpu().numpy(), dtype=np.float32)
    qc_np = np.ascontiguousarray(qc_table.detach().cpu().numpy(), dtype=np.float32)
    cid_np = np.ascontiguousarray(cid_b.detach().cpu().numpy(), dtype=np.int32)
    cos_np = np.ascontiguousarray(cosn_b.detach().cpu().numpy(), dtype=np.float32)
    sin_np = np.ascontiguousarray(sinn_b.detach().cpu().numpy(), dtype=np.float32)
    codes_np = np.ascontiguousarray(codes_b.detach().cpu().numpy(), dtype=np.uint8)
    mins_np = np.ascontiguousarray(mins_b.detach().cpu().numpy(), dtype=np.float32)
    dlts_np = np.ascontiguousarray(deltas_b.detach().cpu().numpy(), dtype=np.float32)
    mask_np = np.ascontiguousarray(mask_b.detach().cpu().numpy(), dtype=np.float32).ravel()
    stages["ascontig_x10"] = (time.perf_counter() - t) * 1000.0

    A_, S, dim = qrot_np.shape
    B, T = codes_np.shape[:2]
    n_groups = qgs_np.shape[-1]
    K = qc_np.shape[-1]

    t = time.perf_counter()
    flat = rroq4_fn(
        qrot_np.ravel(), qgs_np.ravel(), qc_np.ravel(),
        codes_np.ravel(), mins_np.ravel(), dlts_np.ravel(),
        cid_np.ravel(), cos_np.ravel(), sin_np.ravel(),
        A_, B, S, T, dim, n_groups, group_size, K,
        None, mask_np,
    )
    stages["rust_kernel"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    scores = flat.reshape(A_, B)[0]
    final_k = min(top_k, B)
    top_idx = np.argpartition(-scores, final_k - 1)[:final_k]
    _ = top_idx[np.argsort(-scores[top_idx])]
    stages["argpartition_topk"] = (time.perf_counter() - t) * 1000.0

    return stages


def _profile_one_query_gpu(
    payload, query_np: np.ndarray, n_candidates: int, top_k: int,
    rng: np.random.Generator,
):
    """Time each stage of one GPU query against the synthetic payload."""
    from colsearch._internal.kernels.triton_roq_rroq4_riem import (
        roq_maxsim_rroq4_riem,
    )

    device = torch.device("cuda")
    n_docs = payload["cid"].shape[0]
    cand_idx_np = rng.choice(n_docs, size=n_candidates, replace=False).astype(np.int64)
    cand_idx = torch.from_numpy(cand_idx_np).to(device)

    fwht_seed = payload["fwht_seed"]
    group_size = payload["group_size"]
    rotator = payload["rotator"]

    stages: dict[str, float] = {}

    torch.cuda.synchronize()
    t = time.perf_counter()
    q_inputs = encode_query_for_rroq4_riem(
        query_np, None, fwht_seed=fwht_seed,
        group_size=group_size, rotator=rotator,
        skip_qc_table=True,
    )
    torch.cuda.synchronize()
    stages["encode_query"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    q_rot = torch.from_numpy(q_inputs["q_rot"][None, :, :]).to(device)
    q_gs = torch.from_numpy(q_inputs["q_group_sums"][None, :, :]).to(device)
    q_dev = torch.from_numpy(np.ascontiguousarray(query_np, dtype=np.float32)).to(device)
    torch.cuda.synchronize()
    stages["h2d_query"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    qc_table = (q_dev @ payload["centroids"].T).unsqueeze(0).to(torch.float32)
    torch.cuda.synchronize()
    stages["qc_table_gemm"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    cid_b = payload["cid"].index_select(0, cand_idx).to(torch.int32)
    cosn_b = payload["cosn"].index_select(0, cand_idx).to(torch.float32)
    sinn_b = payload["sinn"].index_select(0, cand_idx).to(torch.float32)
    codes_b = payload["codes"].index_select(0, cand_idx)
    mins_b = payload["mins"].index_select(0, cand_idx).to(torch.float32)
    deltas_b = payload["deltas"].index_select(0, cand_idx).to(torch.float32)
    mask_b = payload["mask"].index_select(0, cand_idx)
    torch.cuda.synchronize()
    stages["index_select_x7"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    scores = roq_maxsim_rroq4_riem(
        queries_rot=q_rot,
        queries_group_sums=q_gs,
        qc_table=qc_table,
        docs_centroid_id=cid_b,
        docs_cos_norm=cosn_b,
        docs_sin_norm=sinn_b,
        docs_codes_packed=codes_b if codes_b.dtype == torch.uint8 else codes_b.to(torch.uint8),
        docs_mins=mins_b,
        docs_deltas=deltas_b,
        documents_mask=mask_b,
        group_size=group_size,
    ).squeeze(0)
    torch.cuda.synchronize()
    stages["triton_kernel"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    final_k = min(top_k, scores.shape[0])
    top_sc, top_idx = scores.topk(final_k)
    _ = top_idx.cpu().tolist()
    _ = top_sc.cpu().tolist()
    torch.cuda.synchronize()
    stages["topk_d2h"] = (time.perf_counter() - t) * 1000.0

    return stages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", choices=["cpu", "gpu", "both"], default="both",
                   help="Profile CPU stack, GPU stack, or both.")
    p.add_argument("--n-docs", type=int, default=2000,
                   help="Number of synthetic docs in the payload "
                        "(matches max_docs_exact for arguana-class cells).")
    p.add_argument("--n-d-tok", type=int, default=32, help="Doc tokens per doc.")
    p.add_argument("--n-q-tok", type=int, default=32, help="Query tokens per query.")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--K", type=int, default=8192,
                   help="Centroid count (production default = 8192).")
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--n-candidates", type=int, default=2000,
                   help="How many docs to score per query (matches max_docs_exact).")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", type=Path,
                   default=Path("reports/diag_rroq4_riem_breakdown.json"))
    args = p.parse_args()

    args.n_candidates = min(args.n_candidates, args.n_docs)

    rng = np.random.default_rng(0)
    queries_np = rng.standard_normal((args.n_iters + args.n_warmup, args.n_q_tok, args.dim)).astype(np.float32)

    out: dict = {
        "shape": {
            "n_docs_payload": args.n_docs,
            "n_d_tok": args.n_d_tok,
            "n_q_tok": args.n_q_tok,
            "dim": args.dim,
            "K": args.K,
            "group_size": args.group_size,
            "n_candidates_per_query": args.n_candidates,
            "top_k": args.top_k,
            "n_warmup": args.n_warmup,
            "n_iters": args.n_iters,
        },
    }

    if args.device == "both":
        devices = ["cpu", "gpu"]
    else:
        devices = [args.device]

    for label in devices:
        if label == "gpu" and not torch.cuda.is_available():
            print(f"[diag] skipping {label} — CUDA not available")
            continue

        torch_device = "cuda" if label == "gpu" else "cpu"
        print(f"[diag] {label}: building payload "
              f"({args.n_docs} docs x {args.n_d_tok} tok x dim={args.dim}, K={args.K})")
        payload = _build_payload(
            args.n_docs, args.n_d_tok, args.dim, args.K, args.group_size, torch_device,
        )
        prof = _profile_one_query_cpu if label == "cpu" else _profile_one_query_gpu
        dev = label

        for i in range(args.n_warmup):
            prof(payload, queries_np[i], args.n_candidates, args.top_k, rng)

        per_stage_runs: dict[str, list[float]] = {}
        per_query_total_ms: list[float] = []
        for i in range(args.n_iters):
            stages = prof(
                payload, queries_np[args.n_warmup + i],
                args.n_candidates, args.top_k, rng,
            )
            total = sum(stages.values())
            per_query_total_ms.append(total)
            for name, v in stages.items():
                per_stage_runs.setdefault(name, []).append(v)

        device_summary: dict = {
            "p50_total_ms": _percentile(per_query_total_ms, 50),
            "p95_total_ms": _percentile(per_query_total_ms, 95),
            "stages": {},
        }
        for name, runs in per_stage_runs.items():
            p50 = _percentile(runs, 50)
            p95 = _percentile(runs, 95)
            share = p50 / device_summary["p50_total_ms"] if device_summary["p50_total_ms"] > 0 else 0.0
            device_summary["stages"][name] = {
                "p50_ms": p50,
                "p95_ms": p95,
                "share_of_p50_total": share,
            }
        out[dev] = device_summary

        print(f"[diag] {dev}: total p50={device_summary['p50_total_ms']:.2f}ms "
              f"p95={device_summary['p95_total_ms']:.2f}ms")
        for name, s in device_summary["stages"].items():
            print(f"           {name:20s}  p50={s['p50_ms']:7.3f}ms "
                  f"({s['share_of_p50_total']*100:5.1f}%)  p95={s['p95_ms']:7.3f}ms")

        del payload
        if label == "gpu":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
