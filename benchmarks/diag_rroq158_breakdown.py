"""diag_rroq158_breakdown.py — per-stage timing breakdown of the rroq158
dispatch path on CPU and GPU.

Why this script exists
----------------------
The Phase-7 BEIR sweep recorded wrapper-included p95 latency. The rroq158
average CPU p95 is ~7.9x the FP16 CPU p95 (812 ms vs 103 ms on the BEIR-6
average), but the kernel-only Rust microbench at production K=8192 / B=2000
runs in 14 ms p50 (regime B, 8 workers).  Something between the kernel
and the timer is eating the rest.

This script attributes per-query latency to each logical stage of the
``_score_rroq158_cpu`` / ``score_rroq158_topk`` dispatch path:

  encode_query                   (FWHT + bit-plane packing)
  from_numpy_q                   (numpy -> torch wrapping for query tensors)
  build_padded_payload           (variable-T -> dense padded numpy build)
  index_select_x7                (gather candidate rows from padded payload)
  ascontig_x9                    (np.ascontiguousarray on each tensor before
                                  the Rust call)
  rust_kernel                    (latence_shard_engine.rroq158_score_batch)
  argpartition_topk              (top-k extraction on the python side)

Mirrors the structure of ``benchmarks/diag_rroq4_riem_breakdown.py`` so
the two codecs can be compared like-for-like.

Run::

    python benchmarks/diag_rroq158_breakdown.py
    python benchmarks/diag_rroq158_breakdown.py --device gpu
    python benchmarks/diag_rroq158_breakdown.py --n-docs 4000 --n-q-tok 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from colsearch._internal.inference.quantization.rroq158 import (
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    get_cached_fwht_rotator,
    pack_doc_codes_to_int32_words,
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
    BEIR-style padded payload, so the per-query dispatch matches what
    ``_score_rroq158_candidates`` produces in production."""
    rng = np.random.default_rng(seed)
    n_total = n_docs * n_d_tok
    tokens = rng.standard_normal((n_total, dim)).astype(np.float32)

    cfg = Rroq158Config(
        K=K, group_size=group_size,
        fit_sample_cap=min(n_total, 60_000),
        encode_chunk=min(n_total, 60_000),
        seed=seed,
    )
    enc = encode_rroq158(tokens, cfg)

    sign_words = pack_doc_codes_to_int32_words(enc.sign_plane)
    nz_words = pack_doc_codes_to_int32_words(enc.nonzero_plane)
    n_words = sign_words.shape[1]
    n_groups = enc.scales.shape[1]

    sign_doc = sign_words.reshape(n_docs, n_d_tok, n_words)
    nz_doc = nz_words.reshape(n_docs, n_d_tok, n_words)
    scl_doc = enc.scales.astype(np.float32).reshape(n_docs, n_d_tok, n_groups)
    cid_doc = enc.centroid_id.astype(np.int32).reshape(n_docs, n_d_tok)
    cos_doc = enc.cos_norm.astype(np.float32).reshape(n_docs, n_d_tok)
    sin_doc = enc.sin_norm.astype(np.float32).reshape(n_docs, n_d_tok)
    mask = np.ones((n_docs, n_d_tok), dtype=np.float32)

    rotator = get_cached_fwht_rotator(dim=dim, seed=enc.fwht_seed)
    return {
        "sign_t": torch.from_numpy(sign_doc).to(device),
        "nz_t": torch.from_numpy(nz_doc).to(device),
        "scl_t": torch.from_numpy(scl_doc).to(device),
        "cid_t": torch.from_numpy(cid_doc).to(device),
        "cos_t": torch.from_numpy(cos_doc).to(device),
        "sin_t": torch.from_numpy(sin_doc).to(device),
        "mask_t": torch.from_numpy(mask).to(device),
        "sign_np": sign_doc,
        "nz_np": nz_doc,
        "scl_np": scl_doc,
        "cid_np": cid_doc,
        "cos_np": cos_doc,
        "sin_np": sin_doc,
        "mask_np": mask,
        "centroids": torch.from_numpy(enc.centroids).to(device),
        "centroids_np": np.ascontiguousarray(enc.centroids, dtype=np.float32),
        "rotator": rotator,
        "fwht_seed": enc.fwht_seed,
        "n_words": n_words,
        "n_groups": n_groups,
        "K": K,
        "dim": dim,
        "group_size": group_size,
    }


def _profile_one_query_cpu(
    payload, query_np: np.ndarray, n_candidates: int, top_k: int,
    rng: np.random.Generator, n_threads: int | None,
):
    """Time each logical stage of one CPU query against the synthetic
    payload. Mirrors the BEIR sweep's ``_rroq158_score_candidates`` flow:
    encode_query -> torch.from_numpy(q*) -> torch.index_select(payload x7)
    -> score_rroq158_topk -> _score_rroq158_cpu (ascontig + Rust kernel +
    argpartition)."""
    import latence_shard_engine as eng
    rroq_fn = eng.rroq158_score_batch

    n_docs = payload["sign_t"].shape[0]
    cand_idx_np = rng.choice(n_docs, size=n_candidates, replace=False).astype(np.int64)

    centroids_np = payload["centroids_np"]
    fwht_seed = payload["fwht_seed"]
    rotator = payload["rotator"]
    use_torch_gather = bool(payload.get("use_torch_gather", False))

    stages: dict[str, float] = {}

    t = time.perf_counter()
    q_inputs = encode_query_for_rroq158(
        query_np, centroids_np,
        fwht_seed=fwht_seed, query_bits=4, rotator=rotator,
    )
    stages["encode_query"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    qp_t = torch.from_numpy(q_inputs["q_planes"][None, :, :, :])
    qm_t = torch.from_numpy(q_inputs["q_meta"][None, :, :])
    qct_t = torch.from_numpy(q_inputs["qc_table"][None, :, :])
    stages["from_numpy_q"] = (time.perf_counter() - t) * 1000.0

    if use_torch_gather:
        # Pre-fix path: torch.index_select on torch.from_numpy-backed
        # CPU tensors. Recorded for before/after comparison only.
        cand_idx = torch.from_numpy(cand_idx_np)
        t = time.perf_counter()
        sign_b = payload["sign_t"].index_select(0, cand_idx)
        nz_b = payload["nz_t"].index_select(0, cand_idx)
        scl_b = payload["scl_t"].index_select(0, cand_idx)
        cid_b = payload["cid_t"].index_select(0, cand_idx)
        cos_b = payload["cos_t"].index_select(0, cand_idx)
        sin_b = payload["sin_t"].index_select(0, cand_idx)
        mask_b = payload["mask_t"].index_select(0, cand_idx)
        stages["index_select_x7"] = (time.perf_counter() - t) * 1000.0
    else:
        # Post-fix path: numpy fancy indexing on the underlying numpy
        # storage (zero-copy view via torch.from_numpy at payload build
        # time, so payload[...].numpy() is itself a free view).
        sign_np_full = payload["sign_np"]
        nz_np_full = payload["nz_np"]
        scl_np_full = payload["scl_np"]
        cid_np_full = payload["cid_np"]
        cos_np_full = payload["cos_np"]
        sin_np_full = payload["sin_np"]
        mask_np_full = payload["mask_np"]
        t = time.perf_counter()
        sign_b = torch.from_numpy(sign_np_full[cand_idx_np])
        nz_b = torch.from_numpy(nz_np_full[cand_idx_np])
        scl_b = torch.from_numpy(scl_np_full[cand_idx_np])
        cid_b = torch.from_numpy(cid_np_full[cand_idx_np])
        cos_b = torch.from_numpy(cos_np_full[cand_idx_np])
        sin_b = torch.from_numpy(sin_np_full[cand_idx_np])
        mask_b = torch.from_numpy(mask_np_full[cand_idx_np])
        stages["numpy_gather_x7"] = (time.perf_counter() - t) * 1000.0

    # _score_rroq158_cpu's _to_np with the new zero-copy fast path: when
    # the tensor is CPU + contiguous + matching dtype it returns
    # ``t.numpy()`` directly with no allocation. Keeps the ascontig
    # branch around for any non-fast-path tensors.
    t = time.perf_counter()
    qp_np = np.ascontiguousarray(qp_t.detach().cpu().numpy(), dtype=np.int32)
    qm_np = np.ascontiguousarray(qm_t.detach().cpu().numpy(), dtype=np.float32)
    qc_np = np.ascontiguousarray(qct_t.detach().cpu().numpy(), dtype=np.float32)
    cid_np = np.ascontiguousarray(cid_b.detach().cpu().numpy(), dtype=np.int32)
    cos_np = np.ascontiguousarray(cos_b.detach().cpu().numpy(), dtype=np.float32)
    sin_np = np.ascontiguousarray(sin_b.detach().cpu().numpy(), dtype=np.float32)
    sg_np = np.ascontiguousarray(sign_b.detach().cpu().numpy(), dtype=np.int32)
    nz_np = np.ascontiguousarray(nz_b.detach().cpu().numpy(), dtype=np.int32)
    scl_np = np.ascontiguousarray(scl_b.detach().cpu().numpy(), dtype=np.float32)
    d_mask_np = np.ascontiguousarray(mask_b.detach().cpu().numpy(), dtype=np.float32).ravel()
    stages["ascontig_x10"] = (time.perf_counter() - t) * 1000.0

    A_, S, query_bits, n_words = qp_np.shape
    B, T = sg_np.shape[:2]
    n_groups = scl_np.shape[-1]
    K = qc_np.shape[-1]

    t = time.perf_counter()
    flat = rroq_fn(
        qp_np.ravel(),
        qm_np.ravel(),
        qc_np.ravel(),
        sg_np.ravel(),
        nz_np.ravel(),
        scl_np.ravel(),
        cid_np.ravel(),
        cos_np.ravel(),
        sin_np.ravel(),
        A_, B, S, T, n_words, n_groups, query_bits, K,
        None,
        d_mask_np,
        n_threads=n_threads,
    )
    stages["rust_kernel"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    scores = flat.reshape(A_, B)[0]
    final_k = min(top_k, B)
    top_idx = np.argpartition(-scores, final_k - 1)[:final_k]
    _ = top_idx[np.argsort(-scores[top_idx])]
    stages["argpartition_topk"] = (time.perf_counter() - t) * 1000.0

    return stages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", choices=["cpu"], default="cpu",
                   help="rroq158 GPU dispatch is fast (~5 ms p95) and not "
                        "the bottleneck — CPU is the focus of this audit.")
    p.add_argument("--n-docs", type=int, default=2000,
                   help="Number of synthetic docs in the payload.")
    p.add_argument("--n-d-tok", type=int, default=32, help="Doc tokens per doc.")
    p.add_argument("--n-q-tok", type=int, default=32, help="Query tokens per query.")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--n-candidates", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n-threads", type=int, default=None,
                   help="Cap rayon pool used inside the Rust kernel. "
                        "Default = let rayon use all cores. Production CPU "
                        "lane uses cpu_count // n_workers (~16 on a 128-core "
                        "box with 8 python workers).")
    p.add_argument("--use-torch-gather", action="store_true",
                   help="Use torch.index_select for the candidate gather "
                        "(pre-fix path; default is numpy fancy indexing).")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="If set, calls torch.set_num_threads(N) at startup. "
                        "Defaults to leaving torch on its automatic "
                        "cpu_count // 2 setting. Set to 1 to verify the "
                        "torch intra-op pool isn't fighting with rayon.")
    p.add_argument("--out", type=Path,
                   default=Path("reports/diag_rroq158_breakdown.json"))
    args = p.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        print(f"[diag] torch.set_num_threads({args.torch_threads})")

    args.n_candidates = min(args.n_candidates, args.n_docs)

    rng = np.random.default_rng(0)
    queries_np = rng.standard_normal(
        (args.n_iters + args.n_warmup, args.n_q_tok, args.dim),
    ).astype(np.float32)

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
            "n_threads": args.n_threads,
        },
    }

    print(f"[diag] cpu: building payload "
          f"({args.n_docs} docs x {args.n_d_tok} tok x dim={args.dim}, K={args.K})")
    payload = _build_payload(
        args.n_docs, args.n_d_tok, args.dim, args.K, args.group_size, "cpu",
    )
    payload["use_torch_gather"] = args.use_torch_gather
    prof = _profile_one_query_cpu

    for i in range(args.n_warmup):
        prof(payload, queries_np[i], args.n_candidates, args.top_k, rng,
             args.n_threads)

    per_stage_runs: dict[str, list[float]] = {}
    per_query_total_ms: list[float] = []
    for i in range(args.n_iters):
        stages = prof(
            payload, queries_np[args.n_warmup + i],
            args.n_candidates, args.top_k, rng, args.n_threads,
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
    out["cpu"] = device_summary

    print(f"[diag] cpu (n_threads={args.n_threads}): total "
          f"p50={device_summary['p50_total_ms']:.2f}ms "
          f"p95={device_summary['p95_total_ms']:.2f}ms")
    for name, s in device_summary["stages"].items():
        print(f"           {name:22s}  p50={s['p50_ms']:7.3f}ms "
              f"({s['share_of_p50_total']*100:5.1f}%)  p95={s['p95_ms']:7.3f}ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
