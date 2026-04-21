"""
Phase 1.5(5) gate microbench: rroq158 CPU vs fp16 CPU at production shapes.

Measures the *kernel-only* p50/p95 cost of MaxSim scoring at the candidate-pool
shapes we actually use in production:

    K=8192, dim=128, group_size=32, n_q_tok=8, n_d_tok=32, B=2000

Compares:
  1. Rust SIMD `latence_shard_engine.rroq158_score_batch` (the production CPU lane)
  2. Numpy fp32 MaxSim (faithful proxy for the fp16 CPU lane: in
     `brute_force_maxsim`, fp16 doc vectors are cast to fp32 before matmul, so
     the matmul cost is fp32 either way)

Two concurrency regimes:
  - 1 worker, no thread cap (default rayon)
  - 8 workers, n_threads = max(1, cpu_count // 8) per worker (production layout)

Writes `reports/phase15_gate_cpu.json` and prints a verdict for the gate:
"rroq158 CPU p95 <= 1.5x fp16 CPU p95".
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import latence_shard_engine as eng
from colsearch._internal.inference.quantization.rroq158 import (
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)


SHAPE = dict(
    K=8192,
    dim=128,
    group_size=32,
    n_q_tok=8,
    n_d_tok=32,
    n_docs=2000,
    fit_sample_cap=20_000,
    encode_chunk=20_000,
    seed=0,
    n_iters=30,
    n_warmup=5,
)


def _build_rroq158_inputs(shape):
    rng = np.random.default_rng(shape["seed"])
    n_total_d = shape["n_docs"] * shape["n_d_tok"]
    docs = rng.standard_normal((n_total_d, shape["dim"])).astype(np.float32)
    queries = rng.standard_normal((shape["n_q_tok"], shape["dim"])).astype(np.float32)

    cfg = Rroq158Config(
        K=shape["K"],
        group_size=shape["group_size"],
        fit_sample_cap=shape["fit_sample_cap"],
        encode_chunk=shape["encode_chunk"],
        seed=shape["seed"],
    )
    enc = encode_rroq158(docs, cfg)
    q_in = encode_query_for_rroq158(
        queries, enc.centroids, fwht_seed=enc.fwht_seed, query_bits=4,
    )

    sign_words = pack_doc_codes_to_int32_words(enc.sign_plane)
    nz_words = pack_doc_codes_to_int32_words(enc.nonzero_plane)
    n_words = sign_words.shape[1]
    n_groups = enc.scales.shape[1]

    sign_doc = sign_words.reshape(shape["n_docs"], shape["n_d_tok"], n_words)
    nz_doc = nz_words.reshape(shape["n_docs"], shape["n_d_tok"], n_words)
    scales_doc = enc.scales.astype(np.float32).reshape(
        shape["n_docs"], shape["n_d_tok"], n_groups,
    )
    cid_doc = enc.centroid_id.astype(np.int32).reshape(
        shape["n_docs"], shape["n_d_tok"],
    )
    cos_doc = enc.cos_norm.astype(np.float32).reshape(
        shape["n_docs"], shape["n_d_tok"],
    )
    sin_doc = enc.sin_norm.astype(np.float32).reshape(
        shape["n_docs"], shape["n_d_tok"],
    )

    qp = q_in["q_planes"][None, :, :, :]
    qm = q_in["q_meta"][None, :, :]
    qc = q_in["qc_table"][None, :, :]

    payload = dict(
        qp=qp.astype(np.int32, copy=False).ravel(),
        qm=qm.astype(np.float32, copy=False).ravel(),
        qc=qc.astype(np.float32, copy=False).ravel(),
        sg=sign_doc.astype(np.int32, copy=False).ravel(),
        nz=nz_doc.astype(np.int32, copy=False).ravel(),
        sc=scales_doc.astype(np.float32, copy=False).ravel(),
        ci=cid_doc.astype(np.int32, copy=False).ravel(),
        co=cos_doc.astype(np.float32, copy=False).ravel(),
        si=sin_doc.astype(np.float32, copy=False).ravel(),
        A=qp.shape[0],
        B=sign_doc.shape[0],
        S=qp.shape[1],
        T=sign_doc.shape[1],
        n_words=n_words,
        n_groups=n_groups,
        query_bits=qp.shape[2],
        K=qc.shape[-1],
    )
    return payload, docs, queries


def _build_fp16_inputs(shape):
    """fp16 lane proxy: docs as fp32 (cast from fp16), queries as fp32, masks."""
    rng = np.random.default_rng(shape["seed"] + 7)
    docs = rng.standard_normal(
        (shape["n_docs"], shape["n_d_tok"], shape["dim"]),
    ).astype(np.float16).astype(np.float32)
    queries = rng.standard_normal(
        (shape["n_q_tok"], shape["dim"]),
    ).astype(np.float32)
    mask = np.ones((shape["n_docs"], shape["n_d_tok"]), dtype=np.float32)
    return docs, queries, mask


def _fp16_maxsim_numpy(docs, queries, mask):
    """Reference fp16-lane MaxSim. Cost-equivalent to the CPU fp16 path:
    one fp32 matmul + masked max + sum. Single-threaded numpy path is what
    `brute_force_maxsim` reduces to on CPU device='cpu'.
    """
    sims = np.einsum("sd,btd->sbt", queries, docs, optimize=True)
    sims = sims + (mask[None, :, :] - 1.0) * 1e9
    per_qd = sims.max(axis=2)
    return per_qd.sum(axis=0)


def _bench_callable(fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        fn()
    times_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return {
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95)],
        "mean_ms": float(np.mean(times_ms)),
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "n": len(times_ms),
    }


def _bench_concurrent(fn, n_warmup, n_iters, n_workers):
    for _ in range(n_warmup):
        fn()

    def _one(_):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_one, i) for i in range(n_iters)]
        times_ms = [f.result() for f in as_completed(futs)]
    times_ms.sort()
    return {
        "p50_ms": times_ms[len(times_ms) // 2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95)],
        "mean_ms": float(np.mean(times_ms)),
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "n": len(times_ms),
        "n_workers": n_workers,
    }


def main():
    shape = SHAPE.copy()
    n_iters = shape.pop("n_iters")
    n_warmup = shape.pop("n_warmup")
    print(f"[microbench] building inputs at shape K={shape['K']} dim={shape['dim']} "
          f"S={shape['n_q_tok']} T={shape['n_d_tok']} B={shape['n_docs']} ...")

    rroq_p, _docs_full, _q = _build_rroq158_inputs(shape)
    fp_docs, fp_q, fp_mask = _build_fp16_inputs(shape)

    def call_rroq(n_threads=None):
        kwargs = {"n_threads": n_threads} if n_threads is not None else {}
        eng.rroq158_score_batch(
            rroq_p["qp"], rroq_p["qm"], rroq_p["qc"],
            rroq_p["sg"], rroq_p["nz"], rroq_p["sc"],
            rroq_p["ci"], rroq_p["co"], rroq_p["si"],
            rroq_p["A"], rroq_p["B"], rroq_p["S"], rroq_p["T"],
            rroq_p["n_words"], rroq_p["n_groups"],
            rroq_p["query_bits"], rroq_p["K"],
            **kwargs,
        )

    def call_fp16():
        _fp16_maxsim_numpy(fp_docs, fp_q, fp_mask)

    cpu = os.cpu_count() or 8
    n_workers = 8
    threads_per_worker = max(1, cpu // n_workers)

    print(f"[microbench] cpu_count={cpu}  n_workers={n_workers}  "
          f"threads_per_worker={threads_per_worker}")

    print("[microbench] regime A: 1 worker, default rayon ...")
    a_rroq = _bench_callable(lambda: call_rroq(None), n_warmup, n_iters)
    a_fp16 = _bench_callable(call_fp16, n_warmup, n_iters)

    print("[microbench] regime B: 8 workers, threads_per_worker rayon ...")
    b_rroq = _bench_concurrent(
        lambda: call_rroq(threads_per_worker), n_warmup, n_iters, n_workers,
    )
    b_fp16 = _bench_concurrent(call_fp16, n_warmup, n_iters, n_workers)

    out = {
        "shape": {**shape, "n_iters": n_iters, "n_warmup": n_warmup},
        "regimeA_1w_default": {"rroq158": a_rroq, "fp16": a_fp16},
        "regimeB_8w_capped": {"rroq158": b_rroq, "fp16": b_fp16},
    }

    def ratio(rroq, fp): return rroq["p95_ms"] / max(fp["p95_ms"], 1e-9)
    out["gate"] = {
        "regimeA_p95_ratio": ratio(a_rroq, a_fp16),
        "regimeB_p95_ratio": ratio(b_rroq, b_fp16),
        "regimeB_p50_ratio": b_rroq["p50_ms"] / max(b_fp16["p50_ms"], 1e-9),
        "threshold": 1.5,
        "regimeA_pass": ratio(a_rroq, a_fp16) <= 1.5,
        "regimeB_pass": ratio(b_rroq, b_fp16) <= 1.5,
        "decision": "FLIP_CPU_DEFAULT"
            if (ratio(b_rroq, b_fp16) <= 1.5)
            else "KEEP_CPU_FP16",
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/phase15_gate_cpu.json").write_text(json.dumps(out, indent=2))

    def fmt(d): return f"p50={d['p50_ms']:.2f}ms p95={d['p95_ms']:.2f}ms"
    print()
    print("=" * 80)
    print("REGIME A (1 worker, default rayon)")
    print(f"  rroq158: {fmt(a_rroq)}")
    print(f"  fp16   : {fmt(a_fp16)}")
    print(f"  p95 ratio rroq158/fp16 = {out['gate']['regimeA_p95_ratio']:.2f}x")
    print()
    print(f"REGIME B ({n_workers} workers, n_threads={threads_per_worker})")
    print(f"  rroq158: {fmt(b_rroq)}")
    print(f"  fp16   : {fmt(b_fp16)}")
    print(f"  p95 ratio rroq158/fp16 = {out['gate']['regimeB_p95_ratio']:.2f}x")
    print(f"  p50 ratio rroq158/fp16 = {out['gate']['regimeB_p50_ratio']:.2f}x")
    print()
    print("GATE THRESHOLD: rroq158 p95 <= 1.5x fp16 p95")
    print(f"DECISION:       {out['gate']['decision']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
