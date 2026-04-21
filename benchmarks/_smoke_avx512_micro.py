"""CPU microbench: AVX-512 VPOPCNTDQ tier (X86V4) vs scalar popcntq tier (X86V3).

Production rroq158 shape (dim=128, n_groups=1, group_words=4, query_bits=4):
that's the only shape the V4 fast path specialises for. Other shapes
self-fall-back to V3 in the dispatcher, so we don't bench them here.

Workload: B=2000 candidate docs × T=32 doc-tokens × S=32 query-tokens.
This is the LEMUR post-routing candidate-pool shape used by the production
CPU path (`shard_engine._manager.search`).

Method:
  1. Generate synthetic rroq158-encoded docs + query.
  2. Force backend to "x86v3", run `rroq158_score_batch` 100 iter, measure p50/p95.
  3. Force backend to "x86v4" (i.e. "auto" on a host with VPOPCNTDQ), repeat.
  4. Verify scores match (bit-exact) between the two backends.
  5. Report speedup.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import latence_shard_engine as eng  # noqa: E402

from voyager_index._internal.inference.quantization.rroq158 import (  # noqa: E402
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)


SHAPE = dict(
    K=1024,
    dim=128,
    group_size=128,  # n_groups=1, group_words=4 → V4 fast path
    n_q_tok=32,
    n_d_tok=32,
    n_docs=2000,
    n_iters=100,
    n_warmup=20,
    seed=0,
)


def build_inputs():
    rng = np.random.default_rng(SHAPE["seed"])
    dim = SHAPE["dim"]
    K = SHAPE["K"]
    B = SHAPE["n_docs"]
    T = SHAPE["n_d_tok"]
    S = SHAPE["n_q_tok"]

    cfg = Rroq158Config(K=K, group_size=SHAPE["group_size"], seed=SHAPE["seed"])

    # Train a centroid table on a sample
    sample = rng.standard_normal((20_000, dim)).astype(np.float32)
    sample /= np.linalg.norm(sample, axis=1, keepdims=True) + 1e-8
    enc_sample = encode_rroq158(sample, cfg)
    centroids = enc_sample.centroids
    fwht_seed = enc_sample.fwht_seed

    # Encode B*T document-tokens at once
    all_doc = rng.standard_normal((B * T, dim)).astype(np.float32)
    all_doc /= np.linalg.norm(all_doc, axis=1, keepdims=True) + 1e-8
    cfg_d = Rroq158Config(
        K=K, group_size=SHAPE["group_size"], seed=SHAPE["seed"],
    )
    cfg_d.centroids = centroids
    cfg_d.fwht_seed = fwht_seed
    enc_d = encode_rroq158(all_doc, cfg_d)

    n_int32_words = enc_d.sign_plane.shape[1] // 4
    n_groups = enc_d.scales.shape[1]
    sign_words = pack_doc_codes_to_int32_words(enc_d.sign_plane)
    nz_words = pack_doc_codes_to_int32_words(enc_d.nonzero_plane)

    docs_sign = sign_words.reshape(B, T, n_int32_words).astype(np.int32)
    docs_nz = nz_words.reshape(B, T, n_int32_words).astype(np.int32)
    docs_scl = enc_d.scales.reshape(B, T, n_groups).astype(np.float32)
    docs_cid = enc_d.centroid_id.reshape(B, T).astype(np.int32)
    docs_cos = enc_d.cos_norm.reshape(B, T).astype(np.float32)
    docs_sin = enc_d.sin_norm.reshape(B, T).astype(np.float32)

    # Encode query
    q = rng.standard_normal((S, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q_inputs = encode_query_for_rroq158(
        q, centroids,
        fwht_seed=fwht_seed,
        query_bits=4,
        rotator=None,
        skip_qc_table=False,
        cap_blas_threads=False,
    )

    # All inputs as flat row-major
    return {
        "q_planes": q_inputs["q_planes"].astype(np.int32).reshape(-1),
        "q_meta": q_inputs["q_meta"].astype(np.float32).reshape(-1),
        "qc_table": q_inputs["qc_table"].astype(np.float32).reshape(-1),
        "docs_sign": docs_sign.reshape(-1),
        "docs_nz": docs_nz.reshape(-1),
        "docs_scl": docs_scl.reshape(-1),
        "docs_cid": docs_cid.reshape(-1),
        "docs_cos": docs_cos.reshape(-1),
        "docs_sin": docs_sin.reshape(-1),
        "big_a": 1,
        "big_b": B,
        "big_s": S,
        "big_t": T,
        "n_words": n_int32_words,
        "n_groups": n_groups,
        "query_bits": 4,
        "big_k": K,
    }


def bench(inputs: dict, backend: str, n_iters: int, n_warmup: int) -> dict:
    set_b = eng._rroq158_force_backend_for_tests
    set_b(backend)

    # warmup
    for _ in range(n_warmup):
        out = eng.rroq158_score_batch(
            inputs["q_planes"], inputs["q_meta"], inputs["qc_table"],
            inputs["docs_sign"], inputs["docs_nz"], inputs["docs_scl"],
            inputs["docs_cid"], inputs["docs_cos"], inputs["docs_sin"],
            inputs["big_a"], inputs["big_b"], inputs["big_s"], inputs["big_t"],
            inputs["n_words"], inputs["n_groups"], inputs["query_bits"],
            inputs["big_k"],
            n_threads=1,
        )

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = eng.rroq158_score_batch(
            inputs["q_planes"], inputs["q_meta"], inputs["qc_table"],
            inputs["docs_sign"], inputs["docs_nz"], inputs["docs_scl"],
            inputs["docs_cid"], inputs["docs_cos"], inputs["docs_sin"],
            inputs["big_a"], inputs["big_b"], inputs["big_s"], inputs["big_t"],
            inputs["n_words"], inputs["n_groups"], inputs["query_bits"],
            inputs["big_k"],
            n_threads=1,
        )
        times.append((time.perf_counter() - t0) * 1e3)  # ms
    times = np.array(times)
    return {
        "scores": out,
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
    }


def main():
    print("[micro] building synthetic rroq158 inputs (production shape)...")
    inputs = build_inputs()
    print(f"[micro]   B={SHAPE['n_docs']}, T={SHAPE['n_d_tok']}, S={SHAPE['n_q_tok']}, "
          f"dim={SHAPE['dim']}, K={SHAPE['K']}")
    print(f"[micro]   n_words={inputs['n_words']}, n_groups={inputs['n_groups']}, "
          f"query_bits={inputs['query_bits']}, group_words={inputs['n_words']//inputs['n_groups']}")
    print(f"[micro]   shape qualifies for V4 fast path: "
          f"{inputs['n_groups'] == 1 and inputs['n_words']//inputs['n_groups'] == 4 and inputs['query_bits'] == 4 and inputs['n_words'] == 4}")

    print("[micro] benching X86V3 (scalar popcntq + AVX2)...")
    r3 = bench(inputs, "x86v3", SHAPE["n_iters"], SHAPE["n_warmup"])
    print(f"[micro]   p50={r3['p50_ms']:.3f} ms, p95={r3['p95_ms']:.3f} ms")

    print("[micro] benching X86V4 (AVX-512 VPOPCNTDQ)...")
    r4 = bench(inputs, "auto", SHAPE["n_iters"], SHAPE["n_warmup"])
    print(f"[micro]   p50={r4['p50_ms']:.3f} ms, p95={r4['p95_ms']:.3f} ms")

    # Parity
    abs_err = np.abs(r3["scores"] - r4["scores"])
    rel_err = abs_err / (np.abs(r3["scores"]) + 1e-6)
    print(f"[micro] parity: max abs err = {abs_err.max():.4e}, max rel err = {rel_err.max():.4e}")
    bit_exact = np.array_equal(r3["scores"], r4["scores"])
    print(f"[micro]   bit-exact (==): {bit_exact}")

    speedup_p50 = r3["p50_ms"] / r4["p50_ms"]
    speedup_p95 = r3["p95_ms"] / r4["p95_ms"]
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  shape: B={SHAPE['n_docs']}, T={SHAPE['n_d_tok']}, S={SHAPE['n_q_tok']}, "
          f"dim={SHAPE['dim']}, K={SHAPE['K']}")
    print(f"  X86V3 (scalar popcntq + AVX2):     p50 {r3['p50_ms']:7.3f} ms  p95 {r3['p95_ms']:7.3f} ms")
    print(f"  X86V4 (AVX-512 VPOPCNTDQ):         p50 {r4['p50_ms']:7.3f} ms  p95 {r4['p95_ms']:7.3f} ms")
    print(f"  speedup (p50):                     {speedup_p50:.2f}x")
    print(f"  speedup (p95):                     {speedup_p95:.2f}x")
    print(f"  parity bit-exact:                  {bit_exact}")
    print("=" * 70)


if __name__ == "__main__":
    main()
