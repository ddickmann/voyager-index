"""Parity + microbench tests for the rroq4_riem (Riemannian 4-bit
asymmetric) kernels.

Five layers, mirroring tests/test_rroq158_kernel.py:

  1. ``test_rroq4_riem_python_reference_matches_brute`` — verifies the
     pure-numpy reference scorer in ``triton_roq_rroq4_riem.py`` matches
     a direct fp32 reconstruction MaxSim on synthetic data. CPU-only,
     no Triton or Rust required.

  2. ``test_rroq4_riem_rust_simd_matches_python_reference`` — verifies
     the Rust SIMD CPU kernel from ``latence_shard_engine.rroq4_riem_score_batch``
     matches the python reference. Skipped if the Rust wheel isn't
     installed.

  3. ``test_rroq4_riem_triton_matches_python_reference`` — verifies the
     Triton GPU kernel matches the python reference. Skipped if CUDA
     isn't available.

  4. ``test_rroq4_riem_microbench`` — drops a microbench JSON into
     ``reports/kernel_rroq4_riem.json`` (kernel-only Triton p50/p95 at
     the same shape as the rroq158 microbench, so the Phase-7
     wrapper-included p95 can be split into kernel vs wrapper).

  5. ``test_rroq4_riem_rust_simd_microbench`` — same idea for the CPU
     Rust kernel. Drops ``reports/kernel_rroq4_riem_rust.json``.

The first three are the "no-degradation safe fallback" parity
contracts — both kernels must be bit-equivalent (to fp32 rounding) to
the brute-force MaxSim. The last two exist so the README's
"rroq4_riem is still slower than fp16" line can cite measured
kernel-only ms instead of only wrapper-included BEIR p95 — they also
gate regressions on the Phase-7-followup loop-reorder in
``src/kernels/shard_engine/src/fused_rroq4_riem.rs``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from colsearch._internal.inference.quantization.rroq4_riem import (
    Rroq4RiemConfig,
    encode_query_for_rroq4_riem,
    encode_rroq4_riem,
    unpack_4bit,
)
from colsearch._internal.kernels.triton_roq_rroq4_riem import (
    reference_score_rroq4_riem,
)


def _make_fixture(n_docs=4, n_q_tok=3, n_d_tok=12, dim=64, K=32, group_size=32, seed=0):
    """Build a tiny synthetic corpus through the production encoder."""
    rng = np.random.default_rng(seed)
    n_total_d = n_docs * n_d_tok
    tokens = rng.standard_normal((n_total_d, dim)).astype(np.float32)
    queries = rng.standard_normal((n_q_tok, dim)).astype(np.float32)

    cfg = Rroq4RiemConfig(
        K=K,
        group_size=group_size,
        fit_sample_cap=n_total_d,
        encode_chunk=n_total_d,
        seed=seed,
    )
    enc = encode_rroq4_riem(tokens, cfg)

    q_inputs = encode_query_for_rroq4_riem(
        queries,
        enc.centroids,
        fwht_seed=enc.fwht_seed,
        group_size=enc.group_size,
    )
    return enc, q_inputs, queries, tokens, n_docs, n_d_tok


def _reshape_doc_arrays(enc, n_docs, n_d_tok):
    n_groups = enc.mins.shape[1]
    dim = enc.dim
    n_bytes = dim // 2
    codes_doc = enc.codes_packed.reshape(n_docs, n_d_tok, n_bytes)
    mins_doc = enc.mins.astype(np.float32).reshape(n_docs, n_d_tok, n_groups)
    dlts_doc = enc.deltas.astype(np.float32).reshape(n_docs, n_d_tok, n_groups)
    cid_doc = enc.centroid_id.astype(np.int32).reshape(n_docs, n_d_tok)
    cos_doc = enc.cos_norm.astype(np.float32).reshape(n_docs, n_d_tok)
    sin_doc = enc.sin_norm.astype(np.float32).reshape(n_docs, n_d_tok)
    return codes_doc, mins_doc, dlts_doc, cid_doc, cos_doc, sin_doc


def _brute_force_maxsim(queries, enc, n_docs, n_d_tok):
    """fp32 reconstruction → MaxSim, the ground truth for parity.

    Reconstructs ``r_rot = mins + delta * codes`` per group, then scores
    in the rotated frame against ``q_rot = FWHT(query)``, which is what
    the kernel does internally.
    """
    centroids = enc.centroids
    dim = enc.dim
    group_size = enc.group_size
    n_groups = dim // group_size

    codes_full = unpack_4bit(enc.codes_packed, dim).astype(np.float32)
    mins_per_dim = np.repeat(enc.mins.astype(np.float32), group_size, axis=1)
    dlts_per_dim = np.repeat(enc.deltas.astype(np.float32), group_size, axis=1)
    r_rot = mins_per_dim + dlts_per_dim * codes_full  # (n_total_d, dim)

    from colsearch._internal.inference.quantization.rroq158 import (
        get_cached_fwht_rotator,
    )
    rotator = get_cached_fwht_rotator(dim=dim, seed=enc.fwht_seed)
    dense = getattr(rotator, "_dense_matrix_np", None)
    if dense is not None and dense.shape == (dim, dim):
        q_rot = (queries.astype(np.float32) @ dense).astype(np.float32)
    else:
        q_rot = rotator.forward(torch.from_numpy(queries.astype(np.float32))).cpu().numpy()
        if q_rot.shape[1] != dim:
            q_rot = q_rot[:, :dim]

    qc_full = queries @ centroids.T  # (S, K)
    qr_full = q_rot @ r_rot.T        # (S, n_total_d)
    cos_norm = enc.cos_norm.astype(np.float32)
    sin_norm = enc.sin_norm.astype(np.float32)
    qc_per_tok = qc_full[:, enc.centroid_id]  # (S, n_total_d)

    sim_per_pair = cos_norm[None, :] * qc_per_tok + sin_norm[None, :] * qr_full
    sim_per_pair = sim_per_pair.reshape(queries.shape[0], n_docs, n_d_tok)
    out = sim_per_pair.max(axis=2).sum(axis=0)
    return out, q_rot


def test_rroq4_riem_python_reference_matches_brute():
    """The python reference scorer must agree with the fp32 brute-force
    reconstruction MaxSim. This proves the kernel formula
    ``cos*qc + sin*(Σ_g delta_g * <q_rot[g], code[g]> + min_g * Σ q_rot[g])``
    is mathematically equivalent to MaxSim over the dequantized vectors.
    """
    enc, q_in, queries, _, n_docs, n_d_tok = _make_fixture(
        n_docs=4, n_q_tok=3, n_d_tok=8, dim=64, K=32, group_size=32, seed=0,
    )
    codes_doc, mins_doc, dlts_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    qr = q_in["q_rot"][None, :, :]
    qgs = q_in["q_group_sums"][None, :, :]
    qc = q_in["qc_table"][None, :, :]

    ref = reference_score_rroq4_riem(
        qr, qgs, qc, cid_doc, cos_doc, sin_doc, codes_doc, mins_doc, dlts_doc,
        group_size=enc.group_size,
    )[0]

    brute, _ = _brute_force_maxsim(queries, enc, n_docs, n_d_tok)
    np.testing.assert_allclose(ref, brute, rtol=1e-4, atol=1e-3)


def test_rroq4_riem_rust_simd_matches_python_reference():
    """The Rust AVX2/FMA CPU kernel must match the python reference."""
    pytest.importorskip("latence_shard_engine")
    import latence_shard_engine as eng

    rroq4_fn = getattr(eng, "rroq4_riem_score_batch", None)
    if rroq4_fn is None:
        pytest.skip(
            "latence_shard_engine missing rroq4_riem_score_batch; rebuild "
            "the Rust wheel from src/kernels/shard_engine"
        )

    enc, q_in, _, _, n_docs, n_d_tok = _make_fixture(
        n_docs=8, n_q_tok=4, n_d_tok=8, dim=128, K=64, group_size=32, seed=2,
    )
    codes_doc, mins_doc, dlts_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    qr = q_in["q_rot"][None, :, :]
    qgs = q_in["q_group_sums"][None, :, :]
    qc = q_in["qc_table"][None, :, :]

    ref = reference_score_rroq4_riem(
        qr, qgs, qc, cid_doc, cos_doc, sin_doc, codes_doc, mins_doc, dlts_doc,
        group_size=enc.group_size,
    )

    A, S, dim = qr.shape
    B, T = codes_doc.shape[:2]
    n_groups = mins_doc.shape[-1]
    K = qc.shape[-1]

    flat = rroq4_fn(
        qr.astype(np.float32, copy=False).ravel(),
        qgs.astype(np.float32, copy=False).ravel(),
        qc.astype(np.float32, copy=False).ravel(),
        codes_doc.astype(np.uint8, copy=False).ravel(),
        mins_doc.astype(np.float32, copy=False).ravel(),
        dlts_doc.astype(np.float32, copy=False).ravel(),
        cid_doc.astype(np.int32, copy=False).ravel(),
        cos_doc.astype(np.float32, copy=False).ravel(),
        sin_doc.astype(np.float32, copy=False).ravel(),
        A, B, S, T, dim, n_groups, enc.group_size, K,
    )
    out = flat.reshape(A, B)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rroq4_riem_triton_matches_python_reference():
    """The Triton GPU kernel must match the python reference."""
    triton = pytest.importorskip("triton")  # noqa: F841
    from colsearch._internal.kernels.triton_roq_rroq4_riem import (
        roq_maxsim_rroq4_riem,
    )

    enc, q_in, _, _, n_docs, n_d_tok = _make_fixture(
        n_docs=8, n_q_tok=4, n_d_tok=8, dim=128, K=64, group_size=32, seed=3,
    )
    codes_doc, mins_doc, dlts_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    qr = q_in["q_rot"][None, :, :]
    qgs = q_in["q_group_sums"][None, :, :]
    qc = q_in["qc_table"][None, :, :]

    ref = reference_score_rroq4_riem(
        qr, qgs, qc, cid_doc, cos_doc, sin_doc, codes_doc, mins_doc, dlts_doc,
        group_size=enc.group_size,
    )

    dev = torch.device("cuda")
    out = roq_maxsim_rroq4_riem(
        queries_rot=torch.from_numpy(qr).to(dev),
        queries_group_sums=torch.from_numpy(qgs).to(dev),
        qc_table=torch.from_numpy(qc).to(dev),
        docs_centroid_id=torch.from_numpy(cid_doc).to(dev),
        docs_cos_norm=torch.from_numpy(cos_doc).to(dev),
        docs_sin_norm=torch.from_numpy(sin_doc).to(dev),
        docs_codes_packed=torch.from_numpy(codes_doc).to(dev),
        docs_mins=torch.from_numpy(mins_doc).to(dev),
        docs_deltas=torch.from_numpy(dlts_doc).to(dev),
        group_size=enc.group_size,
    ).cpu().numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rroq4_riem_microbench():
    """Microbench drop for the Triton GPU kernel — records p50/p95/QPS
    into ``reports/kernel_rroq4_riem.json``. Same shape as the rroq158
    microbench (32 q-tok x 32 d-tok x 512 docs, K=1024, dim=128) so the
    Phase-7 wrapper-included BEIR p95 can be split into kernel-only ms
    vs wrapper ms."""
    pytest.importorskip("triton")
    from colsearch._internal.kernels.triton_roq_rroq4_riem import (
        roq_maxsim_rroq4_riem,
    )

    rng = np.random.default_rng(7)
    n_q_tok = 32
    n_d_tok = 32
    dim = 128
    K = 1024
    group_size = 32
    n_docs = 512

    n_total_d = n_docs * n_d_tok
    docs = rng.standard_normal((n_total_d, dim)).astype(np.float32)
    queries = rng.standard_normal((n_q_tok, dim)).astype(np.float32)

    cfg = Rroq4RiemConfig(
        K=K,
        group_size=group_size,
        fit_sample_cap=8_000,
        encode_chunk=8_000,
        seed=0,
    )
    enc = encode_rroq4_riem(docs, cfg)
    q_in = encode_query_for_rroq4_riem(
        queries, enc.centroids, fwht_seed=enc.fwht_seed, group_size=enc.group_size,
    )
    codes_doc, mins_doc, dlts_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    dev = torch.device("cuda")
    qr = torch.from_numpy(q_in["q_rot"][None, :, :]).to(dev)
    qgs = torch.from_numpy(q_in["q_group_sums"][None, :, :]).to(dev)
    qc = torch.from_numpy(q_in["qc_table"][None, :, :]).to(dev)
    cid_t = torch.from_numpy(cid_doc).to(dev)
    cos_t = torch.from_numpy(cos_doc).to(dev)
    sin_t = torch.from_numpy(sin_doc).to(dev)
    codes_t = torch.from_numpy(codes_doc).to(dev)
    mins_t = torch.from_numpy(mins_doc).to(dev)
    dlts_t = torch.from_numpy(dlts_doc).to(dev)

    for _ in range(5):
        roq_maxsim_rroq4_riem(
            queries_rot=qr,
            queries_group_sums=qgs,
            qc_table=qc,
            docs_centroid_id=cid_t,
            docs_cos_norm=cos_t,
            docs_sin_norm=sin_t,
            docs_codes_packed=codes_t,
            docs_mins=mins_t,
            docs_deltas=dlts_t,
            group_size=enc.group_size,
        )
    torch.cuda.synchronize()

    n_iters = 50
    times_ms = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        roq_maxsim_rroq4_riem(
            queries_rot=qr,
            queries_group_sums=qgs,
            qc_table=qc,
            docs_centroid_id=cid_t,
            docs_cos_norm=cos_t,
            docs_sin_norm=sin_t,
            docs_codes_packed=codes_t,
            docs_mins=mins_t,
            docs_deltas=dlts_t,
            group_size=enc.group_size,
        )
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms = np.array(times_ms)
    p50 = float(np.median(times_ms))
    p95 = float(np.percentile(times_ms, 95))
    qps = 1000.0 * n_docs / p50

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "kernel_rroq4_riem.json").write_text(json.dumps({
        "n_q_tokens": n_q_tok,
        "n_d_tokens_per_doc": n_d_tok,
        "n_docs_per_batch": n_docs,
        "K": K,
        "group_size": group_size,
        "dim": dim,
        "p50_ms": p50,
        "p95_ms": p95,
        "qps_docs": qps,
        "iters": n_iters,
    }, indent=2))
    print(f"\n[rroq4_riem-triton] p50={p50:.3f}ms  p95={p95:.3f}ms  qps={qps:.0f} docs/s")


def test_rroq4_riem_rust_simd_microbench():
    """Microbench drop for the Rust CPU kernel — records p50/p95/QPS
    into ``reports/kernel_rroq4_riem_rust.json``. CPU-only, no CUDA
    needed. Same shape as the rroq158 Rust microbench so kernel-only
    ms can be compared apples-to-apples."""
    pytest.importorskip("latence_shard_engine")
    import latence_shard_engine as eng

    rroq4_fn = getattr(eng, "rroq4_riem_score_batch", None)
    if rroq4_fn is None:
        pytest.skip(
            "latence_shard_engine missing rroq4_riem_score_batch; rebuild "
            "the Rust wheel from src/kernels/shard_engine"
        )

    rng = np.random.default_rng(11)
    n_q_tok = 32
    n_d_tok = 32
    dim = 128
    K = 1024
    group_size = 32
    n_docs = 512

    n_total_d = n_docs * n_d_tok
    docs = rng.standard_normal((n_total_d, dim)).astype(np.float32)
    queries = rng.standard_normal((n_q_tok, dim)).astype(np.float32)

    cfg = Rroq4RiemConfig(
        K=K,
        group_size=group_size,
        fit_sample_cap=8_000,
        encode_chunk=8_000,
        seed=0,
    )
    enc = encode_rroq4_riem(docs, cfg)
    q_in = encode_query_for_rroq4_riem(
        queries, enc.centroids, fwht_seed=enc.fwht_seed, group_size=enc.group_size,
    )
    codes_doc, mins_doc, dlts_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    qr = q_in["q_rot"][None, :, :]
    qgs = q_in["q_group_sums"][None, :, :]
    qc = q_in["qc_table"][None, :, :]
    A, S, _ = qr.shape
    B, T = codes_doc.shape[:2]
    n_groups = mins_doc.shape[-1]

    qr_f = qr.astype(np.float32, copy=False).ravel()
    qgs_f = qgs.astype(np.float32, copy=False).ravel()
    qc_f = qc.astype(np.float32, copy=False).ravel()
    codes_f = codes_doc.astype(np.uint8, copy=False).ravel()
    mins_f = mins_doc.astype(np.float32, copy=False).ravel()
    dlts_f = dlts_doc.astype(np.float32, copy=False).ravel()
    cid_f = cid_doc.astype(np.int32, copy=False).ravel()
    cos_f = cos_doc.astype(np.float32, copy=False).ravel()
    sin_f = sin_doc.astype(np.float32, copy=False).ravel()

    for _ in range(5):
        rroq4_fn(
            qr_f, qgs_f, qc_f, codes_f, mins_f, dlts_f, cid_f, cos_f, sin_f,
            A, B, S, T, dim, n_groups, group_size, K,
        )

    n_iters = 50
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        rroq4_fn(
            qr_f, qgs_f, qc_f, codes_f, mins_f, dlts_f, cid_f, cos_f, sin_f,
            A, B, S, T, dim, n_groups, group_size, K,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    qps = (n_docs * 1000.0) / p50

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("kernel_rroq4_riem_rust.json").write_text(json.dumps({
        "p50_ms": p50,
        "p95_ms": p95,
        "docs_per_query": n_docs,
        "qps": qps,
        "n_q_tok": n_q_tok,
        "n_d_tok": n_d_tok,
        "dim": dim,
        "K": K,
        "group_size": group_size,
    }, indent=2))
    print(f"\n[rroq4_riem-rust-cpu] p50={p50:.2f}ms  p95={p95:.2f}ms  QPS={qps:.0f} docs/s")
