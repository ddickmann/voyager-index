"""
Parity + microbench for the rroq158 fused Triton kernel.

Three layers:

  1. ``test_rroq158_python_reference_matches_brute``  — verifies the python
     reference implementation in
     ``colsearch/_internal/kernels/triton_roq_rroq158.py`` agrees with
     a fully-decoded brute-force MaxSim on synthetic data. CPU-only, no
     Triton needed. This is the math-correctness test.

  2. ``test_rroq158_triton_matches_python_reference`` — verifies the Triton
     kernel matches the python reference on the same fixture, on a real
     GPU. ``skipif`` when CUDA isn't available.

  3. ``test_rroq158_microbench``  — drops a microbench JSON into
     ``reports/kernel_rroq158.json`` so we can check the
     p50 ≤ 0.40 ms / QPS ≥ 40k gates from the plan. Skipped on CPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from colsearch._internal.inference.quantization.rroq158 import (
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)
from colsearch._internal.kernels.triton_roq_rroq158 import (
    reference_score_rroq158,
)


def _make_fixture(n_docs=4, n_q_tok=3, n_d_tok=12, dim=64, K=32, seed=0):
    """Build a tiny synthetic corpus through the production encoder."""
    rng = np.random.default_rng(seed)
    n_total_d = n_docs * n_d_tok
    tokens = rng.standard_normal((n_total_d, dim)).astype(np.float32)
    queries = rng.standard_normal((n_q_tok, dim)).astype(np.float32)

    cfg = Rroq158Config(K=K, group_size=32, fit_sample_cap=n_total_d, encode_chunk=n_total_d, seed=seed)
    enc = encode_rroq158(tokens, cfg)

    q_inputs = encode_query_for_rroq158(
        queries, enc.centroids, fwht_seed=enc.fwht_seed, query_bits=4,
    )
    return enc, q_inputs, queries, tokens, n_docs, n_d_tok


def _reshape_doc_arrays(enc, n_docs, n_d_tok):
    sign_words = pack_doc_codes_to_int32_words(enc.sign_plane)
    nz_words = pack_doc_codes_to_int32_words(enc.nonzero_plane)
    n_words = sign_words.shape[1]
    n_groups = enc.scales.shape[1]
    sign_doc = sign_words.reshape(n_docs, n_d_tok, n_words)
    nz_doc = nz_words.reshape(n_docs, n_d_tok, n_words)
    scales_doc = enc.scales.astype(np.float32).reshape(n_docs, n_d_tok, n_groups)
    cid_doc = enc.centroid_id.astype(np.int32).reshape(n_docs, n_d_tok)
    cos_doc = enc.cos_norm.astype(np.float32).reshape(n_docs, n_d_tok)
    sin_doc = enc.sin_norm.astype(np.float32).reshape(n_docs, n_d_tok)
    return sign_doc, nz_doc, scales_doc, cid_doc, cos_doc, sin_doc


def _brute_force_maxsim(queries, enc, n_docs, n_d_tok):
    """fp32 reconstruction → MaxSim, the ground truth for parity."""
    centroids = enc.centroids
    # Reconstruct r_amb from the stored ternary codes (but in rotated frame:
    # we stored r_rot; need r_amb = FWHT^-1(r_rot)). For parity we just use
    # the stored r_rot frame and rotate the query into it.
    sign_unp = np.unpackbits(enc.sign_plane, axis=1, bitorder="little")[:, : enc.dim]
    nz_unp = np.unpackbits(enc.nonzero_plane, axis=1, bitorder="little")[:, : enc.dim]
    signed = (2.0 * sign_unp.astype(np.float32) - 1.0) * nz_unp.astype(np.float32)
    scales = np.repeat(enc.scales.astype(np.float32), enc.group_size, axis=1)
    r_rot = (signed * scales).astype(np.float32)

    from colsearch._internal.inference.quantization.rotational import FastWalshHadamard
    block_size = 1
    while block_size < enc.dim:
        block_size *= 2
    rotator = FastWalshHadamard(dim=enc.dim, num_rounds=3, block_size=block_size,
                                seed=enc.fwht_seed)
    q_rot = rotator.forward(torch.from_numpy(queries.astype(np.float32))).cpu().numpy()
    if q_rot.shape[1] != enc.dim:
        q_rot = q_rot[:, : enc.dim]
    # <q, c> via the original ambient query
    qc_full = queries @ centroids.T                                  # (S, K)
    qr_full = q_rot @ r_rot.T                                        # (S, n_total_d)
    cos_norm = enc.cos_norm.astype(np.float32)                       # (n_total_d,)
    sin_norm = enc.sin_norm.astype(np.float32)                       # (n_total_d,)
    qc_per_tok = qc_full[:, enc.centroid_id]                         # (S, n_total_d)

    sim_per_pair = cos_norm[None, :] * qc_per_tok + sin_norm[None, :] * qr_full
    # MaxSim per (q, doc) by reshaping doc tokens into (n_docs, n_d_tok)
    sim_per_pair = sim_per_pair.reshape(queries.shape[0], n_docs, n_d_tok)
    out = sim_per_pair.max(axis=2).sum(axis=0)                       # (n_docs,)
    return out


def test_rroq158_rust_simd_matches_python_reference():
    """The Rust AVX2/NEON CPU kernel must produce per-(query, doc) scores
    that are bitwise-identical (up to f32 rounding) to the python reference.
    This is the critical parity test for the production CPU lane.
    """
    pytest.importorskip("latence_shard_engine")
    import latence_shard_engine as eng

    enc, q_in, _queries, _, n_docs, n_d_tok = _make_fixture(
        n_docs=8, n_q_tok=4, n_d_tok=8, dim=128, K=64, seed=2,
    )
    sign_doc, nz_doc, scales_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    qp = q_in["q_planes"][None, :, :, :]
    qm = q_in["q_meta"][None, :, :]
    qc = q_in["qc_table"][None, :, :]

    ref = reference_score_rroq158(
        qp, qm, qc, cid_doc, cos_doc, sin_doc, sign_doc, nz_doc, scales_doc,
    )

    A, S, query_bits, n_words = qp.shape
    B, T = sign_doc.shape[:2]
    n_groups = scales_doc.shape[-1]
    K = qc.shape[-1]

    flat = eng.rroq158_score_batch(
        qp.astype(np.int32, copy=False).ravel(),
        qm.astype(np.float32, copy=False).ravel(),
        qc.astype(np.float32, copy=False).ravel(),
        sign_doc.astype(np.int32, copy=False).ravel(),
        nz_doc.astype(np.int32, copy=False).ravel(),
        scales_doc.astype(np.float32, copy=False).ravel(),
        cid_doc.astype(np.int32, copy=False).ravel(),
        cos_doc.astype(np.float32, copy=False).ravel(),
        sin_doc.astype(np.float32, copy=False).ravel(),
        A, B, S, T, n_words, n_groups, query_bits, K,
    )
    out = flat.reshape(A, B)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_rroq158_rust_simd_microbench():
    """Microbench drop for the Rust CPU kernel — records p50/p95/QPS into
    ``reports/kernel_rroq158_rust.json``. CPU-only, no CUDA needed."""
    pytest.importorskip("latence_shard_engine")
    import latence_shard_engine as eng

    rng = np.random.default_rng(11)
    n_q_tok = 32
    n_d_tok = 32
    dim = 128
    K = 1024
    n_docs = 512

    n_total_d = n_docs * n_d_tok
    docs = rng.standard_normal((n_total_d, dim)).astype(np.float32)
    queries = rng.standard_normal((n_q_tok, dim)).astype(np.float32)

    cfg = Rroq158Config(K=K, group_size=32, fit_sample_cap=8_000,
                        encode_chunk=8_000, seed=0)
    enc = encode_rroq158(docs, cfg)
    q_in = encode_query_for_rroq158(queries, enc.centroids,
                                    fwht_seed=enc.fwht_seed, query_bits=4)
    sign_doc, nz_doc, scales_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    qp = q_in["q_planes"][None, :, :, :]
    qm = q_in["q_meta"][None, :, :]
    qc = q_in["qc_table"][None, :, :]
    A, S, query_bits, n_words = qp.shape
    B, T = sign_doc.shape[:2]
    n_groups = scales_doc.shape[-1]

    qp_f = qp.astype(np.int32, copy=False).ravel()
    qm_f = qm.astype(np.float32, copy=False).ravel()
    qc_f = qc.astype(np.float32, copy=False).ravel()
    sg_f = sign_doc.astype(np.int32, copy=False).ravel()
    nz_f = nz_doc.astype(np.int32, copy=False).ravel()
    sc_f = scales_doc.astype(np.float32, copy=False).ravel()
    ci_f = cid_doc.astype(np.int32, copy=False).ravel()
    co_f = cos_doc.astype(np.float32, copy=False).ravel()
    si_f = sin_doc.astype(np.float32, copy=False).ravel()

    # warm up the rayon pool
    for _ in range(5):
        eng.rroq158_score_batch(
            qp_f, qm_f, qc_f, sg_f, nz_f, sc_f, ci_f, co_f, si_f,
            A, B, S, T, n_words, n_groups, query_bits, K,
        )

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        eng.rroq158_score_batch(
            qp_f, qm_f, qc_f, sg_f, nz_f, sc_f, ci_f, co_f, si_f,
            A, B, S, T, n_words, n_groups, query_bits, K,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    qps = (n_docs * 1000.0) / p50

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("kernel_rroq158_rust.json").write_text(json.dumps({
        "p50_ms": p50,
        "p95_ms": p95,
        "docs_per_query": n_docs,
        "qps": qps,
        "n_q_tok": n_q_tok,
        "n_d_tok": n_d_tok,
        "dim": dim,
        "K": K,
    }, indent=2))
    print(f"\n[rroq158-rust-cpu] p50={p50:.2f}ms  p95={p95:.2f}ms  QPS={qps:.0f} docs/s")


@pytest.mark.parametrize("dim", [96, 128, 160])
def test_rroq158_dense_matrix_fwht_path(dim):
    """The FWHT dense-matrix cache slices `full[:dim, :dim]` from the padded
    `padded_dim x padded_dim` operator. This is only sound because the unused
    rows/cols multiply zero-padded inputs; verify that for non-power-of-2
    dims the cached matrix still matches a fresh per-call rotation.
    """
    from colsearch._internal.inference.quantization.rroq158 import (
        get_cached_fwht_rotator,
        clear_fwht_rotator_cache,
    )
    clear_fwht_rotator_cache()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, dim)).astype(np.float32)

    rotator = get_cached_fwht_rotator(dim=dim, seed=7)
    cached_mat = rotator._dense_matrix_np
    assert cached_mat.shape == (dim, dim)

    direct = rotator.forward(torch.from_numpy(x)).cpu().numpy()
    if direct.shape[1] != dim:
        direct = direct[:, :dim]
    # Production usage in encode_query_for_rroq158 (line ~401) does
    # `q_rot = queries_f32 @ dense` (no transpose). The cached matrix is
    # constructed so that `x @ cached_mat == forward(x)[:, :dim]` for any
    # input shape (N, dim), including non-power-of-2 dim where padded_dim
    # > dim and the rotation spreads info into the padding cols.
    via_matmul = (x @ cached_mat).astype(np.float32)
    np.testing.assert_allclose(via_matmul, direct, rtol=1e-4, atol=1e-4)


def test_rroq158_K8192_parity_when_corpus_fits():
    """K=8192 is the production default. Build a small corpus where K=8192
    barely fits (n_tokens >= K), encode, and verify python-reference vs
    Rust-SIMD parity at that K.
    """
    pytest.importorskip("latence_shard_engine")
    import latence_shard_engine as eng

    rng = np.random.default_rng(42)
    K = 8192
    n_docs = 32
    n_d_tok = 256  # 32 * 256 = 8192 tokens => K-many seeds
    n_q_tok = 4
    dim = 128

    n_total = n_docs * n_d_tok
    docs = rng.standard_normal((n_total, dim)).astype(np.float32)
    queries = rng.standard_normal((n_q_tok, dim)).astype(np.float32)

    cfg = Rroq158Config(
        K=K, group_size=32, fit_sample_cap=n_total, encode_chunk=n_total, seed=0,
    )
    enc = encode_rroq158(docs, cfg)
    q_in = encode_query_for_rroq158(queries, enc.centroids,
                                    fwht_seed=enc.fwht_seed, query_bits=4)
    sign_doc, nz_doc, scales_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    qp = q_in["q_planes"][None, :, :, :]
    qm = q_in["q_meta"][None, :, :]
    qc = q_in["qc_table"][None, :, :]
    A, S, query_bits, n_words = qp.shape
    B, T = sign_doc.shape[:2]
    n_groups = scales_doc.shape[-1]

    ref = reference_score_rroq158(
        qp, qm, qc, cid_doc, cos_doc, sin_doc, sign_doc, nz_doc, scales_doc,
    )
    flat = eng.rroq158_score_batch(
        qp.astype(np.int32, copy=False).ravel(),
        qm.astype(np.float32, copy=False).ravel(),
        qc.astype(np.float32, copy=False).ravel(),
        sign_doc.astype(np.int32, copy=False).ravel(),
        nz_doc.astype(np.int32, copy=False).ravel(),
        scales_doc.astype(np.float32, copy=False).ravel(),
        cid_doc.astype(np.int32, copy=False).ravel(),
        cos_doc.astype(np.float32, copy=False).ravel(),
        sin_doc.astype(np.float32, copy=False).ravel(),
        A, B, S, T, n_words, n_groups, query_bits, K,
    ).reshape(A, B)
    np.testing.assert_allclose(flat, ref, rtol=1e-4, atol=1e-4)


def test_rroq158_skip_qc_table_parity():
    """When `skip_qc_table=True`, encode_query_for_rroq158 must return the
    same q_planes / q_meta as the full path. The qc_table is computed
    elsewhere (GPU), so the planes must be identical regardless.
    """
    rng = np.random.default_rng(3)
    n_q_tok = 4
    dim = 128
    K = 256
    n_d_tok = 32
    n_docs = 16  # 16*32 = 512 tokens >= K=256
    docs = rng.standard_normal((n_docs * n_d_tok, dim)).astype(np.float32)
    queries = rng.standard_normal((n_q_tok, dim)).astype(np.float32)

    cfg = Rroq158Config(K=K, group_size=32, fit_sample_cap=n_docs * n_d_tok,
                        encode_chunk=n_docs * n_d_tok, seed=0)
    enc = encode_rroq158(docs, cfg)

    full = encode_query_for_rroq158(
        queries, enc.centroids, fwht_seed=enc.fwht_seed, query_bits=4,
    )
    skipped = encode_query_for_rroq158(
        queries, None, fwht_seed=enc.fwht_seed, query_bits=4, skip_qc_table=True,
    )
    np.testing.assert_array_equal(full["q_planes"], skipped["q_planes"])
    np.testing.assert_allclose(full["q_meta"], skipped["q_meta"], rtol=1e-6)
    assert "qc_table" in full
    assert "qc_table" not in skipped or skipped["qc_table"] is None


def test_rroq158_config_validation():
    """Rroq158Config rejects bad K, group_size, fit_sample_cap."""
    with pytest.raises(ValueError, match="power of two"):
        Rroq158Config(K=1000, group_size=32, fit_sample_cap=2000)
    with pytest.raises(ValueError, match=r"K \(16\)"):
        Rroq158Config(K=16, group_size=32, fit_sample_cap=2000)
    with pytest.raises(ValueError, match="multiple of 32"):
        Rroq158Config(K=1024, group_size=24, fit_sample_cap=2000)
    with pytest.raises(ValueError, match="fit_sample_cap"):
        Rroq158Config(K=4096, group_size=32, fit_sample_cap=1024)
    Rroq158Config(K=8192, group_size=32, fit_sample_cap=20000)


def test_rroq158_python_reference_matches_brute():
    enc, q_in, queries, _, n_docs, n_d_tok = _make_fixture(
        n_docs=4, n_q_tok=3, n_d_tok=8, dim=64, K=32, seed=0,
    )
    sign_doc, nz_doc, scales_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    # python reference: shape (1, n_docs)
    qp = q_in["q_planes"][None, :, :, :]                  # (A=1, S, query_bits, n_words)
    qm = q_in["q_meta"][None, :, :]                       # (A=1, S, 2)
    qc = q_in["qc_table"][None, :, :]                     # (A=1, S, K)

    out_ref = reference_score_rroq158(
        qp, qm, qc, cid_doc, cos_doc, sin_doc, sign_doc, nz_doc, scales_doc,
    )                                                     # (1, n_docs)

    out_brute = _brute_force_maxsim(queries, enc, n_docs, n_d_tok)
    # Asymmetric ternary + 4-bit query has bounded distortion; a few-percent
    # gap on a tiny synthetic fixture is expected. The tighter parity test
    # (against the kernel-internal numpy reference) lives in test_rroq158_triton.
    np.testing.assert_allclose(out_ref[0], out_brute, rtol=0.20, atol=0.10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rroq158_triton_matches_python_reference():
    pytest.importorskip("triton")
    from colsearch._internal.kernels.triton_roq_rroq158 import roq_maxsim_rroq158

    enc, q_in, _queries, _, n_docs, n_d_tok = _make_fixture(
        n_docs=8, n_q_tok=4, n_d_tok=8, dim=128, K=64, seed=1,
    )
    sign_doc, nz_doc, scales_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    qp = q_in["q_planes"][None, :, :, :]
    qm = q_in["q_meta"][None, :, :]
    qc = q_in["qc_table"][None, :, :]

    ref = reference_score_rroq158(
        qp, qm, qc, cid_doc, cos_doc, sin_doc, sign_doc, nz_doc, scales_doc,
    )

    dev = "cuda"
    out = roq_maxsim_rroq158(
        queries_planes=torch.from_numpy(qp).to(dev),
        queries_meta=torch.from_numpy(qm).to(dev),
        qc_table=torch.from_numpy(qc).to(dev),
        docs_centroid_id=torch.from_numpy(cid_doc).to(dev),
        docs_cos_norm=torch.from_numpy(cos_doc).to(dev),
        docs_sin_norm=torch.from_numpy(sin_doc).to(dev),
        docs_sign=torch.from_numpy(sign_doc).to(dev),
        docs_nz=torch.from_numpy(nz_doc).to(dev),
        docs_scales=torch.from_numpy(scales_doc).to(dev),
    ).cpu().numpy()

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rroq158_microbench():
    """Microbench drop. Records p50/p95/QPS into reports/kernel_rroq158.json
    so the merge-decision memo can check the plan's gates
    (p50 ≤ 0.40 ms, QPS ≥ 40k)."""
    pytest.importorskip("triton")
    from colsearch._internal.kernels.triton_roq_rroq158 import roq_maxsim_rroq158

    rng = np.random.default_rng(7)
    n_q_tok = 32
    n_d_tok = 32
    dim = 128
    K = 1024
    n_docs = 512

    n_total_d = n_docs * n_d_tok
    docs = rng.standard_normal((n_total_d, dim)).astype(np.float32)
    queries = rng.standard_normal((n_q_tok, dim)).astype(np.float32)

    cfg = Rroq158Config(K=K, group_size=32, fit_sample_cap=8_000,
                        encode_chunk=8_000, seed=0)
    enc = encode_rroq158(docs, cfg)
    q_in = encode_query_for_rroq158(queries, enc.centroids,
                                    fwht_seed=enc.fwht_seed, query_bits=4)
    sign_doc, nz_doc, scales_doc, cid_doc, cos_doc, sin_doc = _reshape_doc_arrays(
        enc, n_docs, n_d_tok,
    )

    dev = "cuda"
    qp = torch.from_numpy(q_in["q_planes"][None, :, :, :]).to(dev)
    qm = torch.from_numpy(q_in["q_meta"][None, :, :]).to(dev)
    qc = torch.from_numpy(q_in["qc_table"][None, :, :]).to(dev)
    cid_t = torch.from_numpy(cid_doc).to(dev)
    cos_t = torch.from_numpy(cos_doc).to(dev)
    sin_t = torch.from_numpy(sin_doc).to(dev)
    sign_t = torch.from_numpy(sign_doc).to(dev)
    nz_t = torch.from_numpy(nz_doc).to(dev)
    scales_t = torch.from_numpy(scales_doc).to(dev)

    # Warmup + autotune
    for _ in range(5):
        roq_maxsim_rroq158(qp, qm, qc, cid_t, cos_t, sin_t, sign_t, nz_t, scales_t)
    torch.cuda.synchronize()

    n_iters = 50
    times_ms = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        roq_maxsim_rroq158(qp, qm, qc, cid_t, cos_t, sin_t, sign_t, nz_t, scales_t)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms = np.array(times_ms)
    p50 = float(np.median(times_ms))
    p95 = float(np.percentile(times_ms, 95))
    qps = 1000.0 * n_docs / p50  # docs/sec from one batch

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "kernel_rroq158.json").write_text(json.dumps({
        "n_q_tokens": n_q_tok,
        "n_d_tokens_per_doc": n_d_tok,
        "n_docs_per_batch": n_docs,
        "K": K,
        "dim": dim,
        "p50_ms": p50,
        "p95_ms": p95,
        "qps_docs": qps,
        "iters": n_iters,
    }, indent=2))
    print(f"rroq158 kernel  p50={p50:.3f}ms  p95={p95:.3f}ms  qps={qps:.0f}")
    # Soft gate — record but don't fail tests on it; the merge memo decides.
