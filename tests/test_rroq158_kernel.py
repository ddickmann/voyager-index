"""
Parity + microbench for the rroq158 fused Triton kernel.

Three layers:

  1. ``test_rroq158_python_reference_matches_brute``  — verifies the python
     reference implementation in
     ``voyager_index/_internal/kernels/triton_roq_rroq158.py`` agrees with
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

from voyager_index._internal.inference.quantization.rroq158 import (
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)
from voyager_index._internal.kernels.triton_roq_rroq158 import (
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
    c_per_tok = centroids[enc.centroid_id]
    # Reconstruct r_amb from the stored ternary codes (but in rotated frame:
    # we stored r_rot; need r_amb = FWHT^-1(r_rot)). For parity we just use
    # the stored r_rot frame and rotate the query into it.
    sign_unp = np.unpackbits(enc.sign_plane, axis=1, bitorder="little")[:, : enc.dim]
    nz_unp = np.unpackbits(enc.nonzero_plane, axis=1, bitorder="little")[:, : enc.dim]
    signed = (2.0 * sign_unp.astype(np.float32) - 1.0) * nz_unp.astype(np.float32)
    scales = np.repeat(enc.scales.astype(np.float32), enc.group_size, axis=1)
    r_rot = (signed * scales).astype(np.float32)

    from voyager_index._internal.inference.quantization.rotational import FastWalshHadamard
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
    n_total_d = enc.centroid_id.shape[0]
    cos_norm = enc.cos_norm.astype(np.float32)                       # (n_total_d,)
    sin_norm = enc.sin_norm.astype(np.float32)                       # (n_total_d,)
    qc_per_tok = qc_full[:, enc.centroid_id]                         # (S, n_total_d)

    sim_per_pair = cos_norm[None, :] * qc_per_tok + sin_norm[None, :] * qr_full
    # MaxSim per (q, doc) by reshaping doc tokens into (n_docs, n_d_tok)
    sim_per_pair = sim_per_pair.reshape(queries.shape[0], n_docs, n_d_tok)
    out = sim_per_pair.max(axis=2).sum(axis=0)                       # (n_docs,)
    return out


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
    from voyager_index._internal.kernels.triton_roq_rroq158 import roq_maxsim_rroq158

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
    from voyager_index._internal.kernels.triton_roq_rroq158 import roq_maxsim_rroq158

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

    out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
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
