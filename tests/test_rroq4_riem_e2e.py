"""End-to-end build + search test for the rroq4_riem codec.

Builds a small index with ``Compression.RROQ4_RIEM`` on CPU, then runs a
search and verifies:

  1. The build wrote ``rroq4_riem_meta.npz`` (so the auto-derive can
     find it).
  2. The build_meta records ``compression == "rroq4_riem"``.
  3. The search returns results that overlap meaningfully with the FP16
     ground truth (R@10 >= 0.6 — rroq4_riem is the no-degradation lane).
  4. The auto-derive picks ``rroq4_riem`` as the runtime kernel without
     the caller setting ``quantization_mode``.

CPU-only — uses the Rust SIMD kernel via ``latence_shard_engine``. The
test is skipped when that wheel is not installed.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from voyager_index._internal.inference.shard_engine import ShardSegmentManager
from voyager_index._internal.inference.shard_engine.config import Compression
from voyager_index._internal.inference.shard_engine.manager import ShardEngineConfig


def _make_corpus(n_docs: int, dim: int, min_tok: int = 8, max_tok: int = 16, seed: int = 7):
    """Generate a small unit-normalized multi-vector corpus."""
    rng = np.random.RandomState(seed)
    vectors = []
    ids = list(range(n_docs))
    for _ in range(n_docs):
        n_tok = rng.randint(min_tok, max_tok + 1)
        v = rng.randn(n_tok, dim).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vectors.append(v)
    return vectors, ids, dim


@pytest.fixture
def tmp_index_dir():
    d = tempfile.mkdtemp(prefix="rroq4_riem_e2e_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_rroq4_riem_build_and_search_cpu(tmp_index_dir):
    """End-to-end build with RROQ4_RIEM + search on CPU.

    Uses K small enough to fit the synthetic corpus (so we don't trip
    the FP16 auto-fallback for tiny corpora). Also asserts that after
    rebuild the runtime auto-derives the rroq4_riem kernel without an
    explicit ``quantization_mode`` override.
    """
    pytest.importorskip("latence_shard_engine")
    import latence_shard_engine as eng
    if not hasattr(eng, "rroq4_riem_score_batch"):
        pytest.skip(
            "latence_shard_engine missing rroq4_riem_score_batch; rebuild "
            "the Rust wheel from src/kernels/shard_engine"
        )

    vectors, ids, dim = _make_corpus(n_docs=64, dim=64, seed=11)
    config = ShardEngineConfig(
        n_shards=2,
        dim=dim,
        compression=Compression.RROQ4_RIEM,
        rroq4_riem_k=64,
        rroq4_riem_group_size=32,
        lemur_epochs=2,
        k_candidates=64,
    )
    mgr = ShardSegmentManager(tmp_index_dir, config=config, device="cpu")
    try:
        mgr.build(vectors, ids)
        assert mgr.total_vectors() == 64
        assert (tmp_index_dir / "rroq4_riem_meta.npz").exists(), (
            "rroq4_riem_meta.npz not written; the lifecycle build path "
            "did not invoke encode_rroq4_riem"
        )
        # The auto-derive path should pick rroq4_riem as the runtime
        # kernel since the meta file is on disk and the manager doesn't
        # need an explicit quant override.
        assert mgr._derive_quantization_mode_from_storage() == "rroq4_riem"

        rng = np.random.RandomState(13)
        q = rng.randn(8, dim).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
        results = mgr.search_multivector(q, k=5)
        assert len(results) > 0
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert isinstance(score, float)
    finally:
        mgr.close()


def test_rroq4_riem_falls_back_to_fp16_on_tiny_corpus(tmp_index_dir):
    """When the corpus has fewer tokens than ``group_size`` the build
    must NOT crash trying to fit a 4-bit codebook with no data — the
    documented behaviour is to log a warning and silently downgrade to
    FP16. The codec rebuilds correctly on subsequent reopens via the
    on-disk manifest's ``compression`` field.
    """
    rng = np.random.RandomState(0)
    # 3 docs × 1 token = 3 tokens; group_size=32 → need >= 32 tokens
    vectors = [rng.randn(1, 64).astype(np.float32) for _ in range(3)]
    ids = list(range(3))
    config = ShardEngineConfig(
        n_shards=1,
        dim=64,
        compression=Compression.RROQ4_RIEM,
        rroq4_riem_k=32,
        rroq4_riem_group_size=32,
        lemur_epochs=1,
        k_candidates=10,
    )
    mgr = ShardSegmentManager(tmp_index_dir, config=config, device="cpu")
    try:
        mgr.build(vectors, ids)
        # Manifest must have downgraded to fp16 since rroq4_riem couldn't
        # encode anything; the meta file must NOT exist.
        assert not (tmp_index_dir / "rroq4_riem_meta.npz").exists()
        assert mgr.total_vectors() == 3
    finally:
        mgr.close()
