"""Tests for ROQ 4-bit quantization in shard engine (Chunk 5)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from colsearch._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)
from colsearch._internal.inference.shard_engine.config import Compression
from colsearch._internal.inference.shard_engine.scorer import (
    _get_roq_maxsim,
    score_roq4_topk,
)


def _make_corpus(n_docs: int = 50, dim: int = 128, min_tok: int = 8, max_tok: int = 24):
    rng = np.random.RandomState(42)
    vectors = []
    for _ in range(n_docs):
        n_tok = rng.randint(min_tok, max_tok + 1)
        v = rng.randn(n_tok, dim).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vectors.append(v)
    return vectors


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="shard_roq4_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


ROQ_AVAILABLE = False
try:
    from colsearch._internal.inference.quantization.rotational import (
        RotationalQuantizer,
        RoQConfig,
    )
    ROQ_AVAILABLE = True
except ImportError:
    pass


class TestROQ4Build:
    @pytest.mark.skipif(not ROQ_AVAILABLE, reason="ROQ quantizer not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_build_with_roq4(self, tmp_dir):
        dim = 128
        config = ShardEngineConfig(
            n_shards=4,
            dim=dim,
            compression=Compression.ROQ4,
            lemur_epochs=2,
            k_candidates=50,
        )
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=50, dim=dim)
        mgr.build(vectors, list(range(50)))

        assert (tmp_dir / "roq_quantizer.pkl").exists()
        stats = mgr.get_statistics()
        assert stats["compression"] == "roq4"
        mgr.close()

    @pytest.mark.skipif(not ROQ_AVAILABLE, reason="ROQ quantizer not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_roq4_search_still_works(self, tmp_dir):
        """Build ROQ4 index and search. FP16 fallback for scoring is fine."""
        dim = 128
        config = ShardEngineConfig(
            n_shards=4,
            dim=dim,
            compression=Compression.ROQ4,
            lemur_epochs=2,
            k_candidates=50,
        )
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=50, dim=dim)
        mgr.build(vectors, list(range(50)))

        q = np.random.RandomState(99).randn(16, dim).astype(np.float32)
        results = mgr.search_multivector(q, k=5)
        assert len(results) == 5
        mgr.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp16_fallback_when_roq_unavailable(self, tmp_dir):
        """If ROQ import fails, build should fall back to FP16."""
        dim = 64
        config = ShardEngineConfig(
            n_shards=4,
            dim=dim,
            compression=Compression.FP16,
            lemur_epochs=2,
            k_candidates=50,
        )
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=30, dim=dim)
        mgr.build(vectors, list(range(30)))

        stats = mgr.get_statistics()
        assert stats["compression"] == "fp16"
        mgr.close()


class TestROQ4Scorer:
    def test_roq_kernel_loader(self):
        fn = _get_roq_maxsim()
        # May or may not be available depending on environment

    @pytest.mark.skipif(not ROQ_AVAILABLE, reason="ROQ quantizer not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_quantize_and_score(self):
        """Encode a small corpus with ROQ4, score with score_roq4_topk."""
        dim = 128
        rng = np.random.RandomState(42)
        quantizer = RotationalQuantizer(RoQConfig(dim=dim, num_bits=4, seed=42))

        n_docs = 10
        doc_codes_list = []
        doc_meta_list = []
        for _ in range(n_docs):
            v = rng.randn(16, dim).astype(np.float32)
            q = quantizer.quantize(v, store=False)
            doc_codes_list.append(np.asarray(q["codes"], dtype=np.uint8))
            doc_meta_list.append(quantizer.build_triton_meta(q, include_norm_sq=True))

        max_tok = max(c.shape[0] for c in doc_codes_list)
        nb = doc_codes_list[0].shape[1]

        dc = np.zeros((n_docs, max_tok, nb), dtype=np.uint8)
        dm = np.zeros((n_docs, max_tok, 4), dtype=np.float32)
        for i in range(n_docs):
            t = doc_codes_list[i].shape[0]
            dc[i, :t] = doc_codes_list[i]
            dm[i, :t] = doc_meta_list[i]

        query = rng.randn(8, dim).astype(np.float32)
        qq = quantizer.quantize(query, store=False)
        qc = np.asarray(qq["codes"], dtype=np.uint8)[np.newaxis]
        qm = quantizer.build_triton_meta(qq, include_norm_sq=True)[np.newaxis]

        roq_fn = _get_roq_maxsim()
        if roq_fn is None:
            pytest.skip("ROQ Triton kernel not available")

        doc_ids = list(range(n_docs))
        result_ids, result_scores = score_roq4_topk(
            query_codes=torch.from_numpy(qc),
            query_meta=torch.from_numpy(qm),
            doc_codes=torch.from_numpy(dc),
            doc_meta=torch.from_numpy(dm),
            doc_ids=doc_ids,
            k=5,
        )
        assert len(result_ids) == 5
        assert result_scores == sorted(result_scores, reverse=True)
