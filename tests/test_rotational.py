"""
Tests for Rotational Quantization
"""

import pytest
import numpy as np
import torch
from voyager_index._internal.inference.quantization.rotational import RotationalQuantizer, RoQConfig, FastWalshHadamard

@pytest.fixture
def sample_embeddings():
    np.random.seed(42)
    # 1000 random vectors, dim 128
    return np.random.randn(1000, 128).astype(np.float32)

@pytest.fixture
def query_embedding():
    np.random.seed(123)
    return np.random.randn(1, 128).astype(np.float32)

class TestFWHT:
    def test_shapes(self):
        dim = 128
        fwht = FastWalshHadamard(dim, num_rounds=3, block_size=64)
        x = torch.randn(10, dim)
        out = fwht.forward(x)
        assert out.shape == (10, dim) # Since 128 is multiple of 64
        
    def test_padding(self):
        dim = 100
        fwht = FastWalshHadamard(dim, num_rounds=1, block_size=64)
        # Padding to 128
        assert fwht.padded_dim == 128
        x = torch.randn(5, dim)
        out = fwht.forward(x)
        assert out.shape == (5, 128)
        
    def test_orthogonality_preservation(self):
        # Rotation should preserve norms roughly (FWHT is orthogonal)
        dim = 128
        fwht = FastWalshHadamard(dim, num_rounds=1, block_size=128)
        x = torch.randn(100, dim)
        
        # Normalize x
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        # Rotate
        out = fwht.forward(x)
        
        # Norms of out should be 1.0 (approx)
        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
        # Distances should be preserved
        # d(x, y) = ||x - y||
        # d(Rx, Ry) = ||Rx - Ry|| for orthogonal R
        d_orig = torch.norm(x[0] - x[1])
        d_rot = torch.norm(out[0] - out[1])
        assert torch.allclose(d_orig, d_rot, atol=1e-5)

class TestRoQ:
    def test_quantization_flow(self, sample_embeddings, query_embedding):
        config = RoQConfig(dim=128, num_bits=8)
        roq = RotationalQuantizer(config)
        
        # Fit (noop) and Index
        roq.quantize(sample_embeddings, store=True)
        
        # Search
        idx, dists = roq.search(query_embedding, top_k=10)
        
        assert len(idx) == 10
        assert len(dists) == 10
        
    def test_recall_accuracy(self):
        """Verify recall is high (>0.99)"""
        N = 2000
        dim = 128
        top_k = 10
        
        # Generate random normalized data
        np.random.seed(42)
        data = np.random.randn(N, dim).astype(np.float32)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        
        queries = np.random.randn(50, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        # Ground truth
        # Cosine distance = 1 - dot
        # Since normalized, L2 order is same as cosine order (smallest L2 = largest dot)
        # ||x - y||^2 = 2 - 2<x,y>
        
        # Exact search
        exact_scores = queries @ data.T
        exact_indices = np.argsort(-exact_scores, axis=1)[:, :top_k]
        
        # RoQ Search
        config = RoQConfig(dim=dim, num_rounds=3)
        roq = RotationalQuantizer(config)
        roq.quantize(data, store=True)
        
        roq_idx, roq_dists = roq.search(queries, top_k=top_k)
        
        # Recall@10
        total_recall = 0
        for i in range(len(queries)):
            truth = set(exact_indices[i])
            found = set(roq_idx[i])
            recall = len(truth.intersection(found)) / top_k
            total_recall += recall
            
        avg_recall = total_recall / len(queries)
        print(f"Average Recall@{top_k}: {avg_recall:.4f}")
        
        # Weaviate claims >99%
        assert avg_recall > 0.95, f"Recall {avg_recall} too low"

    def test_build_triton_meta_for_scalar_4bit(self):
        embeddings = np.random.randn(6, 128).astype(np.float32)
        roq = RotationalQuantizer(RoQConfig(dim=128, num_bits=4))

        quantized = roq.quantize(embeddings, store=False)
        meta = roq.build_triton_meta(quantized)
        compact_meta = roq.build_triton_meta(quantized, include_norm_sq=False)
        codes, stacked_meta = roq.stack_triton_inputs(quantized, batch_size=2, item_count=3, device="cpu")
        _, stacked_compact_meta = roq.stack_triton_inputs(
            quantized,
            batch_size=2,
            item_count=3,
            device="cpu",
            include_norm_sq=False,
        )

        assert meta.shape == (6, 4)
        assert compact_meta.shape == (6, 3)
        assert codes.shape[0] == 2
        assert codes.shape[1] == 3
        assert stacked_meta.shape == (2, 3, 4)
        assert stacked_compact_meta.shape == (2, 3, 3)

    def test_build_triton_meta_rejects_grouped_4bit(self):
        embeddings = np.random.randn(4, 128).astype(np.float32)
        roq = RotationalQuantizer(RoQConfig(dim=128, num_bits=4, group_size=16))

        quantized = roq.quantize(embeddings, store=False)

        with pytest.raises(ValueError, match="scalar per-vector"):
            roq.build_triton_meta(quantized)

    def test_1bit_screening_defaults_and_padding(self):
        embeddings = np.random.randn(4, 128).astype(np.float32)
        roq = RotationalQuantizer(RoQConfig(dim=128, num_bits=1))

        quantized = roq.quantize(embeddings, store=False)

        assert roq.effective_dim == 256
        assert quantized["query_bits"] == 5
        assert np.allclose(quantized["scales"], 2.0)
        assert np.allclose(quantized["offsets"], -1.0)

    def test_build_1bit_query_triton_inputs(self):
        queries = np.random.randn(4, 128).astype(np.float32)
        roq = RotationalQuantizer(RoQConfig(dim=128, num_bits=1))

        codes, meta = roq.build_1bit_query_triton_inputs(
            queries,
            batch_size=2,
            item_count=2,
            device="cpu",
            include_norm_sq=False,
        )

        assert codes.shape == (2, 2, 5, 8)
        assert meta.shape == (2, 2, 3)
        assert codes.dtype == torch.int32
        assert meta.dtype == torch.float32
        assert torch.isfinite(meta).all()

    def test_build_1bit_doc_triton_inputs(self):
        embeddings = np.random.randn(4, 128).astype(np.float32)
        roq = RotationalQuantizer(RoQConfig(dim=128, num_bits=1))

        quantized = roq.quantize(embeddings, store=False)
        doc_words = roq.build_1bit_doc_triton_inputs(
            quantized,
            batch_size=2,
            item_count=2,
            device="cpu",
        )

        assert doc_words.shape == (2, 2, 8)
        assert doc_words.dtype == torch.int32

    def test_1bit_screening_prefers_exact_match(self):
        rng = np.random.default_rng(7)
        query = rng.normal(size=(1, 128)).astype(np.float32)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        opposite = -query
        distractors = rng.normal(size=(14, 128)).astype(np.float32)
        distractors /= np.linalg.norm(distractors, axis=1, keepdims=True)
        data = np.concatenate([query, opposite, distractors], axis=0)

        roq = RotationalQuantizer(RoQConfig(dim=128, num_bits=1))
        roq.quantize(data, store=True)

        indices, scores = roq.search(query, top_k=3)

        assert indices[0] == 0
        assert scores[0] > scores[1]

    def test_2bit_grouped_quantization_roundtrip(self):
        embeddings = np.random.randn(6, 128).astype(np.float32)
        roq = RotationalQuantizer(RoQConfig(dim=128, num_bits=2))

        quantized = roq.quantize(embeddings, store=False)
        decoded = roq.decode(quantized["codes"], quantized["scales"], quantized["offsets"]).numpy()

        assert quantized["meta_layout"] == "grouped"
        assert quantized["group_size"] == 16
        assert quantized["scales"].shape == (6, 8)
        assert quantized["offsets"].shape == (6, 8)
        assert quantized["code_sums"].shape == (6, 8)
        assert decoded.shape == embeddings.shape

    def test_build_triton_meta_for_scalar_2bit(self):
        embeddings = np.random.randn(4, 128).astype(np.float32)
        roq = RotationalQuantizer(RoQConfig(dim=128, num_bits=2, group_size=128))

        quantized = roq.quantize(embeddings, store=False)
        meta = roq.build_triton_meta(quantized)
        codes, stacked_meta = roq.stack_triton_inputs(quantized, batch_size=2, item_count=2, device="cpu")

        assert quantized["meta_layout"] == "scalar"
        assert meta.shape == (4, 4)
        assert codes.shape == (2, 2, 32)
        assert stacked_meta.shape == (2, 2, 4)

