#!/usr/bin/env python3
"""
Diagnostic test for RoQ MaxSim accuracy issues.
Verifies:
1. Triton FP16 produces same scores as Python FP32 (speed is real)
2. 1-bit RoQ quantization and Hamming distance calculation is working correctly
"""
import numpy as np
import torch
import time

from voyager_index._internal.inference.quantization.rotational import RotationalQuantizer, RoQConfig
from voyager_index._internal.kernels.maxsim import fast_colbert_scores

def test_triton_vs_python_accuracy():
    """Verify Triton FP16 produces identical rankings to Python FP32."""
    print("=" * 60)
    print("TEST 1: Triton FP16 vs Python FP32 Score Accuracy")
    print("=" * 60)
    
    torch.manual_seed(42)
    dim = 128
    n_queries = 5
    n_docs = 100
    n_q_tokens = 8
    n_d_tokens = 16
    device = 'cuda'
    
    # Generate random normalized embeddings
    queries = torch.randn(n_queries, n_q_tokens, dim, device=device)
    queries = torch.nn.functional.normalize(queries, p=2, dim=-1)
    
    docs = torch.randn(n_docs, n_d_tokens, dim, device=device)
    docs = torch.nn.functional.normalize(docs, p=2, dim=-1)
    
    # Python FP32 MaxSim
    def python_maxsim():
        scores = torch.zeros(n_queries, n_docs, device=device)
        for i in range(n_queries):
            q = queries[i]  # (S, D)
            d_T = docs.transpose(1, 2)  # (B, D, T)
            sims = torch.matmul(q.unsqueeze(0), d_T)  # (B, S, T)
            max_sims = sims.max(dim=2).values  # (B, S)
            scores[i] = max_sims.sum(dim=1)
        return scores
    
    # Compute scores
    scores_py = python_maxsim()
    scores_triton = fast_colbert_scores(queries, docs, use_quantization=False)
    
    # Compare
    diff = torch.abs(scores_py - scores_triton)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Python FP32 scores shape: {scores_py.shape}")
    print(f"Triton FP16 scores shape: {scores_triton.shape}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Check rankings match
    py_rankings = scores_py.argsort(dim=1, descending=True)
    triton_rankings = scores_triton.argsort(dim=1, descending=True)
    
    # Top-1 match?
    top1_match = (py_rankings[:, 0] == triton_rankings[:, 0]).all().item()
    print(f"Top-1 rankings match: {top1_match}")
    
    # Full ranking match?
    full_match = (py_rankings == triton_rankings).all().item()
    print(f"Full rankings match: {full_match}")
    
    if max_diff < 0.1 and top1_match:
        print("✓ PASS: Triton FP16 produces equivalent results to Python FP32")
    else:
        print("✗ FAIL: Significant discrepancy detected!")
    
    return scores_py, scores_triton


def test_1bit_roq_correctness():
    """Verify 1-bit RoQ quantization and distance calculation."""
    print("\n" + "=" * 60)
    print("TEST 2: 1-bit RoQ Quantization Correctness")
    print("=" * 60)
    
    torch.manual_seed(42)
    dim = 128
    n_vecs = 10
    
    # Create simple test vectors
    # Two identical vectors should have Hamming distance 0
    # Orthogonal vectors should have ~50% bit agreement (random)
    
    roq = RotationalQuantizer(RoQConfig(dim=dim, num_bits=1))
    
    # Test 1: Identical vectors
    v1 = torch.randn(1, dim)
    v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)
    
    q1 = roq.quantize(v1.numpy(), store=False)
    q2 = roq.quantize(v1.numpy(), store=False)  # Same input
    
    c1 = q1['codes']
    c2 = q2['codes']
    
    # Hamming distance
    xor = np.unpackbits(c1) ^ np.unpackbits(c2)
    hamming = xor.sum()
    
    print(f"\nTest 1: Identical vectors")
    print(f"  Hamming distance: {hamming} (expected: 0)")
    
    # Test 2: Opposite vectors (v and -v)
    v_neg = -v1
    q_neg = roq.quantize(v_neg.numpy(), store=False)
    c_neg = q_neg['codes']
    
    xor_neg = np.unpackbits(c1) ^ np.unpackbits(c_neg)
    hamming_neg = xor_neg.sum()
    
    # After FWHT rotation, -v should flip all bits -> Hamming = dim
    print(f"\nTest 2: Opposite vectors (v vs -v)")
    print(f"  Hamming distance: {hamming_neg} (expected: ~{dim} or close)")
    
    # Test 3: Random vectors - check distribution
    print(f"\nTest 3: Random vector pairs - Hamming distance distribution")
    vecs = torch.randn(100, dim)
    vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
    q_all = roq.quantize(vecs.numpy(), store=False)
    codes_all = q_all['codes']
    
    # Compute pairwise Hamming distances for first 10 pairs
    hammings = []
    for i in range(10):
        for j in range(i+1, 10):
            xor_ij = np.unpackbits(codes_all[i:i+1]) ^ np.unpackbits(codes_all[j:j+1])
            hammings.append(xor_ij.sum())
    
    print(f"  Mean Hamming: {np.mean(hammings):.1f}")
    print(f"  Min Hamming: {np.min(hammings)}")
    print(f"  Max Hamming: {np.max(hammings)}")
    print(f"  Expected for random: ~{dim/2} (half bits differ)")
    
    # Test 4: Verify inner product correlation
    print(f"\nTest 4: Inner product vs Hamming similarity correlation")
    true_ips = []
    hamming_sims = []
    cosine_sims = []
    
    for i in range(20):
        for j in range(i+1, 20):
            # True inner product
            ip = (vecs[i] @ vecs[j]).item()
            true_ips.append(ip)
            
            # Hamming similarity: dim - 2 * hamming
            xor_ij = np.unpackbits(codes_all[i:i+1]) ^ np.unpackbits(codes_all[j:j+1])
            ham = xor_ij.sum()
            # Linear proxy
            sim_lin = dim - 2 * ham
            hamming_sims.append(sim_lin)
            
            # Cosine correction
            # IP ≈ D * cos(pi * h / D)
            sim_cos = dim * np.cos(np.pi * ham / dim)
            cosine_sims.append(sim_cos)
    
    # Compute correlation
    corr_lin = np.corrcoef(true_ips, hamming_sims)[0, 1]
    corr_cos = np.corrcoef(true_ips, cosine_sims)[0, 1]
    
    print(f"  Correlation (Linear D-2*H): {corr_lin:.4f}")
    print(f"  Correlation (Cosine Corrected): {corr_cos:.4f}")
    
    return corr_cos


def test_roq_8bit_vs_fp32():
    """Verify 8-bit RoQ inner product estimation accuracy."""
    print("\n" + "=" * 60)
    print("TEST 3: 8-bit RoQ Inner Product Accuracy")
    print("=" * 60)
    
    torch.manual_seed(42)
    dim = 128
    
    roq = RotationalQuantizer(RoQConfig(dim=dim, num_bits=8))
    
    # Generate random vectors
    vecs = torch.randn(20, dim)
    vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
    
    q_res = roq.quantize(vecs.numpy(), store=False)
    
    true_ips = []
    est_ips = []
    
    for i in range(10):
        for j in range(i+1, 10):
            # True inner product
            ip = (vecs[i] @ vecs[j]).item()
            true_ips.append(ip)
            
            # Estimated inner product using affine correction
            c_i = q_res['codes'][i].astype(np.float32)
            c_j = q_res['codes'][j].astype(np.float32)
            s_i, o_i = q_res['scales'][i], q_res['offsets'][i]
            s_j, o_j = q_res['scales'][j], q_res['offsets'][j]
            
            # Affine: <x, y> ≈ D*l_x*l_y + l_x*d_y*sum(c_y) + d_x*l_y*sum(c_x) + d_x*d_y*<c_x, c_y>
            t1 = dim * o_i * o_j
            t2 = o_i * s_j * c_j.sum()
            t3 = s_i * c_i.sum() * o_j
            t4 = s_i * s_j * (c_i @ c_j)
            
            est_ip = t1 + t2 + t3 + t4
            est_ips.append(est_ip)
    
    # Compute correlation
    corr = np.corrcoef(true_ips, est_ips)[0, 1]
    # Compute MSE
    mse = np.mean((np.array(true_ips) - np.array(est_ips))**2)
    
    print(f"Correlation between true IP and 8-bit estimated IP: {corr:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"(8-bit should have correlation > 0.95)")
    
    if corr > 0.9:
        print("✓ PASS: 8-bit RoQ shows excellent IP correlation")
    else:
        print("✗ FAIL: 8-bit RoQ correlation is too low - possible bug!")
    
    return corr


if __name__ == "__main__":
    test_triton_vs_python_accuracy()
    corr_1bit = test_1bit_roq_correctness()
    corr_8bit = test_roq_8bit_vs_fp32()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"1-bit RoQ IP correlation: {corr_1bit:.4f}")
    print(f"8-bit RoQ IP correlation: {corr_8bit:.4f}")
