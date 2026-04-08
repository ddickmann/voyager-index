"""
Comprehensive benchmark for Rotational Quantization.
Tests speed and accuracy against targets.
"""

import numpy as np
import torch
import time

from voyager_index._internal.inference.quantization.rotational import RotationalQuantizer, RoQConfig, FastWalshHadamard

def benchmark_rotation_speed():
    """Benchmark FWHT rotation speed. Target: <10μs for 1536-dim."""
    print("\n=== FWHT Rotation Speed Benchmark ===")
    
    for dim in [128, 384, 768, 1536]:
        fwht = FastWalshHadamard(dim, num_rounds=3, block_size=min(256, max(64, dim)))
        x = torch.randn(1, dim)
        
        # Warmup
        for _ in range(10):
            fwht.forward(x)
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            fwht.forward(x)
            elapsed = (time.perf_counter() - start) * 1e6  # microseconds
            times.append(elapsed)
        
        avg = np.mean(times)
        std = np.std(times)
        print(f"dim={dim:4d}: {avg:7.2f} ± {std:.2f} μs")

def benchmark_quantization_speed():
    """Benchmark full quantization speed."""
    print("\n=== Quantization Speed Benchmark ===")
    
    for dim in [128, 384, 768, 1536]:
        config = RoQConfig(dim=dim, num_rounds=3, block_size=min(256, max(64, dim)))
        roq = RotationalQuantizer(config)
        x = np.random.randn(1000, dim).astype(np.float32)
        
        # Warmup
        roq.quantize(x[:100], store=False)
        
        # Benchmark
        start = time.perf_counter()
        roq.quantize(x, store=True)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        per_vec = elapsed / 1000 * 1000  # μs per vector
        print(f"dim={dim:4d}: {elapsed:6.2f} ms for 1000 vectors ({per_vec:.2f} μs/vec)")

def benchmark_search_speed():
    """Benchmark search throughput."""
    print("\n=== Search Speed Benchmark ===")
    
    dim = 1536
    n_index = 10000
    n_queries = 100
    
    config = RoQConfig(dim=dim, num_rounds=3, block_size=256)
    roq = RotationalQuantizer(config)
    
    # Create index
    data = np.random.randn(n_index, dim).astype(np.float32)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    roq.quantize(data, store=True)
    
    # Create queries
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Benchmark
    start = time.perf_counter()
    idx, dists = roq.search(queries, top_k=10)
    elapsed = time.perf_counter() - start
    
    qps = n_queries / elapsed
    print(f"{n_index} vectors, {n_queries} queries: {qps:.0f} QPS ({elapsed*1000:.1f} ms total)")

def benchmark_recall():
    """Validate recall on multiple datasets."""
    print("\n=== Recall Benchmark ===")
    
    from sklearn.neighbors import NearestNeighbors
    
    for dim, name in [(128, "Small"), (384, "MiniLM"), (768, "BERT"), (1536, "OpenAI")]:
        n = 5000
        n_queries = 100
        k = 10
        
        config = RoQConfig(dim=dim, num_rounds=3, block_size=min(256, max(64, dim)))
        roq = RotationalQuantizer(config)
        
        # Generate data
        np.random.seed(42)
        data = np.random.randn(n, dim).astype(np.float32)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        # Ground truth
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(data)
        _, exact_idx = nbrs.kneighbors(queries)
        
        # Quantized search
        roq.quantize(data, store=True)
        roq_idx, _ = roq.search(queries, top_k=k)
        
        # Compute recall
        total_recall = 0
        for i in range(n_queries):
            truth = set(exact_idx[i])
            found = set(roq_idx[i])
            total_recall += len(truth & found) / k
        
        avg_recall = total_recall / n_queries
        status = "✅" if avg_recall >= 0.95 else "⚠️"
        print(f"{name:8s} (dim={dim:4d}): Recall@{k} = {avg_recall:.4f} {status}")

def benchmark_compression():
    """Show compression ratio."""
    print("\n=== Compression Ratio ===")
    
    dim = 1536
    n = 1000
    
    config = RoQConfig(dim=dim, num_rounds=3, block_size=256)
    roq = RotationalQuantizer(config)
    
    data = np.random.randn(n, dim).astype(np.float32)
    result = roq.quantize(data, store=False)
    
    original_size = n * dim * 4  # float32
    quantized_size = n * (roq.effective_dim + 4 + 4 + 4)  # codes + scale + offset + norm_sq
    
    ratio = original_size / quantized_size
    print(f"Original: {original_size / 1024:.1f} KB")
    print(f"Quantized: {quantized_size / 1024:.1f} KB")
    print(f"Compression: {ratio:.2f}x")

if __name__ == "__main__":
    print("=" * 60)
    print("Rotational Quantization Benchmark Suite")
    print("=" * 60)
    
    benchmark_compression()
    benchmark_rotation_speed()
    benchmark_quantization_speed()
    benchmark_recall()
    benchmark_search_speed()
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
