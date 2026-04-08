import numpy as np
from voyager_index._internal.inference.quantization.rotational import RotationalQuantizer, RoQConfig

def benchmark_binary_recall():
    """Validate recall for 1-bit quantization (32x compression)."""
    print("\n=== Binary Recall Benchmark (32x Compression) ===")
    
    from sklearn.neighbors import NearestNeighbors
    
    # Check 1536 dim (production target)
    dim = 1536
    name = "OpenAI"
    n = 5000
    n_queries = 100
    k = 10
    
    # num_bits=1 for 32x compression
    config = RoQConfig(dim=dim, num_rounds=3, block_size=256, num_bits=1)
    roq = RotationalQuantizer(config)
    
    # Generate data
    np.random.seed(42)
    data = np.random.randn(n, dim).astype(np.float32)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    
    # Generate Correlated Queries (Synthetic near-neighbors)
    # Pick random data points and adding noise
    query_indices = np.random.choice(n, n_queries, replace=False)
    noise = np.random.randn(n_queries, dim).astype(np.float32)
    noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
    # Mix: mostly data, little noise. 
    # q = 0.9 * d + 0.1 * noise
    queries = 0.9 * data[query_indices] + 0.1 * noise
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Ground truth
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(data)
    exact_dist, exact_idx = nbrs.kneighbors(queries)
    
    # Quantized search
    res = roq.quantize(data, store=True)
    
    # DEBUG: Check bit density
    codes = res['codes']
    unpacked = np.unpackbits(codes, axis=1)
    density = unpacked.mean()
    print(f"Bit density: {density:.4f}")
    
    q_res = roq.quantize(queries, store=False)
    roq_idx, roq_dists = roq.search(queries, top_k=k)
    
    # DEBUG: Correlation for first query's top neighbor (true neighbor)
    true_neighbor_idx = exact_idx[0,0]
    true_neighbor_dist = exact_dist[0,0] # Euclidean dist
    # Convert Euclidean to Cosine Similarity: dist^2 = 2 - 2sim -> sim = 1 - dist^2/2
    true_neighbor_sim = 1.0 - (true_neighbor_dist**2)/2.0
    
    # Get Hamming stat
    # Need to find distance to true_neighbor_idx in roq
    # Manual compute
    q0_u = np.unpackbits(q_res['codes'][0])
    d0_u = np.unpackbits(res['codes'][true_neighbor_idx])
    # Keep only effective dim bits? unpackbits gives 8*bytes. 
    # n_bytes = 1536/8 = 192. So 1536 bits. Matches.
    hamming = np.sum(q0_u != d0_u)
    print(f"Query 0 True Neighbor ({true_neighbor_idx}):")
    print(f"  Float Cosine Sim: {true_neighbor_sim:.4f}")
    print(f"  Hamming Dist: {hamming} / {dim} ({hamming/dim:.4f})")
    print(f"  Hamming Sim: {1.0 - 2.0 * hamming/dim:.4f} (Ideal: match Cosine)")
    
    # DEBUG: Stats
    print(f"Avg Hamming Dist to Top-1: {roq_dists[:,0].mean():.2f}")
    
    # Compute recall
    total_recall = 0
    for i in range(n_queries):
        truth = set(exact_idx[i])
        found = set(roq_idx[i])
        total_recall += len(truth & found) / k
    
    avg_recall = total_recall / n_queries
    # Target >90% for binary
    status = "✅" if avg_recall >= 0.85 else "⚠️"
    print(f"{name:8s} (dim={dim:4d}) 1-bit: Recall@{k} = {avg_recall:.4f} {status}")

if __name__ == "__main__":
    benchmark_binary_recall()
