
import time
import numpy as np
import shutil
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference.index_core.hnsw_manager import HnswSegmentManager

def run_benchmark():
    print("--- Benchmark: HNSW Retrieval Latency ---")
    data_path = "./data/benchmark_shard"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        
    dim = 768 # Realistic dimension (e.g., BERT/ColBERT)
    num_points = 10000 
    
    print(f"Initializing Manager (dim={dim}, num_points={num_points})...")
    # on_disk=False for pure RAM speed test first, or True for realistic cold start? 
    # Let's do in-memory for "best case" raw retrieval speed first.
    manager = HnswSegmentManager(data_path, dim=dim, on_disk=True)
    
    # Generate data
    print("Generating dummy data...")
    vectors = np.random.rand(num_points, dim).astype(np.float32)
    ids = list(range(num_points))
    payloads = [{"i": i, "data": "x" * 100} for i in ids] # ~100 bytes payload
    
    start_add = time.time()
    manager.add(vectors, ids=ids, payloads=payloads)
    print(f"Ingestion took: {time.time() - start_add:.2f}s")
    
    # Flush to ensure persistence overhead is settled if any
    manager.active_segment.flush()
    
    # Benchmark Retrieval
    batch_sizes = [1, 10, 100, 500]
    num_trials = 100
    
    print("\n--- Latency Results ---")
    print(f"{'Batch Size':<12} | {'Avg Latency (ms)':<18} | {'P99 Latency (ms)':<18} | {'Throughput (docs/s)':<20}")
    print("-" * 75)
    
    for b in batch_sizes:
        latencies = []
        for _ in range(num_trials):
            # Random selection of IDs
            batch_ids = np.random.choice(ids, b, replace=False).tolist()
            
            t0 = time.time()
            _ = manager.retrieve(batch_ids)
            t1 = time.time()
            latencies.append((t1 - t0) * 1000) # ms
            
        avg_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99)
        throughput = (b * num_trials) / (sum(latencies) / 1000)
        
        print(f"{b:<12} | {avg_lat:<18.3f} | {p99_lat:<18.3f} | {throughput:<20.1f}")
        
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_benchmark()
