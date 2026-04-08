"""
Integration test for HNSW Segment Manager with Qdrant
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from voyager_index._internal.inference.index_core.hnsw_manager import HnswSegmentManager


@pytest.fixture
def temp_shard_path():
    """Create temporary shard directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_shard"


def test_basic_indexing_and_search(temp_shard_path):
    """Test basic HNSW indexing and search."""
    dim = 128
    n_vectors = 1000
    
    # Create manager
    manager = HnswSegmentManager(
        shard_path=temp_shard_path,
        dim=dim,
        distance_metric="cosine",
        m=16,
        ef_construct=100
    )
    
    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize
    
    # Index vectors
    ids = manager.add(vectors)
    assert len(ids) == n_vectors
    assert manager.total_vectors() == n_vectors
    
    # Search
    query = vectors[0]  # Use first vector as query
    results = manager.search(query, k=10)
    
    assert len(results) > 0
    assert results[0][0] == ids[0], "First result should be the query itself"
    assert results[0][1] > 0.99, "Self-similarity should be ~1.0"
    
    print(f"✓ Indexed {n_vectors} vectors, retrieved {len(results)} neighbors")


def test_payload_filtering(temp_shard_path):
    """Test search with payload filtering."""
    dim = 64
    n_vectors = 500
    
    manager = HnswSegmentManager(
        shard_path=temp_shard_path,
        dim=dim
    )
    
    # Generate vectors with metadata
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    payloads = []
    for i in range(n_vectors):
        payloads.append({
            "role": "evidence" if i % 2 == 0 else "example",
            "density": float(i / n_vectors)
        })
    
    # Index with payloads
    ids = manager.add(vectors, payloads=payloads)
    
    # Search without filter
    query = vectors[0]
    results_no_filter = manager.search(query, k=20)
    assert len(results_no_filter) > 0
    
    # TODO: Enable once filter conversion is implemented in Rust
    # Search with filter (role=evidence)
    # results_filtered = manager.search(query, k=20, filters={"role": "evidence"})
    # assert len(results_filtered) > 0
    # assert len(results_filtered) <= len(results_no_filter)
    
    print(f"✓ Payload filtering test passed (filter impl pending)")


def test_segment_sealing(temp_shard_path):
    """Test active segment sealing and multi-segment search."""
    dim = 32
    
    manager = HnswSegmentManager(
        shard_path=temp_shard_path,
        dim=dim
    )
    
    # Add first batch
    np.random.seed(42)
    batch1 = np.random.randn(100, dim).astype(np.float32)
    ids1 = manager.add(batch1)
    
    # Seal active segment
    manager.seal_active_segment()
    assert len(manager.sealed_segments) == 1
    
    # Add second batch
    batch2 = np.random.randn(100, dim).astype(np.float32)
    ids2 = manager.add(batch2)
    
    # Search should query both segments
    query = batch1[0]
    results = manager.search(query, k=10)
    
    assert len(results) > 0
    assert manager.total_vectors() == 200
    
    # First result should be from first batch (sealed segment)
    assert results[0][0] in ids1
    
    print(f"✓ Multi-segment search works ({len(manager.sealed_segments)} sealed + 1 active)")


def test_recall(temp_shard_path):
    """Verify HNSW recall is acceptable."""
    dim = 128
    n_vectors = 1000
    k = 10
    
    manager = HnswSegmentManager(
        shard_path=temp_shard_path,
        dim=dim,
        m=32,  # Higher M for better recall
        ef_construct=200
    )
    
    # Generate vectors
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    ids = manager.add(vectors)
    
    # Compute ground truth (brute force)
    query = vectors[0]
    similarities = vectors @ query
    ground_truth_ids = np.argsort(similarities)[::-1][:k]
    ground_truth_ids = [ids[i] for i in ground_truth_ids]
    
    # HNSW search
    results = manager.search(query, k=k, ef=k * 4)
    hnsw_ids = [r[0] for r in results]
    
    # Calculate recall
    recall = len(set(hnsw_ids) & set(ground_truth_ids)) / k
    
    assert recall >= 0.95, f"Recall too low: {recall:.2%}"
    print(f"✓ HNSW Recall@{k}: {recall:.2%}")


if __name__ == "__main__":
    import sys
    
    # Run tests manually
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_shard"
        
        print("=" * 60)
        print("Running HNSW Integration Tests")
        print("=" * 60)
        
        try:
            test_basic_indexing_and_search(path)
            print()
            
            path = Path(tmpdir) / "test_shard2"
            test_payload_filtering(path)
            print()
            
            path = Path(tmpdir) / "test_shard3"
            test_segment_sealing(path)
            print()
            
            path = Path(tmpdir) / "test_shard4"
            test_recall(path)
            print()
            
            print("=" * 60)
            print("ALL TESTS PASSED ✓")
            print("=" * 60)
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
