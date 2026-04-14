"""
Audit benchmark: realistic clustered data to evaluate GEM router recall + latency.

Creates synthetic embeddings with clear cluster structure (like real ColBERT/ColPali
embeddings) and measures recall vs brute-force MaxSim.
"""
import time
import numpy as np
from latence_gem_router import PyGemRouter

DIM = 128
N_TOPICS = 20
DOCS_PER_TOPIC = 500
N_DOCS = N_TOPICS * DOCS_PER_TOPIC
VECS_PER_DOC = 32
N_QUERIES = 100

np.random.seed(42)

# Generate topic centroids
topic_centroids = np.random.randn(N_TOPICS, DIM).astype(np.float32)
topic_centroids /= np.linalg.norm(topic_centroids, axis=1, keepdims=True)

# Generate documents: each doc's tokens are close to its topic centroid
print(f"Generating {N_DOCS} docs ({N_TOPICS} topics x {DOCS_PER_TOPIC} per topic), "
      f"{VECS_PER_DOC} tokens/doc, dim={DIM}")
doc_topics = []
all_vecs = []
offsets = []
offset = 0
for topic_id in range(N_TOPICS):
    centroid = topic_centroids[topic_id]
    for _ in range(DOCS_PER_TOPIC):
        noise = np.random.randn(VECS_PER_DOC, DIM).astype(np.float32) * 0.3
        doc_vecs = centroid[None, :] + noise
        doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        all_vecs.append(doc_vecs)
        offsets.append((offset, offset + VECS_PER_DOC))
        offset += VECS_PER_DOC
        doc_topics.append(topic_id)

data = np.concatenate(all_vecs, axis=0).astype(np.float32)
doc_ids = list(range(N_DOCS))
print(f"Total vectors: {data.shape[0]}")

# Build router
print("\nBuilding GEM router...")
router = PyGemRouter(dim=DIM)
t0 = time.perf_counter()
router.build(data, doc_ids, offsets, n_fine=256, n_coarse=64, max_kmeans_iter=30)
build_s = time.perf_counter() - t0
print(f"Build: {build_s:.2f}s  |  {router.n_fine()} fine, {router.n_coarse()} coarse")

# Generate queries: each is from a random topic
queries = []
query_topics = []
for _ in range(N_QUERIES):
    topic = np.random.randint(N_TOPICS)
    centroid = topic_centroids[topic]
    noise = np.random.randn(16, DIM).astype(np.float32) * 0.25
    q = centroid[None, :] + noise
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    queries.append(q.astype(np.float32))
    query_topics.append(topic)

# Brute-force top-K
print("\nComputing brute-force MaxSim baselines...")
bf_topks = {}
bf_latencies = []
for iq, q in enumerate(queries):
    t0 = time.perf_counter()
    scores = np.empty(N_DOCS, dtype=np.float32)
    for i in range(N_DOCS):
        doc = data[offsets[i][0]:offsets[i][1]]
        sim = q @ doc.T
        scores[i] = sim.max(axis=1).sum()
    bf_latencies.append((time.perf_counter() - t0) * 1000)
    bf_topks[iq] = {
        10: set(np.argsort(scores)[-10:][::-1]),
        50: set(np.argsort(scores)[-50:][::-1]),
        100: set(np.argsort(scores)[-100:][::-1]),
    }
bf_mean = sum(bf_latencies) / len(bf_latencies)
print(f"Brute-force mean: {bf_mean:.1f}ms")

# GEM Router routing
print("\nBenchmarking GEM router...")
for n_probes in [2, 4, 8]:
    for max_cand in [100, 200, 500]:
        route_latencies = []
        recall_10 = []
        recall_50 = []
        recall_100 = []
        for iq, q in enumerate(queries):
            t0 = time.perf_counter()
            results = router.route_query(q, n_probes=n_probes, max_candidates=max_cand)
            route_latencies.append((time.perf_counter() - t0) * 1000)
            router_set = set(int(d) for d, _ in results)
            recall_10.append(len(bf_topks[iq][10] & router_set) / 10)
            recall_50.append(len(bf_topks[iq][50] & router_set) / 50)
            recall_100.append(len(bf_topks[iq][100] & router_set) / 100)

        mean_lat = sum(route_latencies) / len(route_latencies)
        speedup = bf_mean / mean_lat
        r10 = sum(recall_10) / len(recall_10)
        r50 = sum(recall_50) / len(recall_50)
        r100 = sum(recall_100) / len(recall_100)
        print(f"  probes={n_probes} cand={max_cand:4d}  |  lat={mean_lat:6.2f}ms  "
              f"speedup={speedup:5.1f}x  |  R@10={r10:.3f}  R@50={r50:.3f}  R@100={r100:.3f}")

# Promotion gates
print("\n--- PROMOTION GATES ---")
best_r10, best_r50, best_r100 = 0, 0, 0
for n_probes in [4, 8]:
    for max_cand in [200, 500]:
        recall_10 = []
        recall_50 = []
        recall_100 = []
        for iq, q in enumerate(queries):
            results = router.route_query(q, n_probes=n_probes, max_candidates=max_cand)
            router_set = set(int(d) for d, _ in results)
            recall_10.append(len(bf_topks[iq][10] & router_set) / 10)
            recall_50.append(len(bf_topks[iq][50] & router_set) / 50)
            recall_100.append(len(bf_topks[iq][100] & router_set) / 100)
        r10 = sum(recall_10) / len(recall_10)
        r50 = sum(recall_50) / len(recall_50)
        r100 = sum(recall_100) / len(recall_100)
        best_r10 = max(best_r10, r10)
        best_r50 = max(best_r50, r50)
        best_r100 = max(best_r100, r100)

gates = {
    "recall@10 >= 0.80": best_r10 >= 0.80,
    "recall@50 >= 0.60": best_r50 >= 0.60,
    "recall@100 >= 0.50": best_r100 >= 0.50,
    "speedup >= 10x": speedup >= 10.0,
}
for name, passed in gates.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
all_pass = all(gates.values())
print(f"\nOverall: {'ALL GATES PASSED' if all_pass else 'SOME GATES FAILED'}")
