"""
Pareto frontier evaluation: recall vs. latency for the full 7.5K corpus.

Evaluates three pipelines:
  1. CPU-only graph search (various ef values)
  2. GPU 2-stage: GPU qCH brute-force → Triton MaxSim reranking
  3. CPU graph search (high ef) → Triton MaxSim reranking

Saves results to benchmarks/results/pareto_frontier.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latence_gem_index import GemSegment
from tests.data.msmarco_loader import load_combined_dataset, MSMARCODataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INDEX_PATH = Path.home() / ".cache" / "voyager-qa" / "full_corpus_index.bin"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "pareto_frontier.json"


def load_or_build_index(ds: MSMARCODataset) -> GemSegment:
    seg = GemSegment()
    if INDEX_PATH.exists():
        log.info("Loading cached index from %s", INDEX_PATH)
        seg.load(str(INDEX_PATH))
        return seg

    log.info("Building index for %d docs ...", ds.n_docs)
    t0 = time.time()
    seg.build(
        ds.all_vectors,
        ds.doc_ids,
        ds.offsets,
        n_fine=0,
        n_coarse=32,
        max_degree=32,
        ef_construction=200,
        max_kmeans_iter=20,
        ctop_r=3,
        use_emd=False,
        dual_graph=True,
        store_raw_vectors=True,
    )
    build_time = time.time() - t0
    log.info("Build done in %.1fs", build_time)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    seg.save(str(INDEX_PATH))
    log.info("Index saved to %s", INDEX_PATH)
    return seg


def brute_force_gt(ds: MSMARCODataset, seg: GemSegment, n_eval: int, k: int = 100):
    """Compute ground truth via Rust brute_force_maxsim for n_eval queries."""
    gts = []
    for qi in range(n_eval):
        qv = ds.query_vecs[qi]
        gt = seg.brute_force_maxsim(qv, k)
        gts.append(gt)
    return gts


def recall_at_k(results, gt, k):
    gt_ids = {doc_id for doc_id, _ in gt[:k]}
    res_ids = {doc_id for doc_id, _ in results[:k]}
    if not gt_ids:
        return 1.0
    return len(gt_ids & res_ids) / len(gt_ids)


def cpu_recall_sweep(ds: MSMARCODataset, seg: GemSegment, gts: list, n_eval: int):
    """CPU-only search at various ef values."""
    ef_values = [16, 32, 64, 100, 200, 400, 800, 1500, 2500]
    results = []

    for ef in ef_values:
        recalls_10 = []
        recalls_100 = []
        latencies = []

        for qi in range(n_eval):
            qv = ds.query_vecs[qi]
            t0 = time.perf_counter()
            res = seg.search(qv, k=100, ef=ef, n_probes=4)
            lat = (time.perf_counter() - t0) * 1e6
            latencies.append(lat)
            recalls_10.append(recall_at_k(res, gts[qi], 10))
            recalls_100.append(recall_at_k(res, gts[qi], 100))

        row = {
            "pipeline": "cpu_only",
            "ef": ef,
            "R@10": float(np.mean(recalls_10)),
            "R@100": float(np.mean(recalls_100)),
            "p50_us": float(np.percentile(latencies, 50)),
            "p95_us": float(np.percentile(latencies, 95)),
        }
        results.append(row)
        log.info("CPU ef=%d  R@10=%.4f  R@100=%.4f  p50=%.0fus  p95=%.0fus",
                 ef, row["R@10"], row["R@100"], row["p50_us"], row["p95_us"])

    return results


def gpu_qch_maxsim_pipeline(ds: MSMARCODataset, seg: GemSegment, gts: list, n_eval: int):
    """GPU 2-stage: GPU qCH brute-force → top-N → Triton MaxSim reranking."""
    from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer
    from voyager_index._internal.kernels.maxsim import fast_colbert_scores

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No GPU available; skipping GPU pipeline")
        return []

    log.info("Initializing GPU qCH scorer ...")
    scorer = GpuQchScorer.from_gem_segment(seg, device=device)

    candidate_counts = [50, 100, 200, 500, 1000, 2500]
    results = []

    for n_cand in candidate_counts:
        recalls_10 = []
        latencies = []

        for qi in range(n_eval):
            qv = ds.query_vecs[qi]
            qv_t = torch.from_numpy(qv).float().to(device)

            t0 = time.perf_counter()

            # Stage 1: GPU qCH brute-force scoring of all docs
            proxy_scores = scorer.score_query(qv_t)  # (n_docs,) lower=better
            # Take top-N candidates (lowest proxy score = closest)
            topk_vals, topk_idxs = torch.topk(proxy_scores, n_cand, largest=False)
            cand_indices = topk_idxs.cpu().numpy()

            # Stage 2: Triton MaxSim reranking of candidates
            cand_doc_vecs = [ds.doc_vecs[idx] for idx in cand_indices]
            query_tensor = torch.from_numpy(qv).float().unsqueeze(0).to(device)
            cand_tensors = [torch.from_numpy(v).float().to(device) for v in cand_doc_vecs]

            maxsim_scores = fast_colbert_scores(query_tensor, cand_tensors)
            maxsim_scores = maxsim_scores.squeeze(0)

            # Map back to doc_ids and sort (higher MaxSim = better)
            ranked_indices = torch.argsort(maxsim_scores, descending=True).cpu().numpy()
            final_results = [(int(ds.doc_ids[cand_indices[ri]]), float(-maxsim_scores[ri].cpu())) for ri in ranked_indices]

            lat = (time.perf_counter() - t0) * 1e6
            latencies.append(lat)
            recalls_10.append(recall_at_k(final_results, gts[qi], 10))

        row = {
            "pipeline": "gpu_qch_maxsim",
            "n_candidates": n_cand,
            "R@10": float(np.mean(recalls_10)),
            "p50_us": float(np.percentile(latencies, 50)),
            "p95_us": float(np.percentile(latencies, 95)),
        }
        results.append(row)
        log.info("GPU qCH→MaxSim n_cand=%d  R@10=%.4f  p50=%.0fus  p95=%.0fus",
                 n_cand, row["R@10"], row["p50_us"], row["p95_us"])

    return results


def cpu_graph_gpu_rerank_pipeline(ds: MSMARCODataset, seg: GemSegment, gts: list, n_eval: int):
    """CPU graph search (high ef) → Triton MaxSim GPU reranking."""
    from voyager_index._internal.kernels.maxsim import fast_colbert_scores

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No GPU for reranking; skipping CPU+GPU pipeline")
        return []

    configs = [
        (200, 50), (400, 100), (800, 200), (1500, 500), (2500, 1000),
    ]
    results = []

    for ef, n_rerank in configs:
        recalls_10 = []
        latencies = []

        for qi in range(n_eval):
            qv = ds.query_vecs[qi]

            t0 = time.perf_counter()

            # Stage 1: CPU graph search
            raw_results = seg.search(qv, k=n_rerank, ef=ef, n_probes=4)
            cand_doc_ids_scores = raw_results[:n_rerank]

            # Stage 2: Triton MaxSim reranking
            cand_doc_ids = [int(did) for did, _ in cand_doc_ids_scores]
            cand_doc_vecs = [ds.doc_vecs[did] for did in cand_doc_ids]

            query_tensor = torch.from_numpy(qv).float().unsqueeze(0).to(device)
            cand_tensors = [torch.from_numpy(v).float().to(device) for v in cand_doc_vecs]

            maxsim_scores = fast_colbert_scores(query_tensor, cand_tensors)
            maxsim_scores = maxsim_scores.squeeze(0)

            ranked_indices = torch.argsort(maxsim_scores, descending=True).cpu().numpy()
            final_results = [(cand_doc_ids[ri], float(-maxsim_scores[ri].cpu())) for ri in ranked_indices]

            lat = (time.perf_counter() - t0) * 1e6
            latencies.append(lat)
            recalls_10.append(recall_at_k(final_results, gts[qi], 10))

        row = {
            "pipeline": "cpu_graph_gpu_rerank",
            "ef": ef,
            "n_rerank": n_rerank,
            "R@10": float(np.mean(recalls_10)),
            "p50_us": float(np.percentile(latencies, 50)),
            "p95_us": float(np.percentile(latencies, 95)),
        }
        results.append(row)
        log.info("CPU(ef=%d)→GPU rerank(n=%d)  R@10=%.4f  p50=%.0fus  p95=%.0fus",
                 ef, n_rerank, row["R@10"], row["p50_us"], row["p95_us"])

    return results


def main():
    parser = argparse.ArgumentParser(description="Pareto frontier: recall vs latency")
    parser.add_argument("--n-eval", type=int, default=200, help="Number of queries to evaluate")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU pipelines")
    args = parser.parse_args()

    log.info("Loading combined 7.5K dataset ...")
    ds = load_combined_dataset()
    log.info("Dataset: %d docs, %d queries, dim=%d", ds.n_docs, ds.n_queries, ds.dim)

    seg = load_or_build_index(ds)

    n_eval = min(args.n_eval, ds.n_queries)
    log.info("Computing brute-force ground truth for %d queries ...", n_eval)
    gts = brute_force_gt(ds, seg, n_eval, k=100)

    log.info("=== CPU-only recall sweep ===")
    cpu_results = cpu_recall_sweep(ds, seg, gts, n_eval)

    gpu_qch_results = []
    cpu_gpu_results = []
    if not args.skip_gpu:
        log.info("=== GPU qCH → MaxSim pipeline ===")
        gpu_qch_results = gpu_qch_maxsim_pipeline(ds, seg, gts, n_eval)

        log.info("=== CPU graph → GPU MaxSim rerank pipeline ===")
        cpu_gpu_results = cpu_graph_gpu_rerank_pipeline(ds, seg, gts, n_eval)

    all_results = {
        "dataset": {"n_docs": ds.n_docs, "n_queries": ds.n_queries, "n_eval": n_eval},
        "cpu_only": cpu_results,
        "gpu_qch_maxsim": gpu_qch_results,
        "cpu_graph_gpu_rerank": cpu_gpu_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    log.info("Results saved to %s", RESULTS_PATH)

    # Print summary table
    print("\n" + "=" * 85)
    print(f"{'Pipeline':<30} {'Config':<15} {'R@10':>8} {'p50(ms)':>10} {'p95(ms)':>10}")
    print("-" * 85)
    for r in cpu_results:
        print(f"{'CPU-only':<30} {'ef=' + str(r['ef']):<15} {r['R@10']:>8.4f} {r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    for r in gpu_qch_results:
        print(f"{'GPU qCH→MaxSim':<30} {'n=' + str(r['n_candidates']):<15} {r['R@10']:>8.4f} {r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    for r in cpu_gpu_results:
        config = f"ef={r['ef']},n={r['n_rerank']}"
        print(f"{'CPU→GPU rerank':<30} {config:<15} {r['R@10']:>8.4f} {r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
