"""
100K GEM Index — Build, Quantize, and Evaluate.

Phases:
  A. Build index with aggressive hyperparams (n_fine=2048, n_coarse=128, ...)
  B. Compute FP32 brute-force ground truth
  C. Quantize all doc vectors to ROQ 4-bit
  D. 4-pipeline Pareto frontier evaluation:
     1. CPU-only (graph search, proxy scores)
     2. GPU qCH → FP16 MaxSim (quality ceiling)
     3. GPU qCH → ROQ 4-bit MaxSim (production target)
     4. CPU graph → ROQ 4-bit rerank (hybrid)

Usage:
  python benchmarks/eval_100k.py [--n-eval 300] [--skip-build] [--skip-gpu]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latence_gem_index import GemSegment
from tests.data.msmarco_loader import load_beir_100k_dataset, MSMARCODataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "voyager-qa"
INDEX_PATH = CACHE_DIR / "beir_100k_index.bin"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "eval_100k.json"

BUILD_PARAMS = dict(
    n_fine=2048,
    n_coarse=128,
    max_degree=48,
    ef_construction=400,
    max_kmeans_iter=20,
    ctop_r=4,
    use_emd=False,
    dual_graph=True,
    store_raw_vectors=True,
)


def build_index(ds: MSMARCODataset) -> GemSegment:
    """Build and save the 100K index."""
    seg = GemSegment()
    log.info("Building 100K index: %d docs, %d vectors, dim=%d",
             ds.n_docs, ds.all_vectors.shape[0], ds.dim)
    log.info("Params: %s", BUILD_PARAMS)

    t0 = time.time()
    seg.build(ds.all_vectors, ds.doc_ids, ds.offsets, **BUILD_PARAMS)
    build_time = time.time() - t0
    log.info("Build completed in %.1fs (%.1f min)", build_time, build_time / 60)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    seg.save(str(INDEX_PATH))
    size_mb = INDEX_PATH.stat().st_size / 1024**2
    log.info("Index saved to %s (%.1f MB)", INDEX_PATH, size_mb)
    return seg


def load_index() -> GemSegment:
    seg = GemSegment()
    log.info("Loading index from %s", INDEX_PATH)
    seg.load(str(INDEX_PATH))
    return seg


def graph_health(seg: GemSegment) -> dict:
    """Report graph connectivity metrics."""
    n_comp, gf, cr = seg.graph_connectivity_report()
    n_nodes = seg.n_nodes()
    n_edges = seg.n_edges()
    mean_deg = 2 * n_edges / max(n_nodes, 1)
    report = {
        "n_docs": seg.n_docs(),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": mean_deg,
        "n_components": n_comp,
        "giant_component_frac": gf,
        "cross_cluster_edge_ratio": cr,
    }
    log.info("Graph health: comp=%d, giant=%.4f, cross_ratio=%.4f, "
             "nodes=%d, edges=%d, mean_deg=%.1f",
             n_comp, gf, cr, n_nodes, n_edges, mean_deg)
    return report


def compute_ground_truth(ds: MSMARCODataset, seg: GemSegment, n_eval: int, k: int = 100):
    """Compute brute-force MaxSim ground truth for n_eval queries."""
    log.info("Computing brute-force ground truth for %d queries (k=%d) ...", n_eval, k)
    t0 = time.time()
    gts = []
    for qi in range(n_eval):
        gt = seg.brute_force_maxsim(ds.query_vecs[qi], k)
        gts.append(gt)
        if (qi + 1) % 50 == 0:
            elapsed = time.time() - t0
            log.info("  GT: %d/%d queries (%.1fs, %.1f q/s)",
                     qi + 1, n_eval, elapsed, (qi + 1) / elapsed)
    log.info("Ground truth done in %.1fs", time.time() - t0)
    return gts


def recall_at_k(results, gt, k):
    gt_ids = {doc_id for doc_id, _ in gt[:k]}
    res_ids = {doc_id for doc_id, _ in results[:k]}
    if not gt_ids:
        return 1.0
    return len(gt_ids & res_ids) / len(gt_ids)


# ---------------------------------------------------------------------------
# Pipeline 1: CPU-only graph search
# ---------------------------------------------------------------------------

def cpu_recall_sweep(ds, seg, gts, n_eval):
    ef_values = [100, 400, 800, 2000, 5000, 10000]
    results = []
    for ef in ef_values:
        recalls_10, latencies = [], []
        for qi in range(n_eval):
            t0 = time.perf_counter()
            res = seg.search(ds.query_vecs[qi], k=100, ef=ef, n_probes=4)
            latencies.append((time.perf_counter() - t0) * 1e6)
            recalls_10.append(recall_at_k(res, gts[qi], 10))
        row = {
            "pipeline": "cpu_only", "ef": ef,
            "R@10": float(np.mean(recalls_10)),
            "p50_us": float(np.percentile(latencies, 50)),
            "p95_us": float(np.percentile(latencies, 95)),
        }
        results.append(row)
        log.info("CPU ef=%d  R@10=%.4f  p50=%.0fus", ef, row["R@10"], row["p50_us"])
    return results


# ---------------------------------------------------------------------------
# Pipeline 2: GPU qCH → FP16 MaxSim (quality ceiling)
# ---------------------------------------------------------------------------

def gpu_qch_fp16_pipeline(ds, seg, gts, n_eval):
    from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer
    from voyager_index._internal.kernels.maxsim import fast_colbert_scores

    device = torch.device("cuda")
    scorer = GpuQchScorer.from_gem_segment(seg, device="cuda")

    candidate_counts = [500, 1000, 2500, 5000, 7500]
    results = []

    for n_cand in candidate_counts:
        recalls_10, latencies = [], []
        for qi in range(n_eval):
            qv = ds.query_vecs[qi]
            qv_gpu = torch.from_numpy(qv).float().to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            proxy_scores = scorer.score_query(qv_gpu)
            _, top_idxs = proxy_scores.topk(n_cand, largest=False)
            cands = top_idxs.cpu().numpy()

            cand_vecs = [ds.doc_vecs[idx] for idx in cands]
            max_tok = max(v.shape[0] for v in cand_vecs)
            D_pad = np.zeros((len(cands), max_tok, ds.dim), dtype=np.float32)
            D_mask = np.zeros((len(cands), max_tok), dtype=np.float32)
            for i, v in enumerate(cand_vecs):
                D_pad[i, :v.shape[0]] = v
                D_mask[i, :v.shape[0]] = 1.0

            D_gpu = torch.from_numpy(D_pad).to(device)
            D_mask_gpu = torch.from_numpy(D_mask).to(device)
            q_t = torch.from_numpy(qv).float().unsqueeze(0).to(device)

            scores = fast_colbert_scores(q_t, D_gpu, documents_mask=D_mask_gpu).squeeze(0)
            top_k = scores.topk(min(10, n_cand), dim=-1)
            torch.cuda.synchronize()

            final = [(int(ds.doc_ids[cands[j]]), float(-scores[j].cpu()))
                      for j in top_k.indices.cpu().tolist()]
            latencies.append((time.perf_counter() - t0) * 1e6)
            recalls_10.append(recall_at_k(final, gts[qi], 10))

            del D_gpu, D_mask_gpu
            if qi % 50 == 0 and qi > 0:
                torch.cuda.empty_cache()

        row = {
            "pipeline": "gpu_qch_fp16_maxsim", "n_candidates": n_cand,
            "R@10": float(np.mean(recalls_10)),
            "p50_us": float(np.percentile(latencies, 50)),
            "p95_us": float(np.percentile(latencies, 95)),
        }
        results.append(row)
        log.info("GPU qCH→FP16 n=%d  R@10=%.4f  p50=%.0fus", n_cand, row["R@10"], row["p50_us"])

    return results


# ---------------------------------------------------------------------------
# Pipeline 3: GPU qCH → ROQ 4-bit MaxSim (production target)
# ---------------------------------------------------------------------------

def quantize_all_docs(ds: MSMARCODataset) -> dict:
    """Quantize all documents to ROQ 4-bit, return codes and meta arrays."""
    from voyager_index._internal.inference.quantization.rotational import (
        RotationalQuantizer, RoQConfig,
    )

    log.info("Quantizing %d documents to ROQ 4-bit ...", ds.n_docs)
    quantizer = RotationalQuantizer(RoQConfig(dim=ds.dim, num_bits=4, seed=42))

    all_codes, all_meta = [], []
    t0 = time.time()
    for i in range(ds.n_docs):
        vecs = ds.doc_vecs[i].astype(np.float32)
        q = quantizer.quantize(vecs, store=False)
        codes = np.asarray(q["codes"], dtype=np.uint8)
        meta = quantizer.build_triton_meta(q, include_norm_sq=True)
        all_codes.append(codes)
        all_meta.append(meta)
        if (i + 1) % 10000 == 0:
            log.info("  Quantized %d/%d docs (%.1fs)", i + 1, ds.n_docs, time.time() - t0)

    log.info("Quantization done in %.1fs", time.time() - t0)
    return {"codes": all_codes, "meta": all_meta, "quantizer": quantizer}


def gpu_qch_roq4_pipeline(ds, seg, gts, n_eval, roq_data):
    from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer
    from voyager_index._internal.kernels.roq import roq_maxsim_4bit

    device = torch.device("cuda")
    scorer = GpuQchScorer.from_gem_segment(seg, device="cuda")
    quantizer = roq_data["quantizer"]

    candidate_counts = [500, 1000, 2500, 5000, 7500]
    results = []

    for n_cand in candidate_counts:
        recalls_10, latencies = [], []
        for qi in range(n_eval):
            qv = ds.query_vecs[qi].astype(np.float32)
            qv_gpu = torch.from_numpy(qv).float().to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            proxy_scores = scorer.score_query(qv_gpu)
            _, top_idxs = proxy_scores.topk(n_cand, largest=False)
            cands = top_idxs.cpu().numpy()

            q_roq = quantizer.quantize(qv, store=False)
            q_codes = torch.from_numpy(np.asarray(q_roq["codes"], dtype=np.uint8)).unsqueeze(0).to(device)
            q_meta = torch.from_numpy(quantizer.build_triton_meta(q_roq, include_norm_sq=True)).unsqueeze(0).to(device)

            cand_codes = [roq_data["codes"][idx] for idx in cands]
            cand_meta = [roq_data["meta"][idx] for idx in cands]
            max_tok = max(c.shape[0] for c in cand_codes)
            nb = cand_codes[0].shape[1]

            d_codes_np = np.zeros((len(cands), max_tok, nb), dtype=np.uint8)
            d_meta_np = np.zeros((len(cands), max_tok, 4), dtype=np.float32)
            d_mask_np = np.zeros((len(cands), max_tok), dtype=np.float32)
            for i, (c, m) in enumerate(zip(cand_codes, cand_meta)):
                t = c.shape[0]
                d_codes_np[i, :t] = c
                d_meta_np[i, :t] = m
                d_mask_np[i, :t] = 1.0

            d_codes = torch.from_numpy(d_codes_np).to(device)
            d_meta = torch.from_numpy(d_meta_np).to(device)
            d_mask = torch.from_numpy(d_mask_np).to(device)

            scores = roq_maxsim_4bit(q_codes, q_meta, d_codes, d_meta,
                                      documents_mask=d_mask).squeeze(0)
            top_k = scores.topk(min(10, n_cand), dim=-1)
            torch.cuda.synchronize()

            final = [(int(ds.doc_ids[cands[j]]), float(-scores[j].cpu()))
                      for j in top_k.indices.cpu().tolist()]
            latencies.append((time.perf_counter() - t0) * 1e6)
            recalls_10.append(recall_at_k(final, gts[qi], 10))

            if qi % 50 == 0 and qi > 0:
                torch.cuda.empty_cache()

        row = {
            "pipeline": "gpu_qch_roq4_maxsim", "n_candidates": n_cand,
            "R@10": float(np.mean(recalls_10)),
            "p50_us": float(np.percentile(latencies, 50)),
            "p95_us": float(np.percentile(latencies, 95)),
        }
        results.append(row)
        log.info("GPU qCH→ROQ4 n=%d  R@10=%.4f  p50=%.0fus", n_cand, row["R@10"], row["p50_us"])

    return results


# ---------------------------------------------------------------------------
# Pipeline 4: CPU graph → ROQ 4-bit rerank (hybrid)
# ---------------------------------------------------------------------------

def cpu_graph_roq4_pipeline(ds, seg, gts, n_eval, roq_data):
    from voyager_index._internal.kernels.roq import roq_maxsim_4bit

    device = torch.device("cuda")
    quantizer = roq_data["quantizer"]

    configs = [(400, 100), (800, 200), (2000, 500), (5000, 1000), (10000, 2500)]
    results = []

    for ef, n_rerank in configs:
        recalls_10, latencies = [], []
        for qi in range(n_eval):
            qv = ds.query_vecs[qi].astype(np.float32)

            t0 = time.perf_counter()

            raw_results = seg.search(qv, k=n_rerank, ef=ef, n_probes=4)
            cand_doc_ids = [int(did) for did, _ in raw_results[:n_rerank]]

            q_roq = quantizer.quantize(qv, store=False)
            q_codes = torch.from_numpy(np.asarray(q_roq["codes"], dtype=np.uint8)).unsqueeze(0).to(device)
            q_meta = torch.from_numpy(quantizer.build_triton_meta(q_roq, include_norm_sq=True)).unsqueeze(0).to(device)

            cand_codes = [roq_data["codes"][did] for did in cand_doc_ids]
            cand_meta = [roq_data["meta"][did] for did in cand_doc_ids]
            max_tok = max(c.shape[0] for c in cand_codes)
            nb = cand_codes[0].shape[1]

            d_codes_np = np.zeros((len(cand_doc_ids), max_tok, nb), dtype=np.uint8)
            d_meta_np = np.zeros((len(cand_doc_ids), max_tok, 4), dtype=np.float32)
            d_mask_np = np.zeros((len(cand_doc_ids), max_tok), dtype=np.float32)
            for i, (c, m) in enumerate(zip(cand_codes, cand_meta)):
                t = c.shape[0]
                d_codes_np[i, :t] = c
                d_meta_np[i, :t] = m
                d_mask_np[i, :t] = 1.0

            d_codes = torch.from_numpy(d_codes_np).to(device)
            d_meta = torch.from_numpy(d_meta_np).to(device)
            d_mask = torch.from_numpy(d_mask_np).to(device)

            scores = roq_maxsim_4bit(q_codes, q_meta, d_codes, d_meta,
                                      documents_mask=d_mask).squeeze(0)
            top_k = scores.topk(min(10, len(cand_doc_ids)), dim=-1)
            torch.cuda.synchronize()

            final = [(cand_doc_ids[j], float(-scores[j].cpu()))
                      for j in top_k.indices.cpu().tolist()]
            latencies.append((time.perf_counter() - t0) * 1e6)
            recalls_10.append(recall_at_k(final, gts[qi], 10))

            if qi % 50 == 0 and qi > 0:
                torch.cuda.empty_cache()

        row = {
            "pipeline": "cpu_graph_roq4_rerank", "ef": ef, "n_rerank": n_rerank,
            "R@10": float(np.mean(recalls_10)),
            "p50_us": float(np.percentile(latencies, 50)),
            "p95_us": float(np.percentile(latencies, 95)),
        }
        results.append(row)
        log.info("CPU(ef=%d)→ROQ4(n=%d)  R@10=%.4f  p50=%.0fus",
                 ef, n_rerank, row["R@10"], row["p50_us"])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(all_results):
    print("\n" + "=" * 90)
    print(f"{'Pipeline':<30} {'Config':<20} {'R@10':>8} {'p50(ms)':>10} {'p95(ms)':>10}")
    print("-" * 90)
    for r in all_results.get("cpu_only", []):
        print(f"{'CPU-only':<30} {'ef=' + str(r['ef']):<20} {r['R@10']:>8.4f} {r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    for r in all_results.get("gpu_qch_fp16", []):
        print(f"{'GPU qCH→FP16 MaxSim':<30} {'n=' + str(r['n_candidates']):<20} {r['R@10']:>8.4f} {r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    for r in all_results.get("gpu_qch_roq4", []):
        print(f"{'GPU qCH→ROQ4 MaxSim':<30} {'n=' + str(r['n_candidates']):<20} {r['R@10']:>8.4f} {r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    for r in all_results.get("cpu_graph_roq4", []):
        cfg = f"ef={r['ef']},n={r['n_rerank']}"
        print(f"{'CPU→ROQ4 rerank':<30} {cfg:<20} {r['R@10']:>8.4f} {r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="100K GEM Index Eval")
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    log.info("Loading BeIR 100K dataset ...")
    ds = load_beir_100k_dataset()
    log.info("Dataset: %d docs, %d queries, %d total vectors, dim=%d",
             ds.n_docs, ds.n_queries, ds.all_vectors.shape[0], ds.dim)

    tok_counts = [ds.offsets[i][1] - ds.offsets[i][0] for i in range(ds.n_docs)]
    log.info("Tokens/doc: mean=%.0f, p50=%.0f, p95=%.0f, max=%d",
             np.mean(tok_counts), np.median(tok_counts),
             np.percentile(tok_counts, 95), max(tok_counts))

    # Phase A: Build or load
    if args.skip_build and INDEX_PATH.exists():
        seg = load_index()
    else:
        seg = build_index(ds)

    gc.collect()

    # Graph health
    health = graph_health(seg)

    # Phase B: Ground truth
    n_eval = min(args.n_eval, ds.n_queries)
    gts = compute_ground_truth(ds, seg, n_eval)

    # Phase C: ROQ 4-bit quantization
    roq_data = None
    if not args.skip_gpu:
        roq_data = quantize_all_docs(ds)

    # Phase D: Evaluation
    log.info("=== Pipeline 1: CPU-only recall sweep ===")
    cpu_results = cpu_recall_sweep(ds, seg, gts, n_eval)

    gpu_fp16_results, gpu_roq4_results, hybrid_results = [], [], []

    if not args.skip_gpu and torch.cuda.is_available():
        log.info("=== Pipeline 2: GPU qCH → FP16 MaxSim ===")
        gpu_fp16_results = gpu_qch_fp16_pipeline(ds, seg, gts, n_eval)

        log.info("=== Pipeline 3: GPU qCH → ROQ 4-bit MaxSim ===")
        gpu_roq4_results = gpu_qch_roq4_pipeline(ds, seg, gts, n_eval, roq_data)

        log.info("=== Pipeline 4: CPU graph → ROQ 4-bit rerank ===")
        hybrid_results = cpu_graph_roq4_pipeline(ds, seg, gts, n_eval, roq_data)

    all_results = {
        "dataset": {
            "n_docs": ds.n_docs, "n_queries": ds.n_queries, "n_eval": n_eval,
            "total_vectors": int(ds.all_vectors.shape[0]), "dim": ds.dim,
            "tokens_per_doc_mean": float(np.mean(tok_counts)),
            "tokens_per_doc_p95": float(np.percentile(tok_counts, 95)),
        },
        "build_params": BUILD_PARAMS,
        "graph_health": health,
        "cpu_only": cpu_results,
        "gpu_qch_fp16": gpu_fp16_results,
        "gpu_qch_roq4": gpu_roq4_results,
        "cpu_graph_roq4": hybrid_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Results saved to %s", RESULTS_PATH)

    print_summary(all_results)


if __name__ == "__main__":
    main()
