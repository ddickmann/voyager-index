"""
100K GEM Index — Build, Quantize, and Evaluate.

Designed for a **23 GB RAM container** with GPU (24 GB VRAM).

Architecture:
  Phase A: Build index — loads float32 vectors (~7 GB), Rust copies internally
           (~7 GB temp), saves index, frees vectors.  Peak ~15 GB.
  Phase B: Evaluate — loads index (~0.3 GB), doc vectors as float16 (~3.4 GB),
           ROQ 4-bit codes (~1 GB), runs 4 pipelines.  Peak ~6 GB.

Pipelines:
  1. CPU-only graph search (proxy scores)
  2. GPU qCH brute-force → FP16 MaxSim rerank (quality ceiling)
  3. GPU qCH brute-force → ROQ 4-bit MaxSim rerank (production target)
  4. CPU graph search → ROQ 4-bit MaxSim rerank (hybrid)

Usage:
  python benchmarks/eval_100k.py [--n-eval 300] [--skip-build] [--skip-gpu]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "voyager-qa"
NPZ_PATH = CACHE_DIR / "beir_100k.npz"
INDEX_PATH = CACHE_DIR / "beir_100k_index.bin"
GT_PATH = CACHE_DIR / "beir_100k_gt.npz"
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
    store_raw_vectors=False,
)


def _mem_gb():
    """Current process RSS in GB."""
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1


def _cgroup_limit_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = open(p).read().strip()
            if v != "max":
                return int(v) / 1e9
        except Exception:
            continue
    return -1


# ======================================================================
# Phase A: Build index  (peak ~15 GB)
# ======================================================================

def build_index():
    """Load float32 vectors, build GEM index, save to disk, free everything."""
    from latence_gem_index import GemSegment

    limit = _cgroup_limit_gb()
    log.info("=== PHASE A: BUILD INDEX ===")
    log.info("Container memory limit: %.1f GB, current RSS: %.1f GB", limit, _mem_gb())

    # Load NPZ — float16 on disk, convert to float32 carefully
    log.info("Loading %s ...", NPZ_PATH)
    npz = np.load(str(NPZ_PATH), allow_pickle=True)
    doc_offsets_arr = npz["doc_offsets"]
    n_docs = int(npz["n_docs"])
    dim = int(npz["dim"])

    # Load float16, immediately convert to float32, delete float16
    log.info("Converting doc_vectors float16 → float32 ...")
    dv_f16 = npz["doc_vectors"]  # (13.3M, 128) float16 = 3.4 GB
    log.info("  float16 loaded: shape=%s, RSS=%.1f GB", dv_f16.shape, _mem_gb())
    all_vectors = dv_f16.astype(np.float32)  # 6.8 GB
    del dv_f16
    npz_files = npz.files  # cache file list before closing
    npz.close()
    gc.collect()
    log.info("  float32 ready: shape=%s, RSS=%.1f GB", all_vectors.shape, _mem_gb())

    offsets = [(int(s), int(e)) for s, e in doc_offsets_arr]
    doc_ids = list(range(n_docs))
    total_tokens = all_vectors.shape[0]

    log.info("Dataset: %d docs, %d total tokens, dim=%d", n_docs, total_tokens, dim)
    tok_counts = [e - s for s, e in offsets]
    log.info("Tokens/doc: mean=%.0f, p50=%.0f, p95=%.0f, max=%d",
             np.mean(tok_counts), np.median(tok_counts),
             np.percentile(tok_counts, 95), max(tok_counts))

    # Build
    seg = GemSegment()
    log.info("Building with params: %s", BUILD_PARAMS)
    log.info("Pre-build RSS: %.1f GB", _mem_gb())

    t0 = time.time()
    seg.build(all_vectors, doc_ids, offsets, **BUILD_PARAMS)
    build_s = time.time() - t0
    log.info("Build done in %.1fs (%.1f min), RSS: %.1f GB", build_s, build_s / 60, _mem_gb())

    # Free the big float32 array — Rust dropped its copy already (store_raw_vectors=False)
    del all_vectors, offsets, doc_ids
    gc.collect()
    log.info("After freeing vectors, RSS: %.1f GB", _mem_gb())

    # Graph health
    n_comp, gf, cr = seg.graph_connectivity_report()
    n_nodes, n_edges = seg.n_nodes(), seg.n_edges()
    log.info("Graph: comp=%d, giant=%.4f, cross_ratio=%.4f, nodes=%d, edges=%d, mean_deg=%.1f",
             n_comp, gf, cr, n_nodes, n_edges, 2 * n_edges / max(n_nodes, 1))

    # Save
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    seg.save(str(INDEX_PATH))
    size_mb = INDEX_PATH.stat().st_size / 1024**2
    log.info("Index saved: %s (%.1f MB)", INDEX_PATH, size_mb)

    health = {
        "n_docs": seg.n_docs(), "n_nodes": n_nodes, "n_edges": n_edges,
        "mean_degree": 2 * n_edges / max(n_nodes, 1),
        "n_components": n_comp, "giant_component_frac": gf,
        "cross_cluster_edge_ratio": cr, "build_time_s": build_s,
        "index_size_mb": size_mb,
    }
    return health


# ======================================================================
# Phase B: Evaluate  (peak ~6 GB)
# ======================================================================

def load_dataset_lightweight():
    """Load dataset keeping vectors in float16 (half the RAM of float32).

    For reranking we convert small candidate batches to float32 on the fly.
    """
    log.info("Loading dataset in lightweight mode (float16) ...")
    npz = np.load(str(NPZ_PATH), allow_pickle=True)
    doc_offsets = npz["doc_offsets"]
    query_offsets = npz["query_offsets"]
    n_docs = int(npz["n_docs"])
    dim = int(npz["dim"])
    qrels_mat = npz["qrels"]

    # Keep doc vectors as float16 — 3.4 GB vs 6.8 GB
    all_doc_f16 = npz["doc_vectors"]  # (total_tokens, dim) float16
    doc_vecs = [all_doc_f16[int(s):int(e)] for s, e in doc_offsets]

    all_q_f16 = npz["query_vectors"]
    query_vecs = [all_q_f16[int(s):int(e)].astype(np.float32) for s, e in query_offsets]

    qrels = {}
    for qi in range(qrels_mat.shape[0]):
        rels = [int(x) for x in qrels_mat[qi] if x >= 0]
        if rels:
            qrels[qi] = rels

    doc_ids = list(range(n_docs))
    offsets = [(int(s), int(e)) for s, e in doc_offsets]

    log.info("Loaded: %d docs, %d queries, dim=%d, RSS=%.1f GB",
             n_docs, len(query_vecs), dim, _mem_gb())

    return {
        "doc_vecs": doc_vecs, "query_vecs": query_vecs, "doc_ids": doc_ids,
        "offsets": offsets, "qrels": qrels, "n_docs": n_docs, "dim": dim,
        "n_queries": len(query_vecs),
    }


def recall_at_k(results, gt, k):
    gt_ids = {doc_id for doc_id, _ in gt[:k]}
    res_ids = {doc_id for doc_id, _ in results[:k]}
    if not gt_ids:
        return 1.0
    return len(gt_ids & res_ids) / len(gt_ids)


# -- Ground truth via GPU batched MaxSim ----------------------------------

def compute_ground_truth(ds, n_eval, k=100):
    """Brute-force MaxSim via Triton kernel, processing docs in 2K batches to limit VRAM."""
    import torch
    from voyager_index._internal.kernels.maxsim import fast_colbert_scores

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH = 2000

    log.info("Computing GPU brute-force GT for %d queries (k=%d, batch=%d docs) ...",
             n_eval, k, BATCH)
    t0 = time.time()
    gts = []

    for qi in range(n_eval):
        qv = ds["query_vecs"][qi]
        q_t = torch.from_numpy(qv).float().unsqueeze(0).to(device)

        all_scores = []
        for start in range(0, ds["n_docs"], BATCH):
            end = min(start + BATCH, ds["n_docs"])
            batch = ds["doc_vecs"][start:end]
            max_tok = max(v.shape[0] for v in batch)
            D = np.zeros((end - start, max_tok, ds["dim"]), dtype=np.float32)
            M = np.zeros((end - start, max_tok), dtype=np.float32)
            for i, v in enumerate(batch):
                fv = v.astype(np.float32)
                D[i, :fv.shape[0]] = fv
                M[i, :fv.shape[0]] = 1.0

            scores = fast_colbert_scores(
                q_t,
                torch.from_numpy(D).to(device),
                documents_mask=torch.from_numpy(M).to(device),
            ).squeeze(0)
            all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores)
        topk = all_scores.topk(min(k, ds["n_docs"]))
        gt = [(int(ds["doc_ids"][j]), float(all_scores[j])) for j in topk.indices.tolist()]
        gts.append(gt)

        if (qi + 1) % 10 == 0:
            elapsed = time.time() - t0
            log.info("  GT %d/%d (%.1fs, %.1f q/s)", qi + 1, n_eval, elapsed, (qi + 1) / elapsed)
            torch.cuda.empty_cache()

    log.info("GT done in %.1fs", time.time() - t0)
    return gts


# -- Pipeline 1: CPU-only -------------------------------------------------

def cpu_recall_sweep(ds, seg, gts, n_eval):
    ef_values = [100, 400, 800, 2000, 5000, 10000]
    results = []
    for ef in ef_values:
        recalls_10, latencies = [], []
        for qi in range(n_eval):
            qv = ds["query_vecs"][qi]
            t0 = time.perf_counter()
            res = seg.search(qv, k=100, ef=ef, n_probes=4)
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


# -- Pipeline 2: GPU qCH → FP16 MaxSim ------------------------------------

def gpu_qch_fp16_pipeline(ds, seg, gts, n_eval):
    import torch
    from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer
    from voyager_index._internal.kernels.maxsim import fast_colbert_scores

    device = torch.device("cuda")
    scorer = GpuQchScorer.from_gem_segment(seg, device="cuda")

    candidate_counts = [500, 1000, 2500, 5000, 7500, 10000]
    results = []

    for n_cand in candidate_counts:
        recalls_10, latencies = [], []
        for qi in range(n_eval):
            qv = ds["query_vecs"][qi]
            qv_gpu = torch.from_numpy(qv).float().to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            proxy = scorer.score_query(qv_gpu)
            _, top_idxs = proxy.topk(n_cand, largest=False)
            cands = top_idxs.cpu().numpy()

            batch = [ds["doc_vecs"][idx] for idx in cands]
            max_tok = max(v.shape[0] for v in batch)
            D = np.zeros((len(cands), max_tok, ds["dim"]), dtype=np.float32)
            M = np.zeros((len(cands), max_tok), dtype=np.float32)
            for i, v in enumerate(batch):
                fv = v.astype(np.float32)
                D[i, :fv.shape[0]] = fv
                M[i, :fv.shape[0]] = 1.0

            q_t = torch.from_numpy(qv).float().unsqueeze(0).to(device)
            scores = fast_colbert_scores(
                q_t,
                torch.from_numpy(D).to(device),
                documents_mask=torch.from_numpy(M).to(device),
            ).squeeze(0)
            top_k = scores.topk(min(10, n_cand))
            torch.cuda.synchronize()

            final = [(int(ds["doc_ids"][cands[j]]), float(scores[j]))
                     for j in top_k.indices.cpu().tolist()]
            latencies.append((time.perf_counter() - t0) * 1e6)
            recalls_10.append(recall_at_k(final, gts[qi], 10))

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


# -- ROQ 4-bit quantization -----------------------------------------------

def quantize_all_docs(ds):
    """Quantize every document to ROQ 4-bit. ~1 GB total."""
    from voyager_index._internal.inference.quantization.rotational import (
        RotationalQuantizer, RoQConfig,
    )

    log.info("Quantizing %d documents to ROQ 4-bit ...", ds["n_docs"])
    quantizer = RotationalQuantizer(RoQConfig(dim=ds["dim"], num_bits=4, seed=42))

    all_codes, all_meta = [], []
    t0 = time.time()
    for i in range(ds["n_docs"]):
        vecs = ds["doc_vecs"][i].astype(np.float32)
        q = quantizer.quantize(vecs, store=False)
        all_codes.append(np.asarray(q["codes"], dtype=np.uint8))
        all_meta.append(quantizer.build_triton_meta(q, include_norm_sq=True))
        if (i + 1) % 20000 == 0:
            log.info("  Quantized %d/%d (%.1fs, RSS=%.1f GB)",
                     i + 1, ds["n_docs"], time.time() - t0, _mem_gb())

    log.info("Quantization done in %.1fs, RSS=%.1f GB", time.time() - t0, _mem_gb())
    return {"codes": all_codes, "meta": all_meta, "quantizer": quantizer}


# -- Pipeline 3: GPU qCH → ROQ 4-bit MaxSim --------------------------------

def gpu_qch_roq4_pipeline(ds, seg, gts, n_eval, roq_data):
    import torch
    from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer
    from voyager_index._internal.kernels.roq import roq_maxsim_4bit

    device = torch.device("cuda")
    scorer = GpuQchScorer.from_gem_segment(seg, device="cuda")
    quantizer = roq_data["quantizer"]

    candidate_counts = [500, 1000, 2500, 5000, 7500, 10000]
    results = []

    for n_cand in candidate_counts:
        recalls_10, latencies = [], []
        for qi in range(n_eval):
            qv = ds["query_vecs"][qi]
            qv_gpu = torch.from_numpy(qv).float().to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            proxy = scorer.score_query(qv_gpu)
            _, top_idxs = proxy.topk(n_cand, largest=False)
            cands = top_idxs.cpu().numpy()

            q_roq = quantizer.quantize(qv, store=False)
            q_codes = torch.from_numpy(
                np.asarray(q_roq["codes"], dtype=np.uint8)
            ).unsqueeze(0).to(device)
            q_meta = torch.from_numpy(
                quantizer.build_triton_meta(q_roq, include_norm_sq=True)
            ).unsqueeze(0).to(device)

            cand_codes = [roq_data["codes"][idx] for idx in cands]
            cand_meta = [roq_data["meta"][idx] for idx in cands]
            max_tok = max(c.shape[0] for c in cand_codes)
            nb = cand_codes[0].shape[1]

            dc = np.zeros((len(cands), max_tok, nb), dtype=np.uint8)
            dm = np.zeros((len(cands), max_tok, 4), dtype=np.float32)
            dmask = np.zeros((len(cands), max_tok), dtype=np.float32)
            for i, (c, m) in enumerate(zip(cand_codes, cand_meta)):
                t = c.shape[0]
                dc[i, :t] = c; dm[i, :t] = m; dmask[i, :t] = 1.0

            scores = roq_maxsim_4bit(
                q_codes, q_meta,
                torch.from_numpy(dc).to(device),
                torch.from_numpy(dm).to(device),
                documents_mask=torch.from_numpy(dmask).to(device),
            ).squeeze(0)
            top_k = scores.topk(min(10, n_cand))
            torch.cuda.synchronize()

            final = [(int(ds["doc_ids"][cands[j]]), float(scores[j]))
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


# -- Pipeline 4: CPU graph → ROQ 4-bit rerank ------------------------------

def cpu_graph_roq4_pipeline(ds, seg, gts, n_eval, roq_data):
    import torch
    from voyager_index._internal.kernels.roq import roq_maxsim_4bit

    device = torch.device("cuda")
    quantizer = roq_data["quantizer"]

    configs = [(400, 100), (800, 200), (2000, 500), (5000, 1000), (10000, 2500)]
    results = []

    for ef, n_rerank in configs:
        recalls_10, latencies = [], []
        for qi in range(n_eval):
            qv = ds["query_vecs"][qi]

            t0 = time.perf_counter()
            raw = seg.search(qv, k=n_rerank, ef=ef, n_probes=4)
            cand_ids = [int(did) for did, _ in raw[:n_rerank]]

            q_roq = quantizer.quantize(qv, store=False)
            q_codes = torch.from_numpy(
                np.asarray(q_roq["codes"], dtype=np.uint8)
            ).unsqueeze(0).to(device)
            q_meta = torch.from_numpy(
                quantizer.build_triton_meta(q_roq, include_norm_sq=True)
            ).unsqueeze(0).to(device)

            cand_codes = [roq_data["codes"][did] for did in cand_ids]
            cand_meta = [roq_data["meta"][did] for did in cand_ids]
            max_tok = max(c.shape[0] for c in cand_codes)
            nb = cand_codes[0].shape[1]

            dc = np.zeros((len(cand_ids), max_tok, nb), dtype=np.uint8)
            dm = np.zeros((len(cand_ids), max_tok, 4), dtype=np.float32)
            dmask = np.zeros((len(cand_ids), max_tok), dtype=np.float32)
            for i, (c, m) in enumerate(zip(cand_codes, cand_meta)):
                t = c.shape[0]
                dc[i, :t] = c; dm[i, :t] = m; dmask[i, :t] = 1.0

            scores = roq_maxsim_4bit(
                q_codes, q_meta,
                torch.from_numpy(dc).to(device),
                torch.from_numpy(dm).to(device),
                documents_mask=torch.from_numpy(dmask).to(device),
            ).squeeze(0)
            top_k = scores.topk(min(10, len(cand_ids)))
            torch.cuda.synchronize()

            final = [(cand_ids[j], float(scores[j]))
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


# ======================================================================
# Main
# ======================================================================

def print_summary(all_results):
    print("\n" + "=" * 90)
    print(f"{'Pipeline':<30} {'Config':<20} {'R@10':>8} {'p50(ms)':>10} {'p95(ms)':>10}")
    print("-" * 90)
    for r in all_results.get("cpu_only", []):
        print(f"{'CPU-only':<30} {'ef=' + str(r['ef']):<20} {r['R@10']:>8.4f} "
              f"{r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    for r in all_results.get("gpu_qch_fp16", []):
        print(f"{'GPU qCH→FP16 MaxSim':<30} {'n=' + str(r['n_candidates']):<20} {r['R@10']:>8.4f} "
              f"{r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    for r in all_results.get("gpu_qch_roq4", []):
        print(f"{'GPU qCH→ROQ4 MaxSim':<30} {'n=' + str(r['n_candidates']):<20} {r['R@10']:>8.4f} "
              f"{r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    for r in all_results.get("cpu_graph_roq4", []):
        cfg = f"ef={r['ef']},n={r['n_rerank']}"
        print(f"{'CPU→ROQ4 rerank':<30} {cfg:<20} {r['R@10']:>8.4f} "
              f"{r['p50_us']/1000:>10.1f} {r['p95_us']/1000:>10.1f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="100K GEM Index Eval")
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-gt", action="store_true",
                        help="Skip GT computation (reuse cached)")
    args = parser.parse_args()

    log.info("Container limit: %.1f GB, current RSS: %.1f GB",
             _cgroup_limit_gb(), _mem_gb())

    # ---- Phase A: Build ----
    if args.skip_build and INDEX_PATH.exists():
        log.info("Skipping build — loading existing index from %s", INDEX_PATH)
        health = {}
    else:
        health = build_index()
        gc.collect()
        log.info("Post-build RSS: %.1f GB", _mem_gb())

    # ---- Phase B: Evaluate ----
    log.info("=== PHASE B: EVALUATE ===")

    from latence_gem_index import GemSegment
    seg = GemSegment()
    seg.load(str(INDEX_PATH))

    # Graph health (if we skipped build)
    if not health:
        n_comp, gf, cr = seg.graph_connectivity_report()
        health = {
            "n_docs": seg.n_docs(), "n_nodes": seg.n_nodes(), "n_edges": seg.n_edges(),
            "mean_degree": 2 * seg.n_edges() / max(seg.n_nodes(), 1),
            "n_components": n_comp, "giant_component_frac": gf,
            "cross_cluster_edge_ratio": cr,
        }
    log.info("Graph: %s", {k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in health.items()})

    # Load dataset lightweight (float16 doc vecs = 3.4 GB)
    ds = load_dataset_lightweight()
    n_eval = min(args.n_eval, ds["n_queries"])

    log.info("RSS after loading dataset: %.1f GB", _mem_gb())

    # Ground truth
    gts = None
    if args.skip_gt and GT_PATH.exists():
        log.info("Loading cached GT from %s", GT_PATH)
        gt_data = np.load(str(GT_PATH), allow_pickle=True)
        gts = [[(int(did), float(sc)) for did, sc in zip(ids, scores)]
               for ids, scores in zip(gt_data["gt_ids"], gt_data["gt_scores"])]
        gts = gts[:n_eval]
    if gts is None or len(gts) < n_eval:
        gts = compute_ground_truth(ds, n_eval)
        gt_ids = np.array([[did for did, _ in gt] for gt in gts], dtype=np.int64)
        gt_scores = np.array([[sc for _, sc in gt] for gt in gts], dtype=np.float32)
        GT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(GT_PATH), gt_ids=gt_ids, gt_scores=gt_scores)
        log.info("GT cached to %s", GT_PATH)

    # Pipeline 1: CPU-only
    log.info("=== Pipeline 1: CPU-only ===")
    cpu_results = cpu_recall_sweep(ds, seg, gts, n_eval)

    gpu_fp16_results, gpu_roq4_results, hybrid_results = [], [], []

    if not args.skip_gpu:
        import torch
        if torch.cuda.is_available():
            # ROQ quantization (~1 GB)
            roq_data = quantize_all_docs(ds)
            log.info("RSS after ROQ quantization: %.1f GB", _mem_gb())

            log.info("=== Pipeline 2: GPU qCH → FP16 MaxSim ===")
            gpu_fp16_results = gpu_qch_fp16_pipeline(ds, seg, gts, n_eval)

            log.info("=== Pipeline 3: GPU qCH → ROQ 4-bit MaxSim ===")
            gpu_roq4_results = gpu_qch_roq4_pipeline(ds, seg, gts, n_eval, roq_data)

            log.info("=== Pipeline 4: CPU graph → ROQ 4-bit rerank ===")
            hybrid_results = cpu_graph_roq4_pipeline(ds, seg, gts, n_eval, roq_data)

    tok_counts = [e - s for s, e in ds["offsets"]]
    all_results = {
        "dataset": {
            "n_docs": ds["n_docs"], "n_queries": ds["n_queries"], "n_eval": n_eval,
            "total_vectors": sum(v.shape[0] for v in ds["doc_vecs"]), "dim": ds["dim"],
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
    log.info("Final RSS: %.1f GB", _mem_gb())


if __name__ == "__main__":
    main()
