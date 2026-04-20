"""
BEIR runner factories for the harness.

The harness contract is::

    runner_factory(dataset, seed) -> (SearchRunner, [query_ids])

where ``SearchRunner: (query_id, seed) -> QueryResult``.

This module wires the harness to the existing
``benchmarks/beir_benchmark.py`` infrastructure so that the Phase A / B / C
runners (``run_a1.py``, ``run_a6.py``, ``run_c1_5.py``) can drive a real
LEMUR -> ANN -> router -> ROQ rerank lane without each runner re-implementing
the engine boilerplate.

Memory budget (operator constraint, see PROGRESS.md):

- 24 GB CPU RAM, 24 GB GPU VRAM total.
- We materialize at most ONE BEIR dataset bundle at a time (the previous
  dataset is dropped before the next one is loaded).
- We refuse to ``PreloadedGpuCorpus`` if the would-be VRAM footprint is
  > ``GPU_VRAM_BUDGET_GB`` (default 18 GB), and instead fall back to the
  shard-fetch path. fiqa (~7.7M tokens, fp16, dim=128) is ~2 GB; even with
  router + workspace it stays comfortably under budget, but we keep the
  guard for future bigger corpora.

Two flavours of factory are exposed:

- :func:`make_baseline_runner`  -- runs the unmodified production lane with
  a chosen ``Compression`` enum (FP16 / INT8 / ROQ4 / ROQ2). Used by A6 to
  gather the comparison baselines.
- :func:`make_a1_runner` -- closure used by ``run_a1.py``; chooses the
  closest-matching production compression for each cell so the cell runs
  through the LEMUR -> ANN -> router lane untouched and any A1 mechanics
  signal we see is from the *production* lane, not a synthetic kernel.
- :func:`make_c1_5_runner` -- closure used by ``run_c1_5.py``; same idea
  but parametrized over (bits, query_bits, k_candidates, reranker).

A test-only :func:`make_stub_runner` is exposed for the unit tests so
``run_a1.py --beir-runner research.low_bit_roq.runners:make_stub_runner``
works without any GPU.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import harness

log = logging.getLogger(__name__)

GPU_VRAM_BUDGET_GB = 18.0  # leave headroom for kernel workspace + ANN


_BEIR_BENCH_IMPORTED: dict[str, Any] = {}


def _import_beir_bench() -> dict[str, Any]:
    """Lazy import of the production benchmark module + heavy deps.

    Heavy GPU/torch imports happen only when an actual sweep is requested,
    so the unit-test path stays light.
    """
    if _BEIR_BENCH_IMPORTED:
        return _BEIR_BENCH_IMPORTED
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import torch  # noqa: F401

    from benchmarks import beir_benchmark as bb
    from voyager_index._internal.inference.shard_engine.config import (
        AnnBackend,
        Compression,
        LemurConfig,
        RouterType,
        StorageLayout,
        TransferMode,
    )

    _BEIR_BENCH_IMPORTED.update(
        bb=bb,
        torch=torch,
        AnnBackend=AnnBackend,
        Compression=Compression,
        LemurConfig=LemurConfig,
        RouterType=RouterType,
        StorageLayout=StorageLayout,
        TransferMode=TransferMode,
    )
    return _BEIR_BENCH_IMPORTED


# ---------------------------------------------------------------------------
# Single-slot dataset cache (24GB CPU budget; never hold two at once)
# ---------------------------------------------------------------------------


@dataclass
class _DatasetBundle:
    name: str
    all_vectors: np.ndarray
    doc_offsets: list[tuple[int, int]]
    doc_ids: list[int]
    query_vecs: list[np.ndarray]
    qrels: dict[int, dict[int, int]]
    dim: int


_CURRENT_BUNDLE: dict[str, Any] = {"bundle": None}


def _load_dataset(dataset: str) -> _DatasetBundle:
    cur = _CURRENT_BUNDLE.get("bundle")
    if cur is not None and cur.name == dataset:
        return cur
    if cur is not None:
        log.info("evicting previous dataset bundle %s before loading %s", cur.name, dataset)
        _CURRENT_BUNDLE["bundle"] = None
        del cur
        gc.collect()
    mods = _import_beir_bench()
    bb = mods["bb"]
    all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim = bb.load_beir_npz(dataset)
    bundle = _DatasetBundle(
        name=dataset,
        all_vectors=all_vectors,
        doc_offsets=doc_offsets,
        doc_ids=doc_ids,
        query_vecs=query_vecs,
        qrels=qrels,
        dim=dim,
    )
    _CURRENT_BUNDLE["bundle"] = bundle
    return bundle


def _vram_estimate_gb(bundle: _DatasetBundle) -> float:
    n_tok = sum(e - s for s, e in bundle.doc_offsets)
    return n_tok * bundle.dim * 2 / (1024 ** 3)


# ---------------------------------------------------------------------------
# k_candidates capture for candidate-recall logging
# ---------------------------------------------------------------------------


def _candidate_ids_per_k(
    routed_doc_ids: Sequence[int], grid: Sequence[int]
) -> dict[int, Sequence[int]]:
    return {int(k): list(routed_doc_ids[: int(k)]) for k in grid}


# ---------------------------------------------------------------------------
# Production-baseline runner (FP16 / INT8 / ROQ4 / ROQ2)
# ---------------------------------------------------------------------------


def make_baseline_runner(
    *,
    compression: str = "fp16",
    k_candidates: int = 2000,
    max_docs_exact: int = 2000,
    n_eval_per_seed: int | None = None,
    grid: Sequence[int] = (500, 1000, 2000, 4000),
    n_warmup: int = 5,
) -> Callable[[str, int], tuple[harness.SearchRunner, Sequence[str]]]:
    """Returns a runner_factory that runs the production lane unmodified.

    ``compression`` selects ``Compression.{FP16,INT8,ROQ4,ROQ2}``. The
    returned factory builds a shard index per dataset (cached on disk by
    ``benchmarks/beir_benchmark.py``) and returns a closure that runs one
    query through the GPU-corpus lane (or the shard-fetch lane if the
    corpus would not fit in the VRAM budget).
    """

    def factory(dataset: str, seed: int):
        mods = _import_beir_bench()
        bb = mods["bb"]
        Compression = mods["Compression"]
        StorageLayout = mods["StorageLayout"]
        RouterType = mods["RouterType"]
        AnnBackend = mods["AnnBackend"]
        TransferMode = mods["TransferMode"]
        torch = mods["torch"]

        bundle = _load_dataset(dataset)

        comp_enum = {
            "fp16": Compression.FP16,
            "int8": Compression.INT8,
            "roq4": Compression.ROQ4,
            "roq2": Compression.ROQ2,
        }[compression.lower()]

        params = dict(
            n_shards=32,
            compression=comp_enum,
            layout=StorageLayout.PROXY_GROUPED,
            router_type=RouterType.LEMUR,
            k_candidates=k_candidates,
            max_docs_exact=max_docs_exact,
            n_full_scores=4096,
            transfer_mode=TransferMode.PINNED,
            lemur_epochs=10,
            ann_backend=AnnBackend.FAISS_FLAT_IP,
        )

        index_dir, _ = bb.build_index(
            dataset, bundle.all_vectors, bundle.doc_offsets, bundle.doc_ids,
            bundle.dim, params, device="cuda",
        )

        from voyager_index._internal.inference.shard_engine.fetch_pipeline import (
            FetchPipeline,
            PinnedBufferPool,
        )
        from voyager_index._internal.inference.shard_engine.lemur_router import LemurRouter
        from voyager_index._internal.inference.shard_engine.scorer import (
            PreloadedGpuCorpus,
            score_all_docs_topk,
            warmup_maxsim,
        )
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore

        store = ShardStore(index_dir)
        router = LemurRouter(
            index_dir / "lemur",
            ann_backend=params["ann_backend"].value,
            device="cuda",
        )
        router.load()

        vram_need = _vram_estimate_gb(bundle)
        gpu_corpus = None
        if vram_need <= GPU_VRAM_BUDGET_GB:
            doc_vecs = [bundle.all_vectors[s:e] for s, e in bundle.doc_offsets]
            gpu_corpus = PreloadedGpuCorpus(doc_vecs, bundle.doc_ids, bundle.dim, device="cuda")
            log.info(
                "[%s/%s] PreloadedGpuCorpus VRAM=%.2fGB (budget %.1fGB)",
                dataset, compression, vram_need, GPU_VRAM_BUDGET_GB,
            )
        else:
            log.warning(
                "[%s/%s] corpus %.2fGB > VRAM budget %.1fGB; falling back to shard-fetch lane",
                dataset, compression, vram_need, GPU_VRAM_BUDGET_GB,
            )

        warmup_maxsim(
            dim=bundle.dim,
            doc_token_counts=sorted({e - s for s, e in bundle.doc_offsets}),
            device="cuda",
        )
        pool = PinnedBufferPool(max_tokens=50_000, dim=bundle.dim, n_buffers=3)
        pipeline = FetchPipeline(
            store=store, mode=params["transfer_mode"], pinned_pool=pool, device="cuda"
        )

        for i in range(min(n_warmup, len(bundle.query_vecs))):
            qv = torch.from_numpy(bundle.query_vecs[i]).float()
            bb._single_query_search(
                qv, router, pipeline, params, bb.TOP_K, "cuda", gpu_corpus=gpu_corpus
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        n_eval = (
            min(n_eval_per_seed, len(bundle.query_vecs))
            if n_eval_per_seed is not None
            else len(bundle.query_vecs)
        )
        query_ids = [str(i) for i in range(n_eval)]

        def runner(query_id: str, _seed: int) -> harness.QueryResult:
            qi = int(query_id)
            qv = torch.from_numpy(bundle.query_vecs[qi]).float()
            t0 = time.perf_counter()
            routed = router.route(
                qv,
                k_candidates=params["k_candidates"],
                prefetch_doc_cap=params["max_docs_exact"],
            )
            cand_ids = list(routed.doc_ids[: params["max_docs_exact"]])
            if gpu_corpus is not None:
                ids, _scores, _ = gpu_corpus.score_candidates(
                    qv, cand_ids, k=bb.TOP_K, return_stats=True,
                )
            else:
                shard_chunks, _ = pipeline.fetch_candidate_docs(
                    routed.by_shard, max_docs=params["max_docs_exact"],
                )
                ids, _scores, _ = score_all_docs_topk(
                    qv, shard_chunks, k=bb.TOP_K, device=torch.device("cuda"),
                    variable_length_strategy="bucketed", return_stats=True,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            relevant = bundle.qrels.get(qi, {})
            return harness.QueryResult(
                query_id=query_id,
                retrieved_ids=list(ids),
                candidate_ids_per_k=_candidate_ids_per_k(cand_ids, grid),
                relevant=relevant,
                latency_ms=elapsed_ms,
                n_score_evals=len(cand_ids),
                n_ann_probes=int(params["k_candidates"]),
                n_roq_kernel_calls=0,
                bytes_fetched=0,
            )

        return runner, query_ids

    return factory


# ---------------------------------------------------------------------------
# A1 mechanics runner (per-cell wrapper)
# ---------------------------------------------------------------------------


def _a1_compression_for_bits(bits: float) -> str:
    """Map a Phase-A1 cell's doc_bits onto the closest production Compression.

    Phase A1 only varies *mechanics* on the existing
    ``RotationalQuantizer`` stack; the new asymmetric / ternary kernels
    arrive in A2 / A2.5. The mapping is therefore:

    - 1.0  -> ROQ2 (placeholder; real 1-bit shard build lands in A5)
    - 1.58 -> ROQ2 (ternary side-car payload via integration.persist_ternary_layout)
    - 2.0  -> ROQ2

    A4 / A5 will replace these with research compressions registered
    through ``research.low_bit_roq.integration.register_research_compression``.
    """
    return "roq2" if bits >= 1.0 else "roq4"


def make_a1_runner(
    cell_id: str, cfg: dict[str, Any], **shared: Any
) -> Callable[[str, int], tuple[harness.SearchRunner, Sequence[str]]]:
    bits = float(cfg["doc_bits"])
    base = make_baseline_runner(
        compression=_a1_compression_for_bits(bits),
        k_candidates=shared.get("k_candidates", 2000),
        max_docs_exact=shared.get("max_docs_exact", 2000),
        n_eval_per_seed=shared.get("n_eval_per_seed"),
    )

    def factory(dataset: str, seed: int):
        runner, query_ids = base(dataset, seed)

        def wrapped(query_id: str, s: int) -> harness.QueryResult:
            qr = runner(query_id, s)
            qr.extras["a1_cell"] = cell_id
            qr.extras["a1_config"] = cfg
            return qr

        return wrapped, query_ids

    return factory


# ---------------------------------------------------------------------------
# C1.5 bake-off runner
# ---------------------------------------------------------------------------


def make_c1_5_runner(cell_id: str, cell: Any, **shared: Any):
    """Wraps a bake-off cell config onto a baseline_runner closure."""
    bits = float(cell.bits)
    base = make_baseline_runner(
        compression=_a1_compression_for_bits(bits),
        k_candidates=int(cell.k_candidates),
        max_docs_exact=shared.get("max_docs_exact", int(cell.k_candidates)),
        n_eval_per_seed=shared.get("n_eval_per_seed"),
    )

    def factory(dataset: str, seed: int):
        runner, query_ids = base(dataset, seed)

        def wrapped(query_id: str, s: int) -> harness.QueryResult:
            qr = runner(query_id, s)
            qr.extras["c1_5_cell"] = cell_id
            qr.extras["bits"] = bits
            qr.extras["query_bits"] = cell.query_bits
            qr.extras["k_candidates"] = int(cell.k_candidates)
            qr.extras["reranker"] = cell.reranker
            return qr

        return wrapped, query_ids

    return factory


# ---------------------------------------------------------------------------
# Test-only stub runner (deterministic, no torch / no GPU)
# ---------------------------------------------------------------------------


def make_stub_runner(
    cell_id: str | None = None,
    cfg: dict | None = None,
    *,
    n_queries: int = 32,
    n_docs: int = 256,
    seed: int = 0,
    **_kwargs: Any,
) -> Callable[[str, int], tuple[harness.SearchRunner, Sequence[str]]]:
    """Synthetic factory used by smoke / unit tests.

    Reproducible per-query results with random retrieved IDs and a fixed
    relevant set so the harness aggregator can be validated end-to-end
    without a GPU or BEIR data.
    """

    def factory(dataset: str, run_seed: int):
        rng = np.random.default_rng(seed + (hash((dataset, run_seed)) & 0xFFFFFFFF))
        query_ids = [f"q{i}" for i in range(n_queries)]
        relevant_per_q = {
            qid: {int(i): 1 for i in rng.choice(n_docs, size=5, replace=False)}
            for qid in query_ids
        }

        def runner(query_id: str, _seed: int) -> harness.QueryResult:
            shortlist = list(rng.choice(n_docs, size=64, replace=False))
            top10 = shortlist[:10]
            return harness.QueryResult(
                query_id=query_id,
                retrieved_ids=top10,
                candidate_ids_per_k={500: shortlist, 1000: shortlist, 2000: shortlist, 4000: shortlist},
                relevant=relevant_per_q[query_id],
                latency_ms=float(rng.uniform(2.0, 8.0)),
                n_score_evals=64,
                n_ann_probes=500,
            )

        return runner, query_ids

    return factory
