"""
GPU-native qCH proxy scorer using PyTorch GEMM + batched max-gather.

Sits alongside (not replacing) the Rust CPU path. Selected by device flag.
Falls back to pure-PyTorch gather when Triton is not available.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from .triton_qch_kernel import (
        TRITON_AVAILABLE,
        qch_max_gather_gpu,
        qch_max_gather_torch,
        qch_pairwise_batch_gpu,
        qch_pairwise_batch_torch,
    )
except ImportError:
    TRITON_AVAILABLE = False
    qch_max_gather_gpu = None
    qch_max_gather_torch = None
    qch_pairwise_batch_gpu = None
    qch_pairwise_batch_torch = None


class GpuQchScorer:
    """
    GPU-accelerated qCH proxy scorer.

    Uploads codebook centroids, IDF weights, and flat doc codes to the GPU.
    Scores queries via GEMM (query @ centroids^T) + IDF multiply + batched
    max-gather over packed doc codes.
    """

    def __init__(
        self,
        codebook_centroids: np.ndarray,
        idf: np.ndarray,
        flat_codes: np.ndarray,
        flat_offsets: np.ndarray,
        flat_lengths: np.ndarray,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.n_fine = codebook_centroids.shape[0]
        self.dim = codebook_centroids.shape[1]
        self.n_docs = flat_offsets.shape[0]

        self.centroids = torch.from_numpy(
            codebook_centroids.astype(np.float32)
        ).to(self.device)  # (n_fine, dim)

        self.idf = torch.from_numpy(
            idf.astype(np.float32)
        ).to(self.device)  # (n_fine,)

        self.codes = torch.from_numpy(
            flat_codes.astype(np.int32)
        ).to(self.device)  # (total_codes,)

        self.offsets = torch.from_numpy(
            flat_offsets.astype(np.int32)
        ).to(self.device)  # (n_docs,)

        self.lengths = torch.from_numpy(
            flat_lengths.astype(np.int32)
        ).to(self.device)  # (n_docs,)

        self._use_triton = TRITON_AVAILABLE and self.device.type == "cuda"
        logger.debug(
            "GpuQchScorer: %d centroids x %d dim, %d docs, triton=%s, device=%s",
            self.n_fine, self.dim, self.n_docs, self._use_triton, self.device,
        )

    def _compute_query_scores(self, query_vecs: torch.Tensor) -> torch.Tensor:
        """Compute IDF-weighted query-centroid similarity scores via GEMM."""
        # query_vecs: (n_query, dim), centroids: (n_fine, dim)
        # scores: (n_query, n_fine) = query_vecs @ centroids^T
        scores = torch.mm(query_vecs, self.centroids.t())

        # Apply IDF: element-wise multiply each row by idf
        scores = scores * self.idf.unsqueeze(0)

        return scores  # (n_query, n_fine)

    @torch.no_grad()
    def score_query(self, query_vecs: torch.Tensor) -> torch.Tensor:
        """
        Score all documents against a multi-vector query.

        Args:
            query_vecs: (n_query, dim) float32 tensor on same device

        Returns:
            (n_docs,) float32 tensor of qCH proxy scores (lower = closer)
        """
        query_vecs = query_vecs.to(self.device)
        n_query = query_vecs.shape[0]
        scores = self._compute_query_scores(query_vecs)  # (n_query, n_fine)
        flat_scores = scores.reshape(-1)  # (n_query * n_fine,)

        if self._use_triton and qch_max_gather_gpu is not None:
            try:
                return qch_max_gather_gpu(
                    flat_scores, self.codes, self.offsets, self.lengths,
                    n_query, self.n_fine,
                )
            except Exception as exc:
                logger.warning("Triton kernel failed (%s), falling back to PyTorch", exc)

        if qch_max_gather_torch is not None:
            return qch_max_gather_torch(
                flat_scores, self.codes, self.offsets, self.lengths,
                n_query, self.n_fine,
            )

        raise RuntimeError("No GPU scoring backend available")

    @torch.no_grad()
    def score_query_filtered(
        self,
        query_vecs: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score only masked documents.

        Args:
            query_vecs: (n_query, dim)
            doc_mask: (n_docs,) bool — True = score this doc

        Returns:
            (n_docs,) float32 — unmasked docs get score inf
        """
        query_vecs = query_vecs.to(self.device)
        doc_mask = doc_mask.to(self.device)

        all_scores = self.score_query(query_vecs)
        all_scores = all_scores.masked_fill(~doc_mask, float("inf"))
        return all_scores

    @classmethod
    def from_gem_segment(
        cls,
        segment,
        device: str = "cuda",
    ) -> "GpuQchScorer":
        """
        Construct a GpuQchScorer from a built GemSegment's exported data.

        Args:
            segment: a latence_gem_index.GemSegment with get_codebook_centroids(),
                     get_idf(), get_flat_codes() methods.
            device: torch device string
        """
        centroids = np.array(segment.get_codebook_centroids())
        idf = np.array(segment.get_idf())
        codes, offsets, lengths = segment.get_flat_codes()
        codes = np.array(codes)
        offsets = np.array(offsets)
        lengths = np.array(lengths)

        return cls(
            codebook_centroids=centroids,
            idf=idf,
            flat_codes=codes,
            flat_offsets=offsets,
            flat_lengths=lengths,
            device=device,
        )


class GpuBuildDistanceComputer:
    """
    GPU batch distance computer for NN-Descent graph construction.

    Uploads the centroid pairwise L2 distance table and flat doc codes to GPU.
    Provides a callable that accepts (pairs_a, pairs_b) global doc ID lists
    and returns qCH distances computed via the Triton pairwise kernel.

    Usage:
        computer = GpuBuildDistanceComputer(centroid_dists, codes, offsets, lengths, n_fine)
        distances = computer(pairs_a_list, pairs_b_list)  # called from Rust via PyO3
    """

    def __init__(
        self,
        centroid_dists: np.ndarray,
        flat_codes: np.ndarray,
        flat_offsets: np.ndarray,
        flat_lengths: np.ndarray,
        n_fine: int,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.n_fine = n_fine
        self._use_triton = TRITON_AVAILABLE and self.device.type == "cuda"
        self._call_count = 0
        self._total_pairs = 0

        self.centroid_dists = torch.from_numpy(
            centroid_dists.astype(np.float32).ravel()
        ).to(self.device)

        self.codes = torch.from_numpy(
            flat_codes.astype(np.int32)
        ).to(self.device)

        self.offsets = torch.from_numpy(
            flat_offsets.astype(np.int32)
        ).to(self.device)

        self.lengths = torch.from_numpy(
            flat_lengths.astype(np.int32)
        ).to(self.device)

        logger.info(
            "GpuBuildDistanceComputer: n_fine=%d, %d docs, %d codes, triton=%s",
            n_fine, flat_offsets.shape[0], flat_codes.shape[0], self._use_triton,
        )

    def __call__(self, pairs_a: list, pairs_b: list) -> list:
        """Compute qCH distances for (pairs_a[i], pairs_b[i]) on GPU."""
        self._call_count += 1
        n = len(pairs_a)
        self._total_pairs += n

        if n == 0:
            return []

        a_tensor = torch.tensor(pairs_a, dtype=torch.int32, device=self.device)
        b_tensor = torch.tensor(pairs_b, dtype=torch.int32, device=self.device)

        if self._use_triton and qch_pairwise_batch_gpu is not None:
            try:
                result = qch_pairwise_batch_gpu(
                    self.centroid_dists, self.codes,
                    self.offsets, self.lengths,
                    a_tensor, b_tensor, self.n_fine,
                )
                return result.cpu().tolist()
            except Exception as exc:
                logger.warning(
                    "Triton pairwise kernel failed (%s), falling back to PyTorch", exc
                )

        if qch_pairwise_batch_torch is not None:
            result = qch_pairwise_batch_torch(
                self.centroid_dists, self.codes,
                self.offsets, self.lengths,
                a_tensor, b_tensor, self.n_fine,
            )
            return result.cpu().tolist()

        raise RuntimeError("No GPU pairwise distance backend available")

    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_pairs": self._total_pairs,
        }


# ---------------------------------------------------------------------------
# Auto-Tiering GPU Search Accelerator
# ---------------------------------------------------------------------------

class GpuSearchAccelerator:
    """Auto-tiering GPU search: picks the fastest pipeline that fits in VRAM.

    Tier 1 (FP32):  Pre-pad all doc vectors as float32 on GPU.
                     BF qCH → FP32 MaxSim.  ~130+ QPS.  ≤335K docs on 24GB.
    Tier 2 (FP16):  Pre-pad as float16.  BF qCH → FP16 MaxSim.  ~100+ QPS.
                     ≤670K docs on 24GB.
    Tier 3 (Graph): CPU graph traversal → per-query GPU MaxSim on candidates.
                     10-30 QPS.  Unlimited scale.

    Usage::

        accel = GpuSearchAccelerator.from_segment(seg, doc_vecs, device="cuda")
        results = accel.search(query_vecs, k=10, ef=2500)
    """

    TIER_FP32 = "fp32"
    TIER_FP16 = "fp16"
    TIER_GRAPH = "graph"

    def __init__(
        self,
        segment,
        qch_scorer: GpuQchScorer,
        tier: str,
        device: torch.device,
        D_gpu: torch.Tensor | None = None,
        M_gpu: torch.Tensor | None = None,
        doc_vecs_cpu: list | None = None,
        n_docs: int = 0,
        dim: int = 128,
        default_top_n: int = 2500,
        default_ef: int = 2500,
        default_n_probes: int = 4,
    ):
        self._seg = segment
        self._scorer = qch_scorer
        self.tier = tier
        self._device = device
        self._D_gpu = D_gpu
        self._M_gpu = M_gpu
        self._doc_vecs_cpu = doc_vecs_cpu
        self.n_docs = n_docs
        self.dim = dim
        self.default_top_n = default_top_n
        self.default_ef = default_ef
        self.default_n_probes = default_n_probes

        try:
            from colsearch._internal.kernels.maxsim import fast_colbert_scores
            self._maxsim_fn = fast_colbert_scores
        except ImportError:
            raise RuntimeError("fast_colbert_scores not available — install colsearch with GPU support")

    @classmethod
    def from_segment(
        cls,
        segment,
        doc_vectors_f16: np.ndarray | None = None,
        doc_offsets: list[tuple[int, int]] | None = None,
        device: str = "cuda",
        vram_budget_gb: float | None = None,
        default_top_n: int = 2500,
        default_ef: int = 2500,
        default_n_probes: int = 4,
    ) -> "GpuSearchAccelerator":
        """Build an accelerator from a GemSegment.

        Args:
            segment: Built GemSegment with codebook/codes.
            doc_vectors_f16: Flat (total_tokens, dim) array in float16/float32.
            doc_offsets: List of (start, end) per doc into doc_vectors_f16.
            device: CUDA device string.
            vram_budget_gb: Override VRAM budget (None = auto-detect).
            default_top_n: Default candidate count for BF tiers.
            default_ef: Default ef for graph tier.
            default_n_probes: Default n_probes for graph tier.
        """
        dev = torch.device(device)
        n_docs = segment.n_docs()
        dim = 128  # default

        scorer = GpuQchScorer.from_gem_segment(segment, device=device)

        if doc_vectors_f16 is None or doc_offsets is None:
            logger.info("GpuSearchAccelerator: no doc vectors provided, using graph tier")
            return cls(
                segment=segment, qch_scorer=scorer, tier=cls.TIER_GRAPH,
                device=dev, n_docs=n_docs, dim=dim,
                default_ef=default_ef, default_n_probes=default_n_probes,
            )

        if len(doc_offsets) > 0:
            dim = doc_vectors_f16.shape[1] if doc_vectors_f16.ndim == 2 else 128

        tok_counts = [e - s for s, e in doc_offsets]
        max_tok = max(tok_counts) if tok_counts else 1
        mean_tok = sum(tok_counts) / max(len(tok_counts), 1)

        if vram_budget_gb is None:
            try:
                total_vram = torch.cuda.get_device_properties(dev).total_mem / 1e9
                allocated = torch.cuda.memory_allocated(dev) / 1e9
                vram_budget_gb = total_vram - allocated - 1.5  # leave headroom
            except Exception:
                vram_budget_gb = 20.0

        fp32_gb = n_docs * max_tok * dim * 4 / 1e9
        fp16_gb = n_docs * max_tok * dim * 2 / 1e9

        if fp32_gb <= vram_budget_gb:
            tier = cls.TIER_FP32
            dtype = torch.float32
        elif fp16_gb <= vram_budget_gb:
            tier = cls.TIER_FP16
            dtype = torch.float16
        else:
            tier = cls.TIER_GRAPH
            dtype = None

        logger.info(
            "GpuSearchAccelerator: %d docs, max_tok=%d, dim=%d, "
            "fp32=%.1fGB, fp16=%.1fGB, budget=%.1fGB → tier=%s",
            n_docs, max_tok, dim, fp32_gb, fp16_gb, vram_budget_gb, tier,
        )

        D_gpu = None
        M_gpu = None
        doc_vecs_cpu = None

        if tier in (cls.TIER_FP32, cls.TIER_FP16):
            D_cpu = np.zeros((n_docs, max_tok, dim), dtype=np.float32)
            M_cpu = np.zeros((n_docs, max_tok), dtype=np.float32)
            for i, (s, e) in enumerate(doc_offsets):
                t = e - s
                D_cpu[i, :t] = doc_vectors_f16[s:e].astype(np.float32)
                M_cpu[i, :t] = 1.0
            D_gpu = torch.from_numpy(D_cpu).to(dtype).to(dev)
            M_gpu = torch.from_numpy(M_cpu).to(dev)
            del D_cpu, M_cpu
            logger.info(
                "GpuSearchAccelerator: uploaded D_gpu %s (%.1f MB), VRAM=%.0f MB",
                list(D_gpu.shape), D_gpu.nelement() * D_gpu.element_size() / 1e6,
                torch.cuda.memory_allocated(dev) / 1e6,
            )
        else:
            doc_vecs_cpu = [
                doc_vectors_f16[s:e] for s, e in doc_offsets
            ]

        return cls(
            segment=segment, qch_scorer=scorer, tier=tier, device=dev,
            D_gpu=D_gpu, M_gpu=M_gpu, doc_vecs_cpu=doc_vecs_cpu,
            n_docs=n_docs, dim=dim, default_top_n=default_top_n,
            default_ef=default_ef, default_n_probes=default_n_probes,
        )

    @torch.no_grad()
    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        *,
        top_n: int | None = None,
        ef: int | None = None,
        n_probes: int | None = None,
    ) -> list[tuple[int, float]]:
        """Search using the auto-selected tier.

        For BF tiers: top_n controls candidate count before MaxSim rerank.
        For graph tier: ef/n_probes control graph traversal, then MaxSim
        reranks the graph candidates.

        Returns list of (doc_id, score) sorted descending.
        """
        if self.tier in (self.TIER_FP32, self.TIER_FP16):
            return self._search_bruteforce(query_vectors, k, top_n or self.default_top_n)
        else:
            return self._search_graph(
                query_vectors, k,
                ef=ef or self.default_ef,
                n_probes=n_probes or self.default_n_probes,
            )

    def _search_bruteforce(
        self, query_vectors: np.ndarray, k: int, top_n: int,
    ) -> list[tuple[int, float]]:
        """BF qCH → top_n → MaxSim rerank on pre-uploaded GPU tensor."""
        qv_gpu = torch.from_numpy(query_vectors.astype(np.float32)).to(self._device)

        proxy = self._scorer.score_query(qv_gpu)
        _, top_idxs = proxy.topk(min(top_n, self.n_docs), largest=False)

        q_t = qv_gpu.unsqueeze(0)
        if self._D_gpu.dtype == torch.float16:
            q_t = q_t.half()

        scores = self._maxsim_fn(
            q_t.float(), self._D_gpu[top_idxs].float(),
            documents_mask=self._M_gpu[top_idxs],
        ).squeeze(0)
        top_k = scores.topk(min(k, top_n))

        results = []
        for j in top_k.indices.cpu().tolist():
            doc_id = int(top_idxs[j])
            results.append((doc_id, float(scores[j])))
        return results

    def _search_graph(
        self, query_vectors: np.ndarray, k: int, ef: int, n_probes: int,
    ) -> list[tuple[int, float]]:
        """Graph traversal → candidate padding → GPU MaxSim rerank."""
        res = self._seg.search(query_vectors, k=ef, ef=ef, n_probes=n_probes)
        cand_ids = [int(did) for did, _ in res]
        if not cand_ids:
            return []

        n_rerank = min(len(cand_ids), ef)
        cand_ids = cand_ids[:n_rerank]

        batch = [self._doc_vecs_cpu[idx] for idx in cand_ids]
        max_t = max(v.shape[0] for v in batch)
        D = torch.zeros((len(cand_ids), max_t, self.dim), dtype=torch.float32, device=self._device)
        Dm = torch.zeros((len(cand_ids), max_t), dtype=torch.float32, device=self._device)
        for i, v in enumerate(batch):
            fv = torch.from_numpy(v.astype(np.float32))
            D[i, :fv.shape[0]] = fv
            Dm[i, :fv.shape[0]] = 1.0

        q_t = torch.from_numpy(query_vectors.astype(np.float32)).unsqueeze(0).to(self._device)
        scores = self._maxsim_fn(q_t, D, documents_mask=Dm).squeeze(0)
        top_k = scores.topk(min(k, len(cand_ids)))

        results = []
        for j in top_k.indices.cpu().tolist():
            results.append((cand_ids[j], float(scores[j])))
        return results

    def vram_used_mb(self) -> float:
        return torch.cuda.memory_allocated(self._device) / 1e6

    def info(self) -> dict:
        return {
            "tier": self.tier,
            "n_docs": self.n_docs,
            "dim": self.dim,
            "vram_mb": self.vram_used_mb(),
            "default_top_n": self.default_top_n,
            "default_ef": self.default_ef,
        }
