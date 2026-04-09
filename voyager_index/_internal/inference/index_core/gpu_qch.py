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
    from .triton_qch_kernel import TRITON_AVAILABLE, qch_max_gather_gpu, qch_max_gather_torch
except ImportError:
    TRITON_AVAILABLE = False
    qch_max_gather_gpu = None
    qch_max_gather_torch = None


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
