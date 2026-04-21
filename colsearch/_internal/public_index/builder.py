"""Fluent builder helpers for the public `Index` facade."""

from __future__ import annotations

from typing import Any, Dict


class IndexBuilder:
    """
    Fluent builder for creating an :class:`Index` with custom configuration.

    Every ``with_*`` method returns ``self`` so calls can be chained::

        idx = (IndexBuilder("my_index", dim=128)
               .with_shard(n_shards=64, k_candidates=512)
               .with_wal(enabled=True)
               .build())

    Args:
        path: Directory to store the index.
        dim: Vector dimensionality (must match the embedding model).
    """

    def __init__(self, path: str, dim: int) -> None:
        self._path = path
        self._dim = dim
        self._engine = "auto"
        self._kwargs: Dict[str, Any] = {}

    def with_gem(self, **kwargs: Any) -> "IndexBuilder":
        """Select the GEM engine with optional keyword overrides.

        Common kwargs: ``seed_batch_size``, ``n_fine``, ``n_coarse``,
        ``max_degree``, ``ef_construction``, ``dual_graph``, ``use_emd``.
        """
        self._engine = "gem"
        self._kwargs.update(kwargs)
        return self

    def with_hnsw(self, **kwargs: Any) -> "IndexBuilder":
        """Select the HNSW engine (legacy single-vector backend)."""
        self._engine = "hnsw"
        self._kwargs.update(kwargs)
        return self

    def with_shard(self, **kwargs: Any) -> "IndexBuilder":
        """Select the LEMUR-routed shard engine for scalable late-interaction retrieval.

        Common kwargs: ``n_shards``, ``k_candidates``, ``lemur_epochs``,
        ``compression``, ``device``.
        """
        self._engine = "shard"
        self._kwargs.update(kwargs)
        return self

    def with_wal(self, enabled: bool = True) -> "IndexBuilder":
        """Enable or disable the write-ahead log for crash recovery.

        Args:
            enabled: ``True`` (default) enables WAL + CRC32 checkpointing.
        """
        self._kwargs["enable_wal"] = enabled
        return self

    def with_quantization(self, n_fine: int = 256, n_coarse: int = 32) -> "IndexBuilder":
        """Configure the two-stage codebook for qCH proxy scoring.

        Args:
            n_fine: Number of fine centroids (higher = better proxy accuracy,
                more build time). Recommended: 128-2048 depending on corpus.
            n_coarse: Number of coarse routing clusters.
        """
        self._kwargs["n_fine"] = n_fine
        self._kwargs["n_coarse"] = n_coarse
        return self

    def with_gpu_rerank(self, device: str = "cuda") -> "IndexBuilder":
        """Enable GPU-accelerated MaxSim reranking.

        GEM proxy search produces candidates ranked by quantized Chamfer
        distance. This option adds a second-stage exact late-interaction
        rerank on the specified device for higher recall.

        Args:
            device: PyTorch device string (default ``'cuda'``).
        """
        self._kwargs["rerank_device"] = device
        return self

    def with_roq(self, bits: int = 4, device: str = "cuda") -> "IndexBuilder":
        """Enable ROQ (Rotational Quantization) compressed reranking.

        Stores document vectors in ``bits``-bit quantized form and runs
        MaxSim reranking with a fused Triton kernel. Reduces memory by
        about 8x (at 4-bit) compared to FP32 with minimal recall loss.

        Args:
            bits: Quantization bit width (default 4).
            device: PyTorch device string (default ``'cuda'``).
        """
        self._kwargs["roq_rerank"] = True
        self._kwargs["roq_bits"] = bits
        self._kwargs["rerank_device"] = device
        return self

    def build(self):
        from colsearch.index import Index

        return Index(
            self._path,
            self._dim,
            engine=self._engine,
            **self._kwargs,
        )


__all__ = ["IndexBuilder"]
