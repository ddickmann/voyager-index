"""
Feature and MaxSim bridges for OSS runtime integrations.
"""

from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np
import torch

from ...kernels.maxsim import fast_colbert_scores

logger = logging.getLogger(__name__)


class FeatureBridge:
    """
    Legacy bridge placeholder.

    The packaged Python OR helpers and research-side feature bridge were moved to
    the private repo during the OSS/private split. The OSS runtime keeps only
    `MaxSimBridge`, which is still used by the public retrieval stack.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "FeatureBridge is no longer part of the OSS runtime. "
            "Use the private extension repo for the solver-side bridge."
        )


class MaxSimBridge:
    """
    Bridge for ColBERT MaxSim scoring.

    Uses the Triton MaxSim kernel for high-performance relevance scoring of
    multi-token embeddings.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def compute_maxsim_scores(
        self,
        query_embeddings: Union[torch.Tensor, np.ndarray],
        candidate_embeddings: list[Union[torch.Tensor, np.ndarray]],
    ) -> torch.Tensor:
        """
        Compute MaxSim scores using the shared kernel path.
        """
        if isinstance(query_embeddings, np.ndarray):
            query_tensor = torch.from_numpy(query_embeddings).to(device=self.device, dtype=torch.float32)
        else:
            query_tensor = query_embeddings.to(device=self.device, dtype=torch.float32)

        if query_tensor.dim() == 2:
            query_tensor = query_tensor.unsqueeze(0)

        documents: list[torch.Tensor] = []
        for embedding in candidate_embeddings:
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding)
            documents.append(embedding.to(device=self.device, dtype=torch.float32))

        try:
            scores = fast_colbert_scores(query_tensor, documents)
        except Exception as exc:
            logger.warning("MaxSim kernel path failed, using local CPU fallback: %s", exc)
            return self._maxsim_cpu(query_embeddings, candidate_embeddings)

        return scores.squeeze(0)

    def _maxsim_cpu(
        self,
        query_embeddings: Union[torch.Tensor, np.ndarray],
        candidate_embeddings: list[Union[torch.Tensor, np.ndarray]],
    ) -> torch.Tensor:
        """CPU fallback for MaxSim."""
        if isinstance(query_embeddings, np.ndarray):
            query_tensor = torch.from_numpy(query_embeddings)
        else:
            query_tensor = query_embeddings

        if query_tensor.dim() == 2:
            query_tensor = query_tensor.unsqueeze(0)

        scores = []
        for embedding in candidate_embeddings:
            if isinstance(embedding, np.ndarray):
                document_tensor = torch.from_numpy(embedding)
            else:
                document_tensor = embedding

            if document_tensor.dim() == 2:
                document_tensor = document_tensor.unsqueeze(0)

            similarities = torch.bmm(query_tensor, document_tensor.transpose(1, 2))
            max_sim = similarities.max(dim=-1).values
            scores.append(max_sim.sum().item())

        return torch.tensor(scores)
