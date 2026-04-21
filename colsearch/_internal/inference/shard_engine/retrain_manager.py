from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from .lemur_router import LemurRouter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrainPolicy:
    retrain_every_ops: int = 50_000
    retrain_dirty_doc_ratio: float = 0.05
    retrain_dirty_shard_ratio: float = 0.10
    retrain_every_hours: float = 24.0
    canary_recall_drop_limit: float = 0.01


class RouterRetrainManager:
    """Shadow-generation retraining controller for LemurRouter.

    The manager is deliberately storage-agnostic. Callers provide the latest
    pooled document snapshot through ``snapshot_fn`` and an optional canary
    validator through ``validate_fn``.
    """

    def __init__(
        self,
        router_dir: Path,
        policy: RetrainPolicy,
        snapshot_fn: Callable[[], tuple[torch.Tensor, torch.Tensor, list[int], dict[int, int]]],
        validate_fn: Optional[Callable[[Path], float]] = None,
    ) -> None:
        self.router_dir = Path(router_dir)
        self.policy = policy
        self.snapshot_fn = snapshot_fn
        self.validate_fn = validate_fn
        self._last_full_retrain_ts = time.time()

    def maybe_retrain(self, live_router: LemurRouter) -> bool:
        age_hours = (time.time() - self._last_full_retrain_ts) / 3600.0
        if not live_router.should_full_retrain(
            retrain_every_ops=self.policy.retrain_every_ops,
            retrain_dirty_doc_ratio=self.policy.retrain_dirty_doc_ratio,
            retrain_dirty_shard_ratio=self.policy.retrain_dirty_shard_ratio,
        ) and age_hours < self.policy.retrain_every_hours:
            return False

        shadow_dir = self.router_dir.parent / f"{self.router_dir.name}_shadow"
        if shadow_dir.exists():
            for p in shadow_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        shadow_dir.mkdir(parents=True, exist_ok=True)

        pooled_vectors, pooled_counts, doc_ids, doc_id_to_shard = self.snapshot_fn()
        shadow = LemurRouter(index_dir=shadow_dir, ann_backend=live_router.ann_backend, device=live_router.device)
        shadow.fit_initial(
            pooled_doc_vectors=pooled_vectors,
            pooled_doc_counts=pooled_counts,
            doc_ids=doc_ids,
            doc_id_to_shard=doc_id_to_shard,
            epochs=10,
        )

        if self.validate_fn is not None:
            recall_drop = float(self.validate_fn(shadow_dir))
            if recall_drop > self.policy.canary_recall_drop_limit:
                logger.warning(
                    "Rejecting shadow LEMUR generation because canary recall drop %.4f exceeded limit %.4f",
                    recall_drop,
                    self.policy.canary_recall_drop_limit,
                )
                return False

        # Atomic-enough swap for single-host benchmark usage.
        for p in self.router_dir.glob("*"):
            if p.is_file():
                p.unlink()
        for src in shadow_dir.glob("*"):
            src.replace(self.router_dir / src.name)
        self._last_full_retrain_ts = time.time()
        return True
