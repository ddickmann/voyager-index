"""Pinned-buffer support for shard fetch staging."""
from __future__ import annotations

import queue

import torch

from .common import logger

class PinnedBufferPool:
    """Pool of pre-allocated pinned host memory buffers."""

    def __init__(self, max_tokens: int, dim: int, n_buffers: int = 3):
        self.max_tokens = max_tokens
        self.dim = dim
        self._pool: queue.Queue[torch.Tensor] = queue.Queue()
        use_pinned = torch.cuda.is_available()
        self.uses_pinned_memory = False
        for _ in range(n_buffers):
            if use_pinned:
                try:
                    buf = torch.empty(max_tokens, dim, dtype=torch.float16, pin_memory=True)
                    self.uses_pinned_memory = True
                except RuntimeError as exc:
                    logger.warning(
                        "Pinned host allocation unavailable, using pageable buffers: %s",
                        exc,
                    )
                    use_pinned = False
                    buf = torch.empty(max_tokens, dim, dtype=torch.float16)
            else:
                buf = torch.empty(max_tokens, dim, dtype=torch.float16)
            self._pool.put(buf)

    def get(self) -> torch.Tensor:
        return self._pool.get()

    def release(self, buf: torch.Tensor):
        self._pool.put(buf)

    @property
    def available(self) -> int:
        return self._pool.qsize()

    @property
    def staging_mode(self) -> str:
        return "pinned" if self.uses_pinned_memory else "pageable_fallback"

