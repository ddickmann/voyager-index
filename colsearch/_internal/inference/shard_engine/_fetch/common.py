"""Shared helpers for shard fetch pipeline internals."""
from __future__ import annotations

import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

ShardChunk = Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]
