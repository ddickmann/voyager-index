"""Configuration and type aliases for Col-Bandit reranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

ShardChunk = Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]

@dataclass(slots=True)
class ColBanditConfig:
    relaxation_eps: float = 0.01
    max_rounds: int = 8
    reveal_query_tokens_per_round: int = 2
    min_candidates_for_bandit: int = 64
    exact_survivor_cap: int = 256
    fallback_full_maxsim: bool = True

