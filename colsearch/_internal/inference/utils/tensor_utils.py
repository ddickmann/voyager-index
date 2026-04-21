"""Tensor utility functions."""

from typing import List, Union

import numpy as np
import torch


def convert_to_tensor(x: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
    """
    Convert various input types to PyTorch tensor.

    Args:
        x: Input data (tensor, numpy array, or list)

    Returns:
        PyTorch tensor
    """
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list):
        # Try to stack if all elements are tensors
        if all(isinstance(item, torch.Tensor) for item in x):
            return torch.stack(x)
        else:
            return torch.tensor(x)
    else:
        return torch.tensor(x)


def pad_embeddings(
    embeddings_list: List[torch.Tensor],
    pad_value: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of variable-length embeddings to the same length.

    Args:
        embeddings_list: List of embeddings with shape (seq_len_i, hidden_dim)
        pad_value: Value to use for padding

    Returns:
        Tuple of (padded, mask):
            - padded: (batch_size, max_seq_len, hidden_dim)
            - mask: (batch_size, max_seq_len), 1 for real tokens, 0 for padding
    """
    if not embeddings_list:
        raise ValueError("embeddings_list cannot be empty")

    # Handle both 2D and 3D inputs
    if embeddings_list[0].dim() == 3:
        # Already batched (B, S, H) - just return
        return embeddings_list[0], None

    # Get max length and hidden dim
    max_len = max(emb.shape[0] for emb in embeddings_list)
    hidden_dim = embeddings_list[0].shape[1]
    device = embeddings_list[0].device
    dtype = embeddings_list[0].dtype

    # Create padded tensor and mask
    batch_size = len(embeddings_list)
    padded = torch.full((batch_size, max_len, hidden_dim), pad_value, dtype=dtype, device=device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

    # Fill in the actual values
    for i, emb in enumerate(embeddings_list):
        seq_len = emb.shape[0]
        padded[i, :seq_len] = emb
        mask[i, :seq_len] = 1.0

    return padded, mask

