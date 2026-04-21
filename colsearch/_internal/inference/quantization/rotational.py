"""
Rotational Quantization (RoQ) for Embeddings using FWHT.

Implements Weaviate-style 8-bit Rotational Quantization using Fast Walsh-Hadamard Transform
(FWHT) for O(D log D) fast pseudorandom rotation.

Key features:
- Fast pseudorandom rotation (checking FWHT)
- 8-bit per-sample quantization
- Symmetric distance estimation
- Metadata storage: c(x), l_x, delta_x, ||x||^2

References:
- https://weaviate.io/blog/8-bit-rotational-quantization
- NumPy FWHT implementation
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RoQConfig:
    """Configuration for Rotational Quantization."""
    dim: int                       # Embedding dimension
    num_bits: int = 8              # Bits per dimension
    num_rounds: int = 3            # Number of rotation rounds
    block_size: int = None         # Block size for FWHT (computed from dim if None)
    seed: int = 42                 # Random seed for reproducibility
    group_size: Optional[int] = None  # Optional group-wise scale size (2-bit/4-bit)
    query_bits: Optional[int] = None  # Optional asymmetric query quantization for screening

    def __post_init__(self):
        if self.num_bits not in (1, 2, 4, 8):
            raise ValueError("num_bits must be one of 1, 2, 4, or 8")
        # Compute block_size as next power of 2 >= dim (or equal to dim if already power of 2)
        if self.block_size is None:
            # Find next power of 2
            p = 1
            while p < self.dim:
                p *= 2
            self.block_size = p
        # 1-bit screening benefits materially from a wider padded rotational space.
        if self.num_bits == 1 and self.block_size < 256:
            self.block_size = 256
        if self.group_size is None and self.num_bits == 2 and self.dim >= 16:
            # Group-wise 2-bit quantization is a much better default screening profile.
            self.group_size = 16
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive when provided")
        if self.group_size is not None and self.num_bits not in (2, 4):
            raise ValueError("group_size is only supported for 2-bit and 4-bit RoQ")
        if self.query_bits is None and self.num_bits == 1:
            # Mirror the asymmetric query-precision story used in robust 1-bit screening.
            self.query_bits = 5
        if self.query_bits is not None and self.query_bits <= 0:
            raise ValueError("query_bits must be positive when provided")


class FastWalshHadamard:
    """
    Fast Walsh-Hadamard Transform (FWHT) based pseudorandom rotation.

    Applies multiple rounds of:
    1. Random sign flipping D_i
    2. Blocked Walsh-Hadamard Transform H
    3. Random permutation P_i (swapping entries)

    R(x) = ... P_2 H D_2 P_1 H D_1 x
    """

    def __init__(self, dim: int, num_rounds: int = 3, block_size: int = 256, seed: int = 42):
        self.dim = dim
        self.num_rounds = num_rounds
        self.block_size = block_size
        self.seed = seed

        # Verify block size is power of 2
        assert (block_size & (block_size - 1) == 0) and block_size > 0

        # Precompute random signs and permutations
        self.rng = np.random.RandomState(seed)
        self.signs = []
        self.permutations = []

        # Calculate padded dimension (multiple of block_size)
        self.padded_dim = ((dim + block_size - 1) // block_size) * block_size

        for _ in range(num_rounds):
            # Random signs {-1, 1}
            s = self.rng.choice([-1.0, 1.0], size=self.padded_dim).astype(np.float32)
            self.signs.append(torch.from_numpy(s))

            # Random permutation
            p = self.rng.permutation(self.padded_dim)
            self.permutations.append(torch.from_numpy(p))

    def _fwht(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fast Walsh-Hadamard Transform in-place on last dimension.
        Iterative implementation for powers of 2.
        """
        n = x.shape[-1]
        h = 1
        while h < n:
            # Reshape to (..., n/(2h), 2, h)
            x = x.view(*x.shape[:-1], -1, 2, h)
            # Butterfly
            a = x[..., 0, :]  # First half of each pair
            b = x[..., 1, :]  # Second half
            x = torch.stack([a + b, a - b], dim=-2)
            # Flatten back to (..., n)
            x = x.view(*x.shape[:-3], -1)
            h *= 2
        return x / np.sqrt(n)  # Normalized FWHT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pseudorandom rotation to input vectors.
        Args:
            x: Input embeddings (N, D)
        Returns:
            Rotated embeddings (N, D) - padded dimensions are trimmed if needed
            Wait, we want to keep them somewhat padded/permuted?
            No, we should probably return original dimension for simplicity,
            or keep padded for quantization efficiency.

            Weaviate keeps them padded/transformed for quantization.
            For simplicity here, we'll return original dim but typically entire pipeline uses padded.
            Let's return padded for now, quantization handles it.
        """
        # Move params to x device if needed
        device = x.device
        dtype = x.dtype

        N = x.shape[0]

        # Pad input
        if self.padded_dim > self.dim:
            padding = torch.zeros(N, self.padded_dim - self.dim, device=device, dtype=dtype)
            x_curr = torch.cat([x, padding], dim=1)
        else:
            x_curr = x.clone()

        for i in range(self.num_rounds):
            # 1. Random signs
            signs = self.signs[i].to(device)
            x_curr = x_curr * signs

            # 2. Blocked FWHT
            # Reshape to (Batch, Blocks, BlockSize)
            reshaped = x_curr.view(N, -1, self.block_size)
            transformed = self._fwht(reshaped)
            x_curr = transformed.view(N, -1)

            # 3. Random permutation (swapping)
            if i < self.num_rounds - 1: # Permute between rounds, maybe not last?
                # Weaviate says: "Between each round of block-smoothening we randomly swap elements"
                perm = self.permutations[i].to(device)
                x_curr = x_curr[:, perm]

        # Trim back to original dim?
        # Rotation spreads info across ALL dimensions (including padding).
        # Trimming would lose info. We must start with padded dim.
        # But for 'quantization', we usually quantize the rotated vector.
        # We will return the full padded rotated vector.
        return x_curr


class RotationalQuantizer:
    """
    8-bit Rotational Quantizer.

    Structure per vector:
    - code: D bytes (uint8)
    - scale: float32
    - offset: float32
    - norm_sq: float32
    """

    def __init__(self, config: RoQConfig):
        self.config = config
        self.rotator = FastWalshHadamard(
            dim=config.dim,
            num_rounds=config.num_rounds,
            block_size=config.block_size,
            seed=config.seed
        )
        self.effective_dim = self.rotator.padded_dim

        # Storage
        self._codes: Optional[np.ndarray] = None    # (N, D_padded) uint8
        self._scales: Optional[np.ndarray] = None   # (N,) float32
        self._offsets: Optional[np.ndarray] = None  # (N,) float32
        self._norms_sq: Optional[np.ndarray] = None # (N,) float32
        self._code_sums: Optional[np.ndarray] = None
        if self.config.num_bits in (2, 4) and self.config.group_size is not None:
            if self.effective_dim % self.config.group_size != 0:
                raise ValueError(
                    f"{self.config.num_bits}-bit group_size={self.config.group_size} must divide the padded dimension {self.effective_dim}"
                )

    def fit(self, x: Union[np.ndarray, torch.Tensor]):
        """Nothing to train for FWHT rotation."""
        pass

    def _group_size(self, meta: Optional[torch.Tensor] = None) -> int:
        if self.config.num_bits not in (2, 4):
            return self.effective_dim
        if meta is not None and meta.ndim >= 2 and meta.shape[1] > 1:
            inferred = int(self.effective_dim // meta.shape[1])
            if inferred > 0 and self.effective_dim % meta.shape[1] == 0:
                return inferred
        return int(self.config.group_size or self.effective_dim)

    @staticmethod
    def _pack_lowbit_codes(quantized: torch.Tensor, bits_per_code: int) -> torch.Tensor:
        if bits_per_code == 1:
            return torch.from_numpy(np.packbits(quantized.cpu().numpy().astype(np.uint8), axis=1))
        if bits_per_code == 2:
            n_rows, dim = quantized.shape
            pad = (-dim) % 4
            if pad:
                quantized = F.pad(quantized, (0, pad))
            groups = quantized.view(n_rows, -1, 4)
            return ((groups[:, :, 0] << 6) | (groups[:, :, 1] << 4) | (groups[:, :, 2] << 2) | groups[:, :, 3]).contiguous()
        if bits_per_code == 4:
            n_rows, dim = quantized.shape
            pad = (-dim) % 2
            if pad:
                quantized = F.pad(quantized, (0, pad))
            groups = quantized.view(n_rows, -1, 2)
            return ((groups[:, :, 0] << 4) | groups[:, :, 1]).contiguous()
        if bits_per_code == 8:
            return quantized.contiguous()
        raise ValueError(f"Unsupported bits_per_code={bits_per_code}")

    @staticmethod
    def _scalar_quantize_tensor(values: torch.Tensor, bits: int) -> torch.Tensor:
        if bits <= 0:
            return values
        levels = float((1 << bits) - 1)
        min_vals = values.min(dim=1).values
        max_vals = values.max(dim=1).values
        ranges = max_vals - min_vals
        ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)
        scales = ranges / levels
        quantized = ((values - min_vals.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, int(levels))
        return quantized * scales.unsqueeze(1) + min_vals.unsqueeze(1)

    def _project_queries(self, queries: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(queries, np.ndarray):
            queries_t = torch.from_numpy(queries)
        else:
            queries_t = queries
        if queries_t.ndim == 1:
            queries_t = queries_t.unsqueeze(0)
        device = queries_t.device if queries_t.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rotated = self.rotator.forward(queries_t.float().to(device))
        if self.config.query_bits is not None:
            return self._scalar_quantize_tensor(rotated, self.config.query_bits)
        return rotated

    def build_1bit_query_triton_inputs(
        self,
        queries: Union[np.ndarray, torch.Tensor],
        *,
        batch_size: int,
        item_count: int,
        device: Union[str, torch.device] = "cpu",
        include_norm_sq: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.num_bits != 1:
            raise ValueError("1-bit Triton query inputs are only valid for 1-bit RoQ")
        if self.config.query_bits is None:
            raise ValueError("1-bit Triton query inputs require query_bits to be configured")
        if isinstance(queries, np.ndarray):
            queries_t = torch.from_numpy(queries)
        else:
            queries_t = queries
        queries_t = queries_t.float().to(device)
        if queries_t.ndim != 2:
            raise ValueError(f"Expected flattened query matrix, got shape {tuple(queries_t.shape)}")
        expected = batch_size * item_count
        if queries_t.shape[0] != expected:
            raise ValueError(f"Expected {expected} flattened queries, got {queries_t.shape[0]}")

        rotated = self.rotator.forward(queries_t)
        levels = float((1 << self.config.query_bits) - 1)
        min_vals = rotated.min(dim=1).values
        max_vals = rotated.max(dim=1).values
        ranges = max_vals - min_vals
        ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)
        scales = ranges / levels
        quantized = ((rotated - min_vals.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, int(levels)).to(torch.uint8)
        code_sums = quantized.float().sum(dim=1)
        meta_fields = [scales, min_vals, code_sums]
        if include_norm_sq:
            meta_fields.append(torch.sum(queries_t ** 2, dim=1))
        meta = torch.stack(meta_fields, dim=1).to(torch.float32)
        quantized_np = np.ascontiguousarray(quantized.detach().to("cpu", dtype=torch.uint8).numpy())
        query_planes = []
        for bit_idx in range(int(self.config.query_bits)):
            plane = ((quantized_np >> bit_idx) & 0x01).astype(np.uint8, copy=False)
            packed = np.packbits(plane, axis=1)
            query_planes.append(packed.view(np.uint32).view(np.int32))
        planes = np.stack(query_planes, axis=1)
        return (
            torch.from_numpy(planes).reshape(batch_size, item_count, int(self.config.query_bits), -1).to(device=device, dtype=torch.int32),
            meta.reshape(batch_size, item_count, -1).to(dtype=torch.float32),
        )

    def build_1bit_doc_triton_inputs(
        self,
        quantized: dict[str, Any],
        *,
        batch_size: int,
        item_count: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        if self.config.num_bits != 1:
            raise ValueError("1-bit Triton doc inputs are only valid for 1-bit RoQ")
        codes = quantized["codes"]
        if isinstance(codes, torch.Tensor):
            codes_np = np.ascontiguousarray(codes.detach().to("cpu", dtype=torch.uint8).numpy())
        else:
            codes_np = np.ascontiguousarray(np.asarray(codes, dtype=np.uint8))
        if codes_np.ndim != 2:
            raise ValueError(f"Expected flattened packed 1-bit codes, got shape {codes_np.shape}")
        if codes_np.shape[0] != batch_size * item_count:
            raise ValueError(
                f"Expected {batch_size * item_count} flattened 1-bit doc codes, got {codes_np.shape[0]}"
            )
        if codes_np.shape[1] % 4 != 0:
            raise ValueError(f"Expected packed byte width divisible by 4, got {codes_np.shape[1]}")
        words = codes_np.view(np.uint32).view(np.int32)
        return torch.from_numpy(words).reshape(batch_size, item_count, -1).to(device=device, dtype=torch.int32)

    @staticmethod
    def _reshape_meta(
        meta: Union[np.ndarray, torch.Tensor],
        *,
        batch_size: int,
        effective_dim: int,
    ) -> torch.Tensor:
        if isinstance(meta, np.ndarray):
            tensor = torch.from_numpy(meta)
        else:
            tensor = meta
        tensor = tensor.float()
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            if batch_size == 1 and tensor.numel() > 1 and effective_dim % tensor.numel() == 0:
                return tensor.reshape(1, -1)
            return tensor.reshape(-1, 1)
        return tensor

    def _unpack_codes(
        self,
        codes: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if isinstance(codes, np.ndarray):
            codes_t = torch.from_numpy(codes)
        else:
            codes_t = codes
        if device is not None:
            codes_t = codes_t.to(device)
        if self.config.num_bits == 8:
            return codes_t.float()
        if self.config.num_bits == 1:
            bit_offsets = torch.arange(7, -1, -1, device=codes_t.device, dtype=torch.uint8)
            unpacked = ((codes_t.unsqueeze(-1) >> bit_offsets) & 0x01).reshape(codes_t.shape[0], -1)
            return unpacked[:, :self.effective_dim].float()
        if self.config.num_bits == 2:
            unpacked = torch.stack(
                [
                    (codes_t >> 6) & 0x03,
                    (codes_t >> 4) & 0x03,
                    (codes_t >> 2) & 0x03,
                    codes_t & 0x03,
                ],
                dim=-1,
            ).reshape(codes_t.shape[0], -1)
            return unpacked[:, :self.effective_dim].float()
        if self.config.num_bits == 4:
            high = (codes_t >> 4) & 0x0F
            low = codes_t & 0x0F
            unpacked = torch.stack([high, low], dim=-1).reshape(codes_t.shape[0], -1)
            return unpacked[:, :self.effective_dim].float()
        raise NotImplementedError(f"Unpacking {self.config.num_bits}-bit codes is not implemented")

    def _asymmetric_inner_products(
        self,
        queries: Union[np.ndarray, torch.Tensor],
        doc_codes: Union[np.ndarray, torch.Tensor],
        doc_scales: Union[np.ndarray, torch.Tensor],
        doc_offsets: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        rotated_queries = self._project_queries(queries)
        device = rotated_queries.device
        scales_t = self._reshape_meta(doc_scales, batch_size=doc_codes.shape[0], effective_dim=self.effective_dim).to(device)
        offsets_t = self._reshape_meta(doc_offsets, batch_size=doc_codes.shape[0], effective_dim=self.effective_dim).to(device)

        unpacked = self._unpack_codes(doc_codes, device=device)
        if self.config.num_bits == 8 or scales_t.shape[1] == 1:
            score_dot = rotated_queries @ unpacked.T
            query_sums = rotated_queries.sum(dim=1, keepdim=True)
            return score_dot * scales_t.reshape(1, -1) + query_sums * offsets_t.reshape(1, -1)

        group_size = self._group_size(scales_t)
        query_groups = rotated_queries.reshape(rotated_queries.shape[0], -1, group_size)
        code_groups = unpacked.reshape(unpacked.shape[0], -1, group_size)
        query_sums = query_groups.sum(dim=-1)
        dot = torch.einsum("qgd,ngd->qng", query_groups, code_groups)
        return (query_sums[:, None, :] * offsets_t[None, :, :] + dot * scales_t[None, :, :]).sum(dim=-1)

    def _symmetric_inner_products(
        self,
        query_codes: Union[np.ndarray, torch.Tensor],
        query_scales: Union[np.ndarray, torch.Tensor],
        query_offsets: Union[np.ndarray, torch.Tensor],
        doc_codes: Union[np.ndarray, torch.Tensor],
        doc_scales: Union[np.ndarray, torch.Tensor],
        doc_offsets: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_codes_t = self._unpack_codes(query_codes, device=device)
        d_codes_t = self._unpack_codes(doc_codes, device=device)
        q_scales_t = self._reshape_meta(query_scales, batch_size=q_codes_t.shape[0], effective_dim=self.effective_dim).to(device)
        q_offsets_t = self._reshape_meta(query_offsets, batch_size=q_codes_t.shape[0], effective_dim=self.effective_dim).to(device)
        d_scales_t = self._reshape_meta(doc_scales, batch_size=d_codes_t.shape[0], effective_dim=self.effective_dim).to(device)
        d_offsets_t = self._reshape_meta(doc_offsets, batch_size=d_codes_t.shape[0], effective_dim=self.effective_dim).to(device)

        if self.config.num_bits == 8 or (q_scales_t.shape[1] == 1 and d_scales_t.shape[1] == 1):
            q_sum = q_codes_t.sum(dim=1, keepdim=True)
            d_sum = d_codes_t.sum(dim=1, keepdim=True)
            dot = q_codes_t @ d_codes_t.T
            return (
                (self.effective_dim * q_offsets_t.reshape(-1, 1) * d_offsets_t.reshape(1, -1))
                + (q_offsets_t.reshape(-1, 1) * d_scales_t.reshape(1, -1) * d_sum.reshape(1, -1))
                + (q_scales_t.reshape(-1, 1) * q_sum.reshape(-1, 1) * d_offsets_t.reshape(1, -1))
                + (q_scales_t.reshape(-1, 1) * d_scales_t.reshape(1, -1) * dot)
            )

        group_size = self._group_size(q_scales_t)
        q_groups = q_codes_t.reshape(q_codes_t.shape[0], -1, group_size)
        d_groups = d_codes_t.reshape(d_codes_t.shape[0], -1, group_size)
        q_sum = q_groups.sum(dim=-1)
        d_sum = d_groups.sum(dim=-1)
        dot = torch.einsum("qgd,ngd->qng", q_groups, d_groups)
        return (
            (
                group_size * q_offsets_t[:, None, :] * d_offsets_t[None, :, :]
                + q_offsets_t[:, None, :] * d_scales_t[None, :, :] * d_sum[None, :, :]
                + q_scales_t[:, None, :] * q_sum[:, None, :] * d_offsets_t[None, :, :]
                + q_scales_t[:, None, :] * d_scales_t[None, :, :] * dot
            )
            .sum(dim=-1)
        )

    def approximate_scores(
        self,
        queries: Union[np.ndarray, torch.Tensor],
        doc_codes: Union[np.ndarray, torch.Tensor],
        doc_scales: Union[np.ndarray, torch.Tensor],
        doc_offsets: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        return self._asymmetric_inner_products(
            queries,
            doc_codes,
            doc_scales,
            doc_offsets,
        ).detach().cpu().numpy()

    def symmetric_scores(
        self,
        query_codes: Union[np.ndarray, torch.Tensor],
        query_scales: Union[np.ndarray, torch.Tensor],
        query_offsets: Union[np.ndarray, torch.Tensor],
        doc_codes: Union[np.ndarray, torch.Tensor],
        doc_scales: Union[np.ndarray, torch.Tensor],
        doc_offsets: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        return self._symmetric_inner_products(
            query_codes,
            query_scales,
            query_offsets,
            doc_codes,
            doc_scales,
            doc_offsets,
        ).detach().cpu().numpy()

    @staticmethod
    def _flatten_triton_meta_field(field: Any, name: str) -> np.ndarray:
        array = np.asarray(field, dtype=np.float32)
        if array.ndim == 0:
            return array.reshape(1)
        if array.ndim == 2 and array.shape[1] == 1:
            return array.reshape(-1)
        if array.ndim != 1:
            raise ValueError(
                f"Triton RoQ expects scalar per-vector {name}; got shape {tuple(array.shape)}"
            )
        return array

    def build_triton_meta(self, quantized: dict[str, Any], *, include_norm_sq: bool = True) -> np.ndarray:
        if self.config.num_bits == 1:
            raise ValueError("1-bit RoQ uses asymmetric screening scores rather than Triton affine metadata")
        scales = self._flatten_triton_meta_field(quantized["scales"], "scale")
        offsets = self._flatten_triton_meta_field(quantized["offsets"], "offset")
        code_sums = self._flatten_triton_meta_field(quantized["code_sums"], "code_sum")
        meta_fields = [scales, offsets, code_sums]
        if include_norm_sq:
            norms_sq = self._flatten_triton_meta_field(quantized["norms_sq"], "norm_sq")
            meta_fields.append(norms_sq)
        if len({len(field) for field in meta_fields}) != 1:
            raise ValueError("Incompatible Triton RoQ metadata lengths")
        return np.stack(meta_fields, axis=1).astype(np.float32, copy=False)

    def stack_triton_inputs(
        self,
        quantized: dict[str, Any],
        *,
        batch_size: int,
        item_count: int,
        device: Union[str, torch.device] = "cpu",
        include_norm_sq: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes = np.asarray(quantized["codes"], dtype=np.uint8)
        meta = self.build_triton_meta(quantized, include_norm_sq=include_norm_sq)
        expected = batch_size * item_count
        if codes.shape[0] != expected or meta.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} quantized rows for Triton reshape, got codes={codes.shape[0]} meta={meta.shape[0]}"
            )
        return (
            torch.as_tensor(codes.reshape(batch_size, item_count, -1), dtype=torch.uint8, device=device),
            torch.as_tensor(meta.reshape(batch_size, item_count, meta.shape[1]), dtype=torch.float32, device=device),
        )

    def quantize(self, x: Union[np.ndarray, torch.Tensor], store: bool = True) -> dict:
        """
        Quantize embeddings.

        Args:
            x: Embeddings (N, D)
            store: Whether to update internal index

        Returns:
            Dictionary with quantized data
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()

        # 1. Compute original norms (squared)
        # Store ||x||^2 for distance correction (needed for 8-bit, less critical for 1-bit but good to have)
        norms_sq = torch.sum(x ** 2, dim=1)

        # 2. Rotate
        rotated = self.rotator.forward(x)

        results = {}

        if self.config.num_bits == 1:
            bits = (rotated >= 0).to(torch.uint8)
            packed = self._pack_lowbit_codes(bits, 1)
            results = {
                'codes': packed.cpu().numpy(),
                # sign(x) = -1 + 2 * bit(x)
                'scales': np.full((rotated.shape[0],), 2.0, dtype=np.float32),
                'offsets': np.full((rotated.shape[0],), -1.0, dtype=np.float32),
                'norms_sq': norms_sq.cpu().numpy(),
                'code_sums': bits.sum(dim=1).cpu().numpy().astype(np.float32),
                'query_bits': self.config.query_bits,
                'meta_layout': 'binary_screening',
            }

        elif self.config.num_bits == 2:
            group_size = self.config.group_size or self.effective_dim
            if group_size == self.effective_dim:
                min_vals = rotated.min(dim=1).values
                max_vals = rotated.max(dim=1).values
                ranges = max_vals - min_vals
                ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)
                scale = ranges / 3.0
                offset = min_vals
                quantized = ((rotated - offset.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 3).to(torch.uint8)
                packed = self._pack_lowbit_codes(quantized, 2)
                results = {
                    'codes': packed.cpu().numpy(),
                    'scales': scale.cpu().numpy(),
                    'offsets': offset.cpu().numpy(),
                    'norms_sq': norms_sq.cpu().numpy(),
                    'code_sums': quantized.float().sum(dim=1).cpu().numpy(),
                    'meta_layout': 'scalar',
                }
            else:
                n_groups = self.effective_dim // group_size
                grouped = rotated.view(rotated.shape[0], n_groups, group_size)
                min_vals = grouped.min(dim=2).values
                max_vals = grouped.max(dim=2).values
                ranges = max_vals - min_vals
                ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)
                scales = ranges / 3.0
                quantized = ((grouped - min_vals.unsqueeze(-1)) / scales.unsqueeze(-1)).round().clamp(0, 3).to(torch.uint8)
                packed = self._pack_lowbit_codes(quantized.view(rotated.shape[0], -1), 2)
                results = {
                    'codes': packed.cpu().numpy(),
                    'scales': scales.cpu().numpy(),
                    'offsets': min_vals.cpu().numpy(),
                    'norms_sq': norms_sq.cpu().numpy(),
                    'code_sums': quantized.float().sum(dim=2).cpu().numpy(),
                    'group_size': group_size,
                    'meta_layout': 'grouped',
                }

        elif self.config.num_bits == 4:
            # 4-bit Quantization: [0, 15]
            # Optional group-wise scaling materially improves fidelity on real corpora.
            group_size = self.config.group_size or self.effective_dim
            if group_size == self.effective_dim:
                min_vals = rotated.min(dim=1).values
                max_vals = rotated.max(dim=1).values
                ranges = max_vals - min_vals
                ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)

                deltas = ranges / 15.0
                offsets = min_vals
                quantized = ((rotated - offsets.unsqueeze(1)) / deltas.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
                packed = self._pack_lowbit_codes(quantized, 4)

                results = {
                    'codes': packed.cpu().numpy(),
                    'scales': deltas.cpu().numpy(),
                    'offsets': offsets.cpu().numpy(),
                    'norms_sq': norms_sq.cpu().numpy(),
                    'code_sums': quantized.float().sum(dim=1).cpu().numpy(),
                    'meta_layout': 'scalar',
                }
            else:
                n_groups = self.effective_dim // group_size
                grouped = rotated.view(rotated.shape[0], n_groups, group_size)
                min_vals = grouped.min(dim=2).values
                max_vals = grouped.max(dim=2).values
                ranges = max_vals - min_vals
                ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)

                deltas = ranges / 15.0
                quantized = ((grouped - min_vals.unsqueeze(-1)) / deltas.unsqueeze(-1)).round().clamp(0, 15).byte()
                packed = self._pack_lowbit_codes(quantized.view(rotated.shape[0], -1), 4)

                results = {
                    'codes': packed.cpu().numpy(),
                    'scales': deltas.cpu().numpy(),
                    'offsets': min_vals.cpu().numpy(),
                    'norms_sq': norms_sq.cpu().numpy(),
                    'code_sums': quantized.float().sum(dim=2).cpu().numpy(),
                    'group_size': group_size,
                    'meta_layout': 'grouped',
                }

        else:
            # 8-bit Quantization
            min_vals = rotated.min(dim=1).values

            # ... (rest of 8-bit logic) ...
            max_vals = rotated.max(dim=1).values
            ranges = max_vals - min_vals
            ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)

            deltas = ranges / 255.0
            offsets = min_vals

            deltas_col = deltas.unsqueeze(1)
            offsets_col = offsets.unsqueeze(1)

            quantized = (rotated - offsets_col) / deltas_col
            quantized = quantized.round().clamp(0, 255).byte()

            results = {
                'codes': quantized.cpu().numpy(),
                'scales': deltas.cpu().numpy(),
                'offsets': offsets.cpu().numpy(),
                'norms_sq': norms_sq.cpu().numpy(),
                'code_sums': quantized.float().sum(dim=1).cpu().numpy()
            }

        if store:
             self._codes = results['codes']
             self._scales = results['scales']
             self._offsets = results['offsets']
             self._norms_sq = results['norms_sq']
             self._code_sums = results.get('code_sums')
             logger.info(f"Stored {len(x)} vectors in index (bits={self.config.num_bits})")

        return results

    def decode(self,
               codes: Union[np.ndarray, torch.Tensor],
               scales: Union[np.ndarray, torch.Tensor],
               offsets: Union[np.ndarray, torch.Tensor]
               ) -> torch.Tensor:
        """
        Reconstruct vectors from quantized codes.

        Args:
            codes: Quantized codes (N, D_packed)
            scales: Scales (N, 1) or (N,)
            offsets: Offsets (N, 1) or (N,)

        Returns:
            Reconstructed vectors (N, D) - float32
        """
        device = 'cpu'
        if isinstance(codes, np.ndarray):
            codes = torch.from_numpy(codes)
        if isinstance(scales, np.ndarray):
            scales = torch.from_numpy(scales)
        if isinstance(offsets, np.ndarray):
            offsets = torch.from_numpy(offsets)

        codes = codes.to(device)
        scales = scales.to(device).float()
        offsets = offsets.to(device).float()

        scales = self._reshape_meta(scales, batch_size=codes.shape[0], effective_dim=self.effective_dim).to(device)
        offsets = self._reshape_meta(offsets, batch_size=codes.shape[0], effective_dim=self.effective_dim).to(device)

        N = codes.shape[0]

        if self.config.num_bits == 8:
            # 8-bit: Direct mapping
            # x ~= (code * scale) + offset
            rotated_recon = (codes.float() * scales) + offsets

        elif self.config.num_bits == 4:
            # 4-bit: Unpack
            # Byte = (High << 4) | Low
            high = (codes >> 4) & 0x0F
            low = codes & 0x0F

            # Stack [High, Low] -> (N, D/2, 2)
            unpacked = torch.stack([high, low], dim=2)
            # Flatten -> (N, D)
            unpacked = unpacked.view(N, -1)[:, :self.effective_dim]
            if scales.shape[1] == 1:
                rotated_recon = (unpacked.float() * scales) + offsets
            else:
                group_size = self._group_size(scales)
                unpacked = unpacked.view(N, scales.shape[1], group_size)
                rotated_recon = (
                    unpacked.float() * scales.unsqueeze(-1) + offsets.unsqueeze(-1)
                ).view(N, -1)

        elif self.config.num_bits == 2:
            unpacked = torch.stack(
                [
                    (codes >> 6) & 0x03,
                    (codes >> 4) & 0x03,
                    (codes >> 2) & 0x03,
                    codes & 0x03,
                ],
                dim=2,
            ).view(N, -1)[:, :self.effective_dim]
            if scales.shape[1] == 1:
                rotated_recon = (unpacked.float() * scales) + offsets
            else:
                group_size = self._group_size(scales)
                unpacked = unpacked.view(N, scales.shape[1], group_size)
                rotated_recon = (
                    unpacked.float() * scales.unsqueeze(-1) + offsets.unsqueeze(-1)
                ).view(N, -1)

        elif self.config.num_bits == 1:
            bit_offsets = torch.arange(7, -1, -1, device=codes.device, dtype=torch.uint8)
            unpacked = ((codes.unsqueeze(-1) >> bit_offsets) & 0x01).reshape(N, -1)[:, :self.effective_dim]
            rotated_recon = (unpacked.float() * scales) + offsets

        else:
            raise NotImplementedError(f"Decoding {self.config.num_bits}-bit not implemented")

        # Inverse Rotation?
        # Rotation is P_k H D_k ...
        # Inverse is ... D_k H P_k^T (since H is symmetric unitary-ish, D is symmetric)
        # Actually H is symmetric. H^-1 = H (normalized).
        # We need to apply inverse steps in REVERSE order.

        # Check Rotator param storage
        # It stores signs and permutations.
        # We assume self.rotator is the SAME one used for encoding (seed match).

        x_recon = rotated_recon

        # Reverse loop
        for i in range(self.rotator.num_rounds - 1, -1, -1):
            # 3. Inverse Permutation
            if i < self.rotator.num_rounds - 1:
                # Permutation P. Inverse is P argsort?
                # P maps src -> dst. y[j] = x[P[j]].
                # Inverse: z[P[j]] = y[j].
                # Or just standard inv perm.
                perm = self.rotator.permutations[i].to(device)
                inv_perm = torch.argsort(perm)
                x_recon = x_recon[:, inv_perm]

            # 2. Blocked FWHT (Self-inverse)
            # x_recon is (N, D_padded)
            reshaped = x_recon.view(N, -1, self.rotator.block_size)
            transformed = self.rotator._fwht(reshaped)
            x_recon = transformed.view(N, -1)

            # 1. Inverse Signs (Self-inverse since {-1, 1})
            signs = self.rotator.signs[i].to(device)
            x_recon = x_recon * signs

        # Trim padding if needed
        if self.config.dim < x_recon.shape[1]:
             x_recon = x_recon[:, :self.config.dim]

        return x_recon

    def _symmetric_distance(self,
                            q_codes: torch.Tensor, q_scale: torch.Tensor, q_offset: torch.Tensor, q_norm_sq: torch.Tensor,
                            doc_codes: torch.Tensor, doc_scale: torch.Tensor, doc_offset: torch.Tensor, doc_norm_sq: torch.Tensor
                            ) -> torch.Tensor:
        """Compute estimated distance between quantized vectors."""
        # Check if binary (packed)
        # Note: q_codes logic below assumes unpacked float for 8-bit.
        # This implementation requires refactoring if we want unified method.
        # For simplicity, search() will handle dispatch for 1-bit vs 8-bit.
        raise NotImplementedError("Use search() which dispatches")

    def search(self, query: Union[np.ndarray, torch.Tensor], top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using quantized index.
        """
        if self._codes is None:
            raise RuntimeError("Index empty")

        scores = self._asymmetric_inner_products(query, self._codes, self._scales, self._offsets)

        if scores.shape[0] == 1:
            scores = scores.squeeze(0)
            topk = torch.topk(scores, k=min(top_k, len(scores)), dim=0, largest=True)
            return topk.indices.cpu().numpy(), topk.values.cpu().numpy()
        topk = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1, largest=True)
        return topk.indices.cpu().numpy(), topk.values.cpu().numpy()


__all__ = ['RotationalQuantizer', 'RoQConfig', 'FastWalshHadamard']
