"""
Triangular (Query-Conditioned Reverse) MaxSim kernels.

Given query token embeddings Q (S x H), context token embeddings C (T x H), and
response token embeddings R (U x H), all L2-normalized, this module computes:

    a[j]     = max_i  Q[i] . C[j]                       (query relevance of c_j)
    g[t]     = max_j  min(R[t] . C[j], a[j])            (Triangular groundedness)
    e[t]     = max_i  R[t] . Q[i]                       (prompt-echo channel)
    jstar[t] = argmax_j min(R[t] . C[j], a[j])          (evidence attribution)
    u[i]     = max_t  min(Q[i] . R[t], g[t])            (grounded coverage)

The min() acts as an AND gate: a response token only earns credit if it matches a
context token that itself matters to the query. See the Reverse MaxSim
falsification experiment under research/triangular_maxsim/ for a sibling test.

The production MaxSim path in triton_maxsim.py is untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _next_pow2(n: int) -> int:
    v = max(int(n), 1)
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1


# ---------------------------------------------------------------------------
# Pass 1: a[j] = max_i Q[i] . C[j]
# ---------------------------------------------------------------------------

@triton.jit
def _pass_a_kernel(
    Q_PTR, C_PTR, A_PTR,
    Q_MASK_PTR, C_MASK_PTR,
    NUM_Q: tl.constexpr,
    NUM_C: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_Q_MASK: tl.constexpr,
    HAS_C_MASK: tl.constexpr,
):
    """One program per BLOCK_C-sized chunk of context tokens."""
    pid = tl.program_id(0)
    c_offsets = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    in_range = c_offsets < NUM_C

    dim_offsets = tl.arange(0, DIM)

    # (BLOCK_C, DIM)
    c_ptrs = C_PTR + c_offsets[:, None] * DIM + dim_offsets[None, :]
    c_block = tl.load(c_ptrs, mask=in_range[:, None], other=0.0).to(tl.float32)

    if HAS_C_MASK:
        c_valid_raw = tl.load(C_MASK_PTR + c_offsets, mask=in_range, other=0.0)
        c_valid = c_valid_raw > 0.5
    else:
        c_valid = in_range

    a_max = tl.full((BLOCK_C,), -1.0e30, dtype=tl.float32)

    for q_idx in range(0, NUM_Q):
        if HAS_Q_MASK:
            q_valid_scalar = tl.load(Q_MASK_PTR + q_idx) > 0.5
        else:
            q_valid_scalar = True

        q_ptrs = Q_PTR + q_idx * DIM + dim_offsets
        q_vec = tl.load(q_ptrs).to(tl.float32)  # (DIM,)

        # sim per c-token: sum over dim of c_block * q_vec
        sim = tl.sum(c_block * q_vec[None, :], axis=1)  # (BLOCK_C,)
        sim = tl.where(q_valid_scalar, sim, -1.0e30)
        a_max = tl.maximum(a_max, sim)

    a_max = tl.where(c_valid, a_max, -1.0e30)
    tl.store(A_PTR + c_offsets, a_max, mask=in_range)


# ---------------------------------------------------------------------------
# Pass 2: g[t], e[t], jstar[t]
# ---------------------------------------------------------------------------

@triton.jit
def _pass_g_kernel(
    R_PTR, Q_PTR, C_PTR, A_PTR,
    G_PTR, E_PTR, JSTAR_PTR,
    R_MASK_PTR, Q_MASK_PTR, C_MASK_PTR,
    NUM_R: tl.constexpr,
    NUM_Q: tl.constexpr,
    NUM_C: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HAS_R_MASK: tl.constexpr,
    HAS_Q_MASK: tl.constexpr,
    HAS_C_MASK: tl.constexpr,
):
    """One program per response token t."""
    t = tl.program_id(0)
    if t >= NUM_R:
        return

    if HAS_R_MASK:
        r_valid = tl.load(R_MASK_PTR + t) > 0.5
    else:
        r_valid = True

    dim_offsets = tl.arange(0, DIM)
    r_ptrs = R_PTR + t * DIM + dim_offsets
    r_vec = tl.load(r_ptrs).to(tl.float32)  # (DIM,)

    # ----- echo: e[t] = max_i r . q_i -----
    e_max = tl.full((), -1.0e30, dtype=tl.float32)

    for q_start in range(0, NUM_Q, BLOCK_Q):
        q_offsets = q_start + tl.arange(0, BLOCK_Q)
        q_in_range = q_offsets < NUM_Q

        q_ptrs = Q_PTR + q_offsets[:, None] * DIM + dim_offsets[None, :]
        q_block = tl.load(q_ptrs, mask=q_in_range[:, None], other=0.0).to(tl.float32)

        sim_q = tl.sum(q_block * r_vec[None, :], axis=1)  # (BLOCK_Q,)

        if HAS_Q_MASK:
            q_valid_block = tl.load(Q_MASK_PTR + q_offsets, mask=q_in_range, other=0.0) > 0.5
            sim_q = tl.where(q_valid_block, sim_q, -1.0e30)
        sim_q = tl.where(q_in_range, sim_q, -1.0e30)

        block_max = tl.max(sim_q, axis=0)
        e_max = tl.maximum(e_max, block_max)

    # ----- groundedness with argmax over context tokens -----
    g_max = tl.full((), -1.0e30, dtype=tl.float32)
    j_best = tl.zeros((), dtype=tl.int32)

    for c_start in range(0, NUM_C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_in_range = c_offsets < NUM_C

        c_ptrs = C_PTR + c_offsets[:, None] * DIM + dim_offsets[None, :]
        c_block = tl.load(c_ptrs, mask=c_in_range[:, None], other=0.0).to(tl.float32)

        a_block = tl.load(A_PTR + c_offsets, mask=c_in_range, other=-1.0e30).to(tl.float32)

        sim_c = tl.sum(c_block * r_vec[None, :], axis=1)  # (BLOCK_C,)
        gated = tl.minimum(sim_c, a_block)

        if HAS_C_MASK:
            c_valid_block = tl.load(C_MASK_PTR + c_offsets, mask=c_in_range, other=0.0) > 0.5
            gated = tl.where(c_valid_block, gated, -1.0e30)
        gated = tl.where(c_in_range, gated, -1.0e30)

        block_max = tl.max(gated, axis=0)
        block_arg = tl.argmax(gated, axis=0).to(tl.int32)

        new_better = block_max > g_max
        g_max = tl.where(new_better, block_max, g_max)
        j_best = tl.where(new_better, c_start + block_arg, j_best)

    if not r_valid:
        g_max = tl.zeros((), dtype=tl.float32)
        e_max = tl.zeros((), dtype=tl.float32)

    tl.store(G_PTR + t, g_max)
    tl.store(E_PTR + t, e_max)
    tl.store(JSTAR_PTR + t, j_best)


# ---------------------------------------------------------------------------
# Pass 3: u[i] = max_t min(Q[i] . R[t], g[t])
# ---------------------------------------------------------------------------

@triton.jit
def _pass_u_kernel(
    Q_PTR, R_PTR, G_PTR, U_PTR,
    Q_MASK_PTR, R_MASK_PTR,
    NUM_Q: tl.constexpr,
    NUM_R: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_Q_MASK: tl.constexpr,
    HAS_R_MASK: tl.constexpr,
):
    """One program per query token i."""
    i = tl.program_id(0)
    if i >= NUM_Q:
        return

    if HAS_Q_MASK:
        q_valid = tl.load(Q_MASK_PTR + i) > 0.5
    else:
        q_valid = True

    dim_offsets = tl.arange(0, DIM)
    q_ptrs = Q_PTR + i * DIM + dim_offsets
    q_vec = tl.load(q_ptrs).to(tl.float32)

    u_max = tl.full((), -1.0e30, dtype=tl.float32)

    for r_start in range(0, NUM_R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        r_in_range = r_offsets < NUM_R

        r_ptrs = R_PTR + r_offsets[:, None] * DIM + dim_offsets[None, :]
        r_block = tl.load(r_ptrs, mask=r_in_range[:, None], other=0.0).to(tl.float32)
        g_block = tl.load(G_PTR + r_offsets, mask=r_in_range, other=-1.0e30).to(tl.float32)

        sim_r = tl.sum(r_block * q_vec[None, :], axis=1)  # (BLOCK_R,)
        gated = tl.minimum(sim_r, g_block)

        if HAS_R_MASK:
            r_valid_block = tl.load(R_MASK_PTR + r_offsets, mask=r_in_range, other=0.0) > 0.5
            gated = tl.where(r_valid_block, gated, -1.0e30)
        gated = tl.where(r_in_range, gated, -1.0e30)

        block_max = tl.max(gated, axis=0)
        u_max = tl.maximum(u_max, block_max)

    if not q_valid:
        u_max = tl.zeros((), dtype=tl.float32)

    tl.store(U_PTR + i, u_max)


# ---------------------------------------------------------------------------
# Python-side reference + wrapper
# ---------------------------------------------------------------------------

@dataclass
class TriangularResult:
    """Per-token signals for one (Q, C, R) triple."""
    a: torch.Tensor       # (T,)   query relevance per context token
    g: torch.Tensor       # (U,)   triangular groundedness per response token
    e: torch.Tensor       # (U,)   prompt-echo per response token
    u: torch.Tensor       # (S,)   grounded coverage per query token
    jstar: torch.Tensor   # (U,)   argmax context index per response token


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def triangular_maxsim_reference(
    Q: torch.Tensor,
    C: torch.Tensor,
    R: torch.Tensor,
    Q_mask: Optional[torch.Tensor] = None,
    C_mask: Optional[torch.Tensor] = None,
    R_mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> TriangularResult:
    """Pure-PyTorch reference implementation. Matches the kernel exactly."""
    Q = Q.float()
    C = C.float()
    R = R.float()
    if normalize:
        Q = _normalize(Q)
        C = _normalize(C)
        R = _normalize(R)

    S = Q.shape[0]
    T = C.shape[0]
    U = R.shape[0]

    NEG = torch.tensor(-1.0e30, device=Q.device)

    if Q_mask is None:
        Q_mask = torch.ones(S, device=Q.device)
    if C_mask is None:
        C_mask = torch.ones(T, device=C.device)
    if R_mask is None:
        R_mask = torch.ones(U, device=R.device)

    Q_mask = Q_mask.bool()
    C_mask = C_mask.bool()
    R_mask = R_mask.bool()

    # sim_QC: (S, T)
    sim_QC = Q @ C.T
    sim_QC = torch.where(Q_mask[:, None], sim_QC, NEG)
    a = sim_QC.max(dim=0).values  # (T,)
    a = torch.where(C_mask, a, NEG)

    # sim_RC: (U, T)
    sim_RC = R @ C.T
    gated = torch.minimum(sim_RC, a[None, :])
    gated = torch.where(C_mask[None, :], gated, NEG)
    g_full = gated.max(dim=1)
    g = g_full.values
    jstar = g_full.indices.to(torch.int32)
    g = torch.where(R_mask, g, torch.zeros_like(g))

    # echo: (U,)
    sim_RQ = R @ Q.T
    sim_RQ = torch.where(Q_mask[None, :], sim_RQ, NEG)
    e = sim_RQ.max(dim=1).values
    e = torch.where(R_mask, e, torch.zeros_like(e))

    # grounded coverage u[i]: (S,)
    sim_QR = Q @ R.T  # (S, U)
    gated_u = torch.minimum(sim_QR, g[None, :])
    gated_u = torch.where(R_mask[None, :], gated_u, NEG)
    u = gated_u.max(dim=1).values
    u = torch.where(Q_mask, u, torch.zeros_like(u))

    return TriangularResult(a=a, g=g, e=e, u=u, jstar=jstar)


def triangular_maxsim(
    Q: torch.Tensor,
    C: torch.Tensor,
    R: torch.Tensor,
    Q_mask: Optional[torch.Tensor] = None,
    C_mask: Optional[torch.Tensor] = None,
    R_mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
    use_kernel: Optional[bool] = None,
    block_c: int = 32,
    block_q: int = 16,
    block_r: int = 16,
) -> TriangularResult:
    """Triton-accelerated triangular MaxSim. Falls back to the PyTorch reference
    on CPU or when ``use_kernel`` is False.

    Inputs are 2D tensors:
        Q: (S, H), C: (T, H), R: (U, H)
    Optional 1D masks (1 for real tokens, 0 for padding).
    Returns a :class:`TriangularResult` with all signals on the input device.
    """
    if Q.dim() != 2 or C.dim() != 2 or R.dim() != 2:
        raise ValueError("Q, C, R must be 2D (num_tokens, dim)")
    if Q.shape[1] != C.shape[1] or Q.shape[1] != R.shape[1]:
        raise ValueError(
            f"embedding dims must match: Q={Q.shape[1]} C={C.shape[1]} R={R.shape[1]}"
        )

    if use_kernel is None:
        use_kernel = Q.is_cuda and torch.cuda.is_available()

    if not use_kernel:
        return triangular_maxsim_reference(
            Q, C, R, Q_mask, C_mask, R_mask, normalize=normalize
        )

    device = Q.device
    if normalize:
        Q = _normalize(Q.float())
        C = _normalize(C.float())
        R = _normalize(R.float())
    else:
        Q = Q.float()
        C = C.float()
        R = R.float()

    H_raw = Q.shape[1]
    H = _next_pow2(max(H_raw, 16))
    if H != H_raw:
        pad = (0, H - H_raw)
        Q = torch.nn.functional.pad(Q, pad)
        C = torch.nn.functional.pad(C, pad)
        R = torch.nn.functional.pad(R, pad)

    Q = Q.contiguous()
    C = C.contiguous()
    R = R.contiguous()

    S = Q.shape[0]
    T = C.shape[0]
    U = R.shape[0]

    HAS_Q_MASK = Q_mask is not None
    HAS_C_MASK = C_mask is not None
    HAS_R_MASK = R_mask is not None

    def _prep_mask(m: Optional[torch.Tensor], n: int) -> torch.Tensor:
        if m is None:
            return torch.empty(0, device=device, dtype=torch.float32)
        m = m.to(device=device, dtype=torch.float32).contiguous()
        if m.numel() != n:
            raise ValueError(f"mask length {m.numel()} != expected {n}")
        return m

    Q_mask_t = _prep_mask(Q_mask, S)
    C_mask_t = _prep_mask(C_mask, T)
    R_mask_t = _prep_mask(R_mask, U)

    a = torch.empty(T, device=device, dtype=torch.float32)
    g = torch.empty(U, device=device, dtype=torch.float32)
    e = torch.empty(U, device=device, dtype=torch.float32)
    jstar = torch.empty(U, device=device, dtype=torch.int32)
    u = torch.empty(S, device=device, dtype=torch.float32)

    grid_a = (triton.cdiv(T, block_c),)
    _pass_a_kernel[grid_a](
        Q, C, a,
        Q_mask_t if HAS_Q_MASK else Q,  # placeholder pointer (unused when mask flag is 0)
        C_mask_t if HAS_C_MASK else C,
        NUM_Q=S, NUM_C=T, DIM=H,
        BLOCK_C=block_c,
        HAS_Q_MASK=HAS_Q_MASK,
        HAS_C_MASK=HAS_C_MASK,
    )

    grid_g = (U,)
    _pass_g_kernel[grid_g](
        R, Q, C, a,
        g, e, jstar,
        R_mask_t if HAS_R_MASK else R,
        Q_mask_t if HAS_Q_MASK else Q,
        C_mask_t if HAS_C_MASK else C,
        NUM_R=U, NUM_Q=S, NUM_C=T, DIM=H,
        BLOCK_C=block_c, BLOCK_Q=block_q,
        HAS_R_MASK=HAS_R_MASK,
        HAS_Q_MASK=HAS_Q_MASK,
        HAS_C_MASK=HAS_C_MASK,
    )

    grid_u = (S,)
    _pass_u_kernel[grid_u](
        Q, R, g, u,
        Q_mask_t if HAS_Q_MASK else Q,
        R_mask_t if HAS_R_MASK else R,
        NUM_Q=S, NUM_R=U, DIM=H,
        BLOCK_R=block_r,
        HAS_Q_MASK=HAS_Q_MASK,
        HAS_R_MASK=HAS_R_MASK,
    )

    # zero out invalid response/query positions to match the reference
    if HAS_R_MASK:
        r_valid = R_mask_t > 0.5
        g = torch.where(r_valid, g, torch.zeros_like(g))
        e = torch.where(r_valid, e, torch.zeros_like(e))
    if HAS_Q_MASK:
        q_valid = Q_mask_t > 0.5
        u = torch.where(q_valid, u, torch.zeros_like(u))

    return TriangularResult(a=a, g=g, e=e, u=u, jstar=jstar)


# ---------------------------------------------------------------------------
# Aggregate scoring helpers
# ---------------------------------------------------------------------------

def weighted_groundedness(
    g: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> float:
    """G(R|Q,C) = sum_t w_t g_t / sum_t w_t.
    If weights is None, falls back to a uniform mean.
    """
    if weights is None:
        return float(g.mean().item())
    w = weights.to(g.device, dtype=g.dtype)
    denom = w.sum().clamp_min(1e-9)
    return float((g * w).sum().item() / denom.item())


def grounded_coverage(u: torch.Tensor, q_mask: Optional[torch.Tensor] = None) -> float:
    """Mean of u[i] over valid query tokens."""
    if q_mask is None:
        return float(u.mean().item())
    m = q_mask.to(u.device, dtype=u.dtype)
    denom = m.sum().clamp_min(1e-9)
    return float((u * m).sum().item() / denom.item())


def naive_reverse_maxsim_qc(
    Q: torch.Tensor, C: torch.Tensor, R: torch.Tensor,
    Q_mask: Optional[torch.Tensor] = None,
    C_mask: Optional[torch.Tensor] = None,
    R_mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> tuple[float, torch.Tensor]:
    """Strawman: g_naive[t] = max_{x in Q union C} R[t] . x. Returns (G, g_per_token)."""
    Q = Q.float(); C = C.float(); R = R.float()
    if normalize:
        Q = _normalize(Q); C = _normalize(C); R = _normalize(R)
    QC = torch.cat([Q, C], dim=0)
    if Q_mask is None:
        Q_mask = torch.ones(Q.shape[0], device=Q.device)
    if C_mask is None:
        C_mask = torch.ones(C.shape[0], device=C.device)
    QC_mask = torch.cat([Q_mask.to(Q.device), C_mask.to(Q.device)], dim=0).bool()

    sim = R @ QC.T
    sim = torch.where(QC_mask[None, :], sim, torch.tensor(-1.0e30, device=sim.device))
    g_per = sim.max(dim=1).values
    if R_mask is not None:
        g_per = torch.where(R_mask.bool().to(g_per.device), g_per, torch.zeros_like(g_per))
    return weighted_groundedness(g_per, weights), g_per


def reverse_maxsim_rc(
    C: torch.Tensor, R: torch.Tensor,
    C_mask: Optional[torch.Tensor] = None,
    R_mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> tuple[float, torch.Tensor]:
    """Reverse MaxSim against just C (no query conditioning)."""
    C = C.float(); R = R.float()
    if normalize:
        C = _normalize(C); R = _normalize(R)
    sim = R @ C.T
    if C_mask is not None:
        sim = torch.where(C_mask.bool().to(sim.device)[None, :], sim,
                          torch.tensor(-1.0e30, device=sim.device))
    g_per = sim.max(dim=1).values
    if R_mask is not None:
        g_per = torch.where(R_mask.bool().to(g_per.device), g_per, torch.zeros_like(g_per))
    return weighted_groundedness(g_per, weights), g_per
