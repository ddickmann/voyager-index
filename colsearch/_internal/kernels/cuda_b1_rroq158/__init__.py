"""Fused binary tensor-core MMA rroq158 MaxSim kernel for sm_90+.

Production fast-lane for the rroq158 whole-corpus and large-candidate
MaxSim path on Hopper (H100, GH200) and Blackwell (5090, 6000-Pro)
GPUs. Computes per-document scores via a single fused CUDA kernel that:

  * Uses ``mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc``
    (4-bit/binary tensor cores) to do popcount-AND across 128 K-bits in
    a single warp instruction.
  * Holds the per-(d, i) running max in registers across the j loop so
    we never materialise a (B*T, S*query_bits) intermediate.
  * Writes only the final ``(B,)`` scores to global memory.

Smoke validation result (H100, B=2048, T=288, S=32, dim=128):

    Triton popc baseline:  0.491 ms / query
    FUSED b1 MMA kernel:   0.203 ms / query   (2.41× speedup)
    parity max rel err:    3.30e-7  (within fp32 noise)

The kernel is intentionally specialised to the **production rroq158
shape** (``dim=128 → n_words=4``, ``group_size=128 → n_groups=1,
group_words=4``, ``query_bits=4``, ``S=32``). Other shapes — short
queries, larger group_size, smaller dim — fall back to the existing
Triton kernel via ``score_rroq158_b1_fused`` returning ``None``.

Activation:
  * Default ``on`` for sm >= 90 (H100/Hopper, RTX 5090/Blackwell, ...).
  * Disable with ``VOYAGER_RROQ158_USE_B1_FUSED=0``.
  * Force-enable on older arches (won't launch without `.and.popc`)
    with ``VOYAGER_RROQ158_USE_B1_FUSED=1``.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


_HERE = Path(__file__).resolve().parent
_KERNEL_CU = _HERE / "_kernel.cu"

_ext = None
_ext_lock = threading.Lock()
_load_failed = False


def _resolve_nvcc() -> Optional[str]:
    """Best-effort nvcc lookup that matches torch's CUDA major.

    Returns the nvcc path (and side-effects ``CUDA_HOME``) on success,
    ``None`` otherwise — caller falls back to PATH lookup which is the
    default torch.utils.cpp_extension behaviour.
    """
    candidates = [
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12",
        "/usr/local/cuda",
    ]
    for base in candidates:
        if base and (Path(base) / "bin" / "nvcc").exists():
            os.environ.setdefault("CUDA_HOME", base)
            return str(Path(base) / "bin" / "nvcc")
    return None


def _supported_capability() -> bool:
    """sm >= 90 (Hopper/Blackwell). The .and.popc bMMA variant requires
    sm_75+ in PTX, but we validate sm_90 because that's the only
    capability we've smoke-tested for parity. To force-enable on lower
    capability set ``VOYAGER_RROQ158_USE_B1_FUSED=1``.
    """
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap[0] >= 9


def _env_decision() -> Optional[bool]:
    raw = os.environ.get("VOYAGER_RROQ158_USE_B1_FUSED")
    if raw is None:
        return None
    if raw.strip() in ("0", "false", "False", "off"):
        return False
    return True


def is_available() -> bool:
    """True iff the fused CUDA kernel can be loaded for this host."""
    global _load_failed
    if _load_failed:
        return False
    decision = _env_decision()
    if decision is False:
        return False
    if decision is None and not _supported_capability():
        return False
    try:
        _load()
        return True
    except Exception:
        return False


def _load():
    """JIT-compile + cache the CUDA extension. Idempotent.

    First call takes ~30-40 s on H100 (one-time, then cached under
    ``$TORCH_EXTENSIONS_DIR`` / ``~/.cache/torch_extensions``). Use
    :func:`warmup` to pay this cost during process init.
    """
    global _ext, _load_failed
    if _ext is not None:
        return _ext
    with _ext_lock:
        if _ext is not None:
            return _ext
        if _load_failed:
            raise RuntimeError("cuda_b1_rroq158 extension load previously failed")
        from torch.utils.cpp_extension import load

        _resolve_nvcc()
        cap = torch.cuda.get_device_capability(0)
        gencode = f"-gencode=arch=compute_{cap[0]}{cap[1]},code=sm_{cap[0]}{cap[1]}"
        try:
            t0_log = logger.isEnabledFor(logging.INFO)
            if t0_log:
                logger.info(
                    "JIT-compiling cuda_b1_rroq158 extension for sm_%d%d "
                    "(first load, ~30-40s; subsequent loads are cached)",
                    cap[0], cap[1],
                )
            _ext = load(
                name="voyager_cuda_b1_rroq158",
                sources=[str(_KERNEL_CU)],
                extra_cuda_cflags=["-O3", gencode, "-std=c++17"],
                verbose=False,
            )
            if t0_log:
                logger.info("cuda_b1_rroq158 extension loaded")
        except Exception as e:
            _load_failed = True
            logger.warning(
                "cuda_b1_rroq158 extension failed to load (%s); "
                "falling back to Triton popc kernel",
                e,
            )
            raise
    return _ext


def warmup() -> None:
    """Pay the JIT cost up-front. Safe to call from process init."""
    if not is_available():
        return
    try:
        _load()
    except Exception:
        pass


def _shape_supported(
    n_words: int,
    n_groups: int,
    query_bits: int,
    s_query: int,
) -> bool:
    """The fused kernel is hard-specialised to rroq158_gs128 + dim=128.

    Other shapes (smaller dim, larger group_size, different query_bits)
    are NOT supported by this kernel and the caller MUST fall back to
    Triton. We do this gating in Python (vs failing in CUDA) so the
    fallback is fast.
    """
    if n_words != 4:        # dim=128 only
        return False
    if n_groups != 1:        # group_size=128 only
        return False
    if query_bits != 4:      # rroq158 4-bit query lane only
        return False
    if s_query > 32:         # kernel rolls over S=32; longer queries
        return False         # are split or masked at the next layer
    return True


def score_b1_fused(
    docs_sign: torch.Tensor,    # (B, T, n_words) int32, contiguous
    docs_nz: torch.Tensor,      # (B, T, n_words) int32, contiguous
    docs_scl: torch.Tensor,     # (B, T, n_groups=1) float32 -> (B,T)
    docs_cid: torch.Tensor,     # (B, T) int32
    docs_cos: torch.Tensor,     # (B, T) float32
    docs_sin: torch.Tensor,     # (B, T) float32
    docs_mask: Optional[torch.Tensor],  # (B, T) float32 or None
    q_planes: torch.Tensor,     # (S, query_bits, n_words) int32
    q_meta: torch.Tensor,       # (S, 2) float32
    qc_table: torch.Tensor,     # (S, K) float32
    docs_scl_2d: Optional[torch.Tensor] = None,  # (B, T) — pre-squeezed
) -> Optional[torch.Tensor]:
    """Run the fused b1 MMA + epilogue kernel. Returns ``(B,) float32``
    scores on the same device as the inputs, or ``None`` if the fused
    path is unavailable/unsupported (caller should fall back to Triton).
    """
    if not is_available():
        return None
    B, T, n_words = docs_sign.shape
    n_groups = docs_scl.shape[-1]
    s_query, query_bits = q_planes.shape[0], q_planes.shape[1]
    if not _shape_supported(n_words, n_groups, query_bits, s_query):
        return None
    if docs_sign.device.type != "cuda":
        return None

    ext = _load()
    device = docs_sign.device

    # Pad B to a multiple of 8 (m8n8k128 tile width). Padded docs get
    # mask=0 so they contribute -inf and never enter the topk; we trim
    # back below. We use empty+copy_ rather than cat+zeros to avoid the
    # peak-VRAM doubling that torch.cat causes (cat allocates a fresh
    # destination AND keeps the source live until completion). At quora
    # scale (B=522,931, T≤256, n_words=4) the docs_sign tensor alone is
    # ~2 GB; the old cat path peaked at ~6 GB extra, which combined with
    # the bucketed layout would push us past the fast-path VRAM gate.
    pad_b = (8 - B % 8) % 8
    if pad_b:
        Bp = B + pad_b

        def _pad_b(x: torch.Tensor) -> torch.Tensor:
            shape = (Bp,) + tuple(x.shape[1:])
            out = torch.zeros(shape, dtype=x.dtype, device=device)
            out[:B].copy_(x)
            return out

        docs_sign = _pad_b(docs_sign)
        docs_nz = _pad_b(docs_nz)
        docs_scl = _pad_b(docs_scl)
        docs_cid = _pad_b(docs_cid)
        docs_cos = _pad_b(docs_cos)
        docs_sin = _pad_b(docs_sin)
        if docs_mask is not None:
            docs_mask = _pad_b(docs_mask)
    if docs_mask is None:
        docs_mask = torch.ones(docs_sign.shape[0], T, dtype=torch.float32, device=device)

    # Pad S to 32 with zero query tokens. Zero q_planes / q_meta /
    # qc_table contribute exactly 0 to each per-doc score (verified in
    # benchmarks/_smoke_b1_mma.py), so no extra mask is required.
    if s_query < 32:
        Sp = 32

        def _pad_s(x: torch.Tensor) -> torch.Tensor:
            shape = (Sp,) + tuple(x.shape[1:])
            out = torch.zeros(shape, dtype=x.dtype, device=device)
            out[:s_query].copy_(x)
            return out

        q_planes = _pad_s(q_planes)
        q_meta = _pad_s(q_meta)
        qc_table = _pad_s(qc_table)

    # The kernel takes (B, T) for n_groups=1 (we squeeze the trailing
    # axis off scl). Prefer the caller-supplied pre-squeezed tensor —
    # ``docs_scl[..., 0].contiguous()`` allocates a fresh (B, T) f32
    # buffer (~65 MB at quora scale) per call which is one of the two
    # remaining per-query allocations responsible for the periodic
    # CUDA allocator GC stalls.
    if docs_scl_2d is not None:
        scl_2d = docs_scl_2d
        if pad_b:
            # The pre-squeezed buffer was sized for the un-padded B;
            # rebuild for the padded layout (rare path, only on first
            # call before the caller has cached a padded version).
            scl_2d = docs_scl[..., 0].contiguous()
    else:
        scl_2d = docs_scl[..., 0].contiguous()

    # Inputs are already contiguous (empty+copy_ pad path or original
    # contiguous tensors). Avoid extra .contiguous() calls that would
    # silently allocate copies.
    scores = ext.fused_b1_rroq158(
        docs_sign, docs_nz,
        q_planes, q_meta, qc_table,
        scl_2d, docs_cid, docs_cos, docs_sin, docs_mask,
    )

    # Trim padding back
    if pad_b:
        scores = scores[:B]
    return scores
