"""Smoke test: binary tensor-core MMA replacement for the rroq158 popcount path.

Goal: prove that PTX `mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc`
delivers a meaningful speedup over the current scalar-popcount Triton kernel
(``voyager_index._internal.kernels.triton_roq_rroq158.roq_maxsim_rroq158``)
*before* committing to wiring it into production.

What this script does:

  1. Builds a small CUDA C++ extension JIT-compiled via
     ``torch.utils.cpp_extension.load_inline`` that exposes ``mma_b1(A, B)``,
     a kernel that computes the popcount-AND matrix
       C[m, n] = popcount( A[m] AND B[n] )    over 128 K bits.
  2. Synthesises a representative rroq158 corpus + query at production shape
     (dim=128, n_words=4, query_bits=4, n_groups=1, group_words=4, S=32,
     T_max=288, B padded to a multiple of 8).
  3. Bit-exact validates the b1-MMA popcount matrix against a torch reference
     (uses ``torch.bitwise_and`` + ``__builtin_popcount`` via int8 tensor).
  4. Runs the existing Triton ``roq_maxsim_rroq158`` end-to-end for the
     baseline timing and the gold scores.
  5. Wires the b1-MMA outputs (m_sums, c_sums) into a small fp32 epilogue
     (pure pytorch ops) that reproduces the rroq158 score formula, and
     verifies the final scores match Triton within fp tolerance.
  6. Microbenches both paths over many iterations and reports speedup.

Output: a single block to stdout that the next decision step (wire-in or
not) reads.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- Resolve a CUDA toolchain that matches torch's CUDA major.
# torch is built against cu128 (CUDA 12.8). nvcc shipped with
# nvidia-cuda-nvcc-cu12 PyPI wheel works for our purpose.
def _find_nvcc() -> str:
    candidates = [
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12",
        "/usr/local/cuda",
    ]
    for base in candidates:
        if base and (Path(base) / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = base
            return str(Path(base) / "bin" / "nvcc")
    # Fallback: nvcc on PATH
    return "nvcc"


_find_nvcc()

from torch.utils.cpp_extension import load_inline  # noqa: E402

from voyager_index._internal.inference.quantization.rroq158 import (  # noqa: E402
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)
from voyager_index._internal.kernels.triton_roq_rroq158 import (  # noqa: E402
    roq_maxsim_rroq158,
)


# ─────────────────────────────────────────────────────────────────────
# CUDA b1 MMA kernel (JIT compiled)
# ─────────────────────────────────────────────────────────────────────

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// One warp = one 8x8 output tile.
//
// PTX m8n8k128.row.col.s32.b1.b1.s32.and.popc:
//   D[m][n] = sum over K=128 bits of popcount( A[m, k_bit] AND B[k_bit, n] )
//
// Per-thread fragment layout (warp of 32 lanes):
//   A is 8x128 row-major: lane t holds 1 .b32 = bits A[row=t/4, bits (t%4)*32..(t%4)*32+31]
//   B is 128x8 col-major: lane t holds 1 .b32 = bits B[bits (t%4)*32..(t%4)*32+31, col=t/4]
//   D is 8x8 row-major:   lane t holds 2 .s32 = D[row=t/4, col=2*(t%4)] and D[row=t/4, col=2*(t%4)+1]
//
// In memory:
//   A is laid out as (M_total, 4) row-major i32, so A[m, w] at A_base[m*4 + w].
//     For a tile, A_warp_base = A_base + tile_m * 8 * 4. Lane t loads
//     A_warp_base[(t/4)*4 + (t%4)] = A_warp_base[t]. Trivial coalesced load.
//   B is also (N_total, 4) row-major i32 in memory, but viewed as col-major
//     (K, N) for the mma. Same formula: lane t loads B_warp_base[t].
extern "C" __global__ void __launch_bounds__(32)
mma_b1_m8n8k128_kernel(
    const uint32_t* __restrict__ A,  // (M_total, 4) row-major
    const uint32_t* __restrict__ B,  // (N_total, 4) row-major
    int32_t* __restrict__ C,         // (M_total, N_total) row-major
    int M_total,
    int N_total) {
    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;
    int tid = threadIdx.x;

    const uint32_t* A_warp = A + tile_m * 8 * 4;
    const uint32_t* B_warp = B + tile_n * 8 * 4;

    uint32_t a_frag = A_warp[tid];
    uint32_t b_frag = B_warp[tid];

    int32_t c0 = 0, c1 = 0;

    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=r"(c0), "=r"(c1)
        : "r"(a_frag), "r"(b_frag), "r"(c0), "r"(c1)
    );

    int row = tile_m * 8 + tid / 4;
    int col = tile_n * 8 + (tid % 4) * 2;
    C[row * N_total + col + 0] = c0;
    C[row * N_total + col + 1] = c1;
}

torch::Tensor mma_b1(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A, B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kInt32 && B.dtype() == torch::kInt32,
                "A, B must be int32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "A, B must be 2D (M, n_words) and (N, n_words)");
    TORCH_CHECK(A.size(1) == 4 && B.size(1) == 4,
                "K_bits must be 128 (4 i32 words)");
    TORCH_CHECK(A.size(0) % 8 == 0, "M must be a multiple of 8");
    TORCH_CHECK(B.size(0) % 8 == 0, "N must be a multiple of 8");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "A, B must be contiguous");

    int M = A.size(0);
    int N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    dim3 grid(M / 8, N / 8);
    dim3 block(32);
    mma_b1_m8n8k128_kernel<<<grid, block>>>(
        reinterpret_cast<const uint32_t*>(A.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(B.data_ptr<int32_t>()),
        C.data_ptr<int32_t>(),
        M, N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

// ─────────────────────────────────────────────────────────────────────
// Fused b1 MMA + epilogue rroq158 kernel (no intermediate materialise)
// ─────────────────────────────────────────────────────────────────────
//
// 1 warp = 8 docs × all S query tokens × all T j-tokens.
// Per j: 32 mma instructions (4 i_tiles × 4 query_bits × 2 (m + c)).
// Per (i_tile, lane): epilogue computes est = cos·qc + sin·resi and
// updates the per-(d, i) max in registers.
// After the j-loop, warp-reduce max-sums across the 4 lanes per d to
// get score[d] and write directly to global. Only output is (B,)
// scores — no (B*T, S*qb) intermediate.
//
// Production shape only: dim=128 (n_words=4), query_bits=4, n_groups=1
// (group_words=4), S=32. Other shapes are handled by the existing
// Triton kernel.

extern "C" __global__ void __launch_bounds__(32)
fused_b1_rroq158_kernel(
    const uint32_t* __restrict__ pos_doc,    // (B, T, 4)
    const uint32_t* __restrict__ neg_doc,    // (B, T, 4)
    const uint32_t* __restrict__ q_planes,   // (S=32, qb=4, 4)
    const float* __restrict__ q_meta,        // (S, 2) [scale, offset]
    const float* __restrict__ qc_table,      // (S, K)
    const float* __restrict__ scl,           // (B, T) (n_groups=1)
    const int32_t* __restrict__ cid,         // (B, T)
    const float* __restrict__ cosn,          // (B, T)
    const float* __restrict__ sinn,          // (B, T)
    const float* __restrict__ mask,          // (B, T)
    float* __restrict__ scores,              // (B,)
    int B,
    int T,
    int K) {
    constexpr int S = 32;
    constexpr int QB = 4;
    constexpr int W = 4;
    constexpr int NUM_I_TILES = 4; // S / 8

    int d_tile = blockIdx.x;
    int tid = threadIdx.x;        // 0..31
    int row = tid >> 2;           // 0..7  doc-in-tile
    int word = tid & 3;           // 0..3  word-in-128bit
    int col_off = word * 2;       // 0, 2, 4, 6  start col of N=8 output
    int d = d_tile * 8 + row;
    if (d >= B) return;

    // Per-thread persistent max-sim across the j-loop. 8 per-thread
    // entries cover all 32 query tokens for `d`:
    //   thread (row, word) owns max_sim[d, i = it*8 + col_off + lane]
    //   for it ∈ 0..3, lane ∈ 0..1.
    float ms[NUM_I_TILES][2];
    #pragma unroll
    for (int it = 0; it < NUM_I_TILES; it++) {
        ms[it][0] = -INFINITY;
        ms[it][1] = -INFINITY;
    }

    const int row_base = (d * T) * W;        // pos[d, 0, 0]
    const int meta_off = d * T;              // scl/cid/cos/sin/mask[d, 0]

    for (int j = 0; j < T; j++) {
        // Per-thread mask. CANNOT use `continue` here: different docs in
        // the same warp have different mask patterns, and divergent
        // control flow would corrupt the warp-collective `__shfl_xor_sync`
        // and `mma.sync` instructions below. Instead we ballot — if every
        // doc in this warp's 8-doc tile has masked-out j, all 32 threads
        // agree to skip together (uniform branch is safe).
        float mask_j = mask[meta_off + j];
        unsigned int any_active = __ballot_sync(0xffffffff, mask_j > 0.0f);
        if (any_active == 0u) continue;

        // Load doc bits for this j (1 uint32 = 32 K-bits per thread,
        // 8 docs × 4 words = 32 fragments per warp = a single coalesced
        // 128-byte transaction per pos and per neg). For masked-out
        // docs the per-thread fragments still get loaded (then ignored
        // in the per-thread max-update at the bottom of the loop).
        uint32_t pos_frag = pos_doc[row_base + j * W + word];
        uint32_t neg_frag = neg_doc[row_base + j * W + word];

        // s_g = popcount(pos) - popcount(neg) summed over the 4 words.
        // Each thread popc's its own 32-bit slice; reduce within the
        // 4-thread row-group via XOR shuffle so every lane gets s_g.
        int s_g_int = __popc(pos_frag) - __popc(neg_frag);
        s_g_int += __shfl_xor_sync(0xffffffff, s_g_int, 1);
        s_g_int += __shfl_xor_sync(0xffffffff, s_g_int, 2);
        float s_g = (float)s_g_int;

        // Per-doc-token scalars (broadcast within the row-group; no
        // extra work — 4 threads coalesce to one cache-line load).
        float scl_j = scl[meta_off + j];
        int cid_j = cid[meta_off + j];
        float cos_j = cosn[meta_off + j];
        float sin_j = sinn[meta_off + j];

        // Per i_tile (4 tiles × 8 query tokens = S=32):
        #pragma unroll
        for (int it = 0; it < NUM_I_TILES; it++) {
            // m_pc[k] / c_pc[k] hold the popcount-AND result for this
            // tile across 4 query bit-planes (k ∈ 0..3). Two int32 per
            // thread per k = 8 ints per array = 32 ints in regs total.
            int m_pc[QB][2];
            int c_pc[QB][2];

            #pragma unroll
            for (int k = 0; k < QB; k++) {
                // q_planes[it*8 + col=tid/4, k, word=tid%4]
                int q_off = ((it * 8 + (tid >> 2)) * QB + k) * W + (tid & 3);
                uint32_t q_frag = q_planes[q_off];

                int32_t cm0 = 0, cm1 = 0;
                asm volatile(
                    "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1}, {%2}, {%3}, {%4, %5};"
                    : "=r"(cm0), "=r"(cm1)
                    : "r"(pos_frag), "r"(q_frag), "r"(cm0), "r"(cm1));
                m_pc[k][0] = cm0;
                m_pc[k][1] = cm1;

                int32_t cc0 = 0, cc1 = 0;
                asm volatile(
                    "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1}, {%2}, {%3}, {%4, %5};"
                    : "=r"(cc0), "=r"(cc1)
                    : "r"(neg_frag), "r"(q_frag), "r"(cc0), "r"(cc1));
                c_pc[k][0] = cc0;
                c_pc[k][1] = cc1;
            }

            // Epilogue per (i_tile, lane). Each thread owns 2 query
            // tokens (lane 0 and lane 1) for this i_tile.
            #pragma unroll
            for (int lane = 0; lane < 2; lane++) {
                int i = it * 8 + col_off + lane;

                // d_g = sum_k (1 << k) · (m_pc - c_pc)
                int d_g_int = 0;
                #pragma unroll
                for (int k = 0; k < QB; k++) {
                    d_g_int += (1 << k) * (m_pc[k][lane] - c_pc[k][lane]);
                }
                float d_g = (float)d_g_int;

                float q_scale_i = q_meta[i * 2 + 0];
                float q_offset_i = q_meta[i * 2 + 1];
                float resi = scl_j * (q_offset_i * s_g + q_scale_i * d_g);
                float qc = qc_table[i * K + cid_j];
                float est = cos_j * qc + sin_j * resi;

                // Mask gate is per-thread (no warp sync needed): masked
                // doc-tokens never contribute to this thread's running
                // max, identical to the scalar/Triton path's behavior.
                if (mask_j > 0.0f && est > ms[it][lane]) ms[it][lane] = est;
            }
        }
    }

    // Final reduction: sum max_sim across all 32 query tokens for `d`.
    // Each thread holds 8 vals (4 i_tiles × 2 lanes); the 4 threads of
    // this row-group collectively own all 32 values.
    float my_sum = 0.0f;
    #pragma unroll
    for (int it = 0; it < NUM_I_TILES; it++) {
        if (ms[it][0] > -INFINITY) my_sum += ms[it][0];
        if (ms[it][1] > -INFINITY) my_sum += ms[it][1];
    }
    my_sum += __shfl_xor_sync(0xffffffff, my_sum, 1);
    my_sum += __shfl_xor_sync(0xffffffff, my_sum, 2);

    // Lane 0 of each row-group writes the final per-doc score.
    if ((tid & 3) == 0) {
        scores[d] = my_sum;
    }
}

torch::Tensor fused_b1_rroq158(
    torch::Tensor pos_doc,    // (B, T, 4) i32
    torch::Tensor neg_doc,    // (B, T, 4) i32
    torch::Tensor q_planes,   // (S, QB, 4) i32
    torch::Tensor q_meta,     // (S, 2)
    torch::Tensor qc_table,   // (S, K)
    torch::Tensor scl,        // (B, T)  -- n_groups=1
    torch::Tensor cid,        // (B, T)  i32
    torch::Tensor cosn,       // (B, T)
    torch::Tensor sinn,       // (B, T)
    torch::Tensor mask) {     // (B, T)
    TORCH_CHECK(pos_doc.is_cuda() && pos_doc.dtype() == torch::kInt32);
    TORCH_CHECK(neg_doc.is_cuda() && neg_doc.dtype() == torch::kInt32);
    TORCH_CHECK(q_planes.is_cuda() && q_planes.dtype() == torch::kInt32);
    TORCH_CHECK(pos_doc.dim() == 3 && pos_doc.size(2) == 4);
    TORCH_CHECK(q_planes.dim() == 3 && q_planes.size(0) == 32 &&
                q_planes.size(1) == 4 && q_planes.size(2) == 4);
    int B = pos_doc.size(0);
    int T = pos_doc.size(1);
    int K = qc_table.size(1);
    TORCH_CHECK(B % 8 == 0, "B must be a multiple of 8");

    auto scores = torch::zeros({B}, torch::dtype(torch::kFloat32).device(pos_doc.device()));
    dim3 grid(B / 8);
    dim3 block(32);
    fused_b1_rroq158_kernel<<<grid, block>>>(
        reinterpret_cast<const uint32_t*>(pos_doc.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(neg_doc.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(q_planes.data_ptr<int32_t>()),
        q_meta.data_ptr<float>(),
        qc_table.data_ptr<float>(),
        scl.data_ptr<float>(),
        cid.data_ptr<int32_t>(),
        cosn.data_ptr<float>(),
        sinn.data_ptr<float>(),
        mask.data_ptr<float>(),
        scores.data_ptr<float>(),
        B, T, K);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in fused: ", cudaGetErrorString(err));
    return scores;
}
"""

CPP_DECL = r"""
torch::Tensor mma_b1(torch::Tensor A, torch::Tensor B);
torch::Tensor fused_b1_rroq158(
    torch::Tensor pos_doc,
    torch::Tensor neg_doc,
    torch::Tensor q_planes,
    torch::Tensor q_meta,
    torch::Tensor qc_table,
    torch::Tensor scl,
    torch::Tensor cid,
    torch::Tensor cosn,
    torch::Tensor sinn,
    torch::Tensor mask);
"""


def _load_extension():
    print("[smoke] JIT-compiling CUDA b1 MMA extension...", flush=True)
    t0 = time.perf_counter()
    ext = load_inline(
        name="smoke_b1_mma",
        cpp_sources=CPP_DECL,
        cuda_sources=CUDA_SRC,
        functions=["mma_b1", "fused_b1_rroq158"],
        verbose=False,
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_90,code=sm_90",
            "-std=c++17",
        ],
    )
    print(f"[smoke] compile done in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext


# ─────────────────────────────────────────────────────────────────────
# Reference popcount-AND (slow but bit-exact)
# ─────────────────────────────────────────────────────────────────────

def torch_popc_and(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C[m, n] = popcount( A[m, :] AND B[n, :] )  for 128-bit rows.

    Works for any 2D (M, n_words) / (N, n_words) int32 tensors. Done as
    a tiny pytorch chain of bitwise AND + per-byte popcount via int64
    bit_count (PyTorch 2.0+).
    """
    A = A.to(torch.int64)
    B = B.to(torch.int64)
    A_exp = A.unsqueeze(1).expand(-1, B.size(0), -1)  # (M, N, n_words)
    B_exp = B.unsqueeze(0).expand(A.size(0), -1, -1)  # (M, N, n_words)
    masked = A_exp & B_exp  # (M, N, n_words) int64
    # bit_count over int64 — but we only stored 32 bits of meaningful data per
    # word. count_ones works for 32-bit values stored in int64 too because
    # higher bits are 0.
    pc = masked.bitwise_and(0xFFFFFFFF).to(torch.int64)
    out = torch.zeros(A.size(0), B.size(0), dtype=torch.int32, device=A.device)
    for w in range(A.size(1)):
        out += _popcount_int32(pc[..., w]).to(torch.int32)
    return out


def _popcount_int32(x: torch.Tensor) -> torch.Tensor:
    """SWAR popcount for 32-bit values held in int64 lanes."""
    x = x & 0xFFFFFFFF
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = (x * 0x01010101) & 0xFFFFFFFF
    return (x >> 24) & 0xFF


# ─────────────────────────────────────────────────────────────────────
# fp32 epilogue: combine m_sums, c_sums into final (A, B) scores
# ─────────────────────────────────────────────────────────────────────

def b1_epilogue(
    m_sums: torch.Tensor,         # (B*T, S, query_bits) int32
    c_sums: torch.Tensor,         # (B*T, S, query_bits) int32
    pos_popc: torch.Tensor,       # (B*T,) int32 = popc(pos_doc[bt])
    neg_popc: torch.Tensor,       # (B*T,) int32 = popc(neg_doc[bt])
    scl: torch.Tensor,            # (B, T, n_groups=1) float32
    cid: torch.Tensor,            # (B, T) int32
    cosn: torch.Tensor,           # (B, T) float32
    sinn: torch.Tensor,           # (B, T) float32
    mask: torch.Tensor,           # (B, T) float32
    q_meta: torch.Tensor,         # (S, 2) float32 = [scale, offset]
    qc_table: torch.Tensor,       # (S, K) float32
    *,
    B: int,
    T: int,
    S: int,
    query_bits: int,
) -> torch.Tensor:
    """Reproduce the rroq158 fp32 math from the popcount sums."""
    weights = (1 << torch.arange(query_bits, device=m_sums.device)).to(torch.float32)
    diff = (m_sums - c_sums).to(torch.float32)  # (B*T, S, query_bits)
    d_g = (diff * weights).sum(dim=-1)  # (B*T, S)
    s_g = (pos_popc - neg_popc).to(torch.float32)  # (B*T,)

    d_g = d_g.view(B, T, S)            # (B, T, S)
    s_g = s_g.view(B, T)               # (B, T)
    d_scale = scl[..., 0]              # (B, T)  (n_groups=1)

    q_scale = q_meta[:, 0]             # (S,)
    q_offset = q_meta[:, 1]            # (S,)

    # resi[b, t, s] = d_scale[b, t] * (q_offset[s] * s_g[b, t] + q_scale[s] * d_g[b, t, s])
    resi = d_scale.unsqueeze(-1) * (
        q_offset[None, None, :] * s_g.unsqueeze(-1)
        + q_scale[None, None, :] * d_g
    )                                   # (B, T, S)

    qc = qc_table[:, cid]              # (S, B, T) — gather over centroid id
    qc = qc.permute(1, 2, 0)           # (B, T, S)
    est = cosn.unsqueeze(-1) * qc + sinn.unsqueeze(-1) * resi  # (B, T, S)
    est = est.where(mask.unsqueeze(-1) > 0, torch.full_like(est, -1e9))
    score_per_qtok = est.amax(dim=1)   # (B, S)
    return score_per_qtok.sum(dim=-1)  # (B,)


# ─────────────────────────────────────────────────────────────────────
# Synthetic rroq158 payload at production shape
# ─────────────────────────────────────────────────────────────────────

def make_payload(
    n_docs: int = 2048,
    avg_tokens: int = 220,
    p95_tokens: int = 273,
    dim: int = 128,
    K: int = 1024,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    # Generate variable-length per-doc-token vectors
    tok_counts = rng.integers(60, max(p95_tokens + 1, 280), size=n_docs)
    tok_counts = np.clip(tok_counts, 1, p95_tokens + 30)
    total_tokens = int(tok_counts.sum())
    all_vec = rng.standard_normal((total_tokens, dim)).astype(np.float32)
    all_vec /= np.linalg.norm(all_vec, axis=1, keepdims=True) + 1e-8

    doc_offsets = []
    cursor = 0
    for n in tok_counts:
        doc_offsets.append((cursor, cursor + int(n)))
        cursor += int(n)

    cfg = Rroq158Config(K=K, group_size=128, seed=seed)
    enc = encode_rroq158(all_vec, cfg)

    n_words = enc.sign_plane.shape[1]
    n_int32_words = n_words // 4
    n_groups = enc.scales.shape[1]
    p95 = int(np.ceil(np.percentile(tok_counts, 95)))
    align = 32
    t_max = ((p95 + align - 1) // align) * align

    sign_dt = np.zeros((n_docs, t_max, n_int32_words), dtype=np.int32)
    nz_dt = np.zeros((n_docs, t_max, n_int32_words), dtype=np.int32)
    scl_dt = np.zeros((n_docs, t_max, n_groups), dtype=np.float32)
    cid_dt = np.zeros((n_docs, t_max), dtype=np.int32)
    cosn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    sinn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    mask_dt = np.zeros((n_docs, t_max), dtype=np.float32)

    for di, (s, e) in enumerate(doc_offsets):
        n_tok = min(e - s, t_max)
        sign_words = pack_doc_codes_to_int32_words(enc.sign_plane[s:s + n_tok])
        nz_words = pack_doc_codes_to_int32_words(enc.nonzero_plane[s:s + n_tok])
        sign_dt[di, :n_tok] = sign_words
        nz_dt[di, :n_tok] = nz_words
        scl_dt[di, :n_tok] = enc.scales[s:s + n_tok].astype(np.float32)
        cid_dt[di, :n_tok] = enc.centroid_id[s:s + n_tok].astype(np.int32)
        cosn_dt[di, :n_tok] = enc.cos_norm[s:s + n_tok].astype(np.float32)
        sinn_dt[di, :n_tok] = enc.sin_norm[s:s + n_tok].astype(np.float32)
        mask_dt[di, :n_tok] = 1.0

    return {
        "sign": torch.from_numpy(sign_dt),
        "nz": torch.from_numpy(nz_dt),
        "scl": torch.from_numpy(scl_dt),
        "cid": torch.from_numpy(cid_dt),
        "cosn": torch.from_numpy(cosn_dt),
        "sinn": torch.from_numpy(sinn_dt),
        "mask": torch.from_numpy(mask_dt),
        "centroids": torch.from_numpy(enc.centroids),
        "fwht_seed": enc.fwht_seed,
        "n_words": n_int32_words,
        "n_groups": n_groups,
        "K": K,
        "t_max": t_max,
        "dim": dim,
    }


def make_query(payload: dict, S: int, dim: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((S, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q_inputs = encode_query_for_rroq158(
        q, None,
        fwht_seed=payload["fwht_seed"],
        query_bits=4,
        rotator=None,
        skip_qc_table=True,
        cap_blas_threads=False,
    )
    return {
        "query_np": q,
        "q_planes": q_inputs["q_planes"],   # (S, query_bits, n_words) int32
        "q_meta": q_inputs["q_meta"],       # (S, 2) float32
    }


# ─────────────────────────────────────────────────────────────────────
# Main smoke test
# ─────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("CUDA not available; this smoke test requires a GPU.")
        sys.exit(2)

    device = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(0)
    print(f"[smoke] GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    if cap[0] < 8:
        print(f"[smoke] WARNING: sm_{cap[0]}{cap[1]} may not support .and.popc; "
              "smoke test may fail to launch")

    ext = _load_extension()

    # ---- Build payload (CPU then push to GPU) ------------------------
    print("[smoke] building synthetic rroq158 payload...", flush=True)
    payload = make_payload(n_docs=2048)
    n_docs = payload["sign"].shape[0]
    t_max = payload["t_max"]
    dim = payload["dim"]
    n_words = payload["n_words"]
    n_groups = payload["n_groups"]
    K = payload["K"]
    print(f"[smoke]   B={n_docs}, T_max={t_max}, dim={dim}, n_words={n_words}, "
          f"n_groups={n_groups}, K={K}")

    # Move to GPU
    for k in ("sign", "nz", "scl", "cid", "cosn", "sinn", "mask", "centroids"):
        payload[k] = payload[k].to(device)

    S = 32
    query = make_query(payload, S, dim)
    q_planes_t = torch.from_numpy(query["q_planes"]).to(device)
    q_meta_t = torch.from_numpy(query["q_meta"]).to(device)
    q_dev = torch.from_numpy(query["query_np"]).to(device)
    qc_table_t = q_dev @ payload["centroids"].T  # (S, K)
    print(f"[smoke]   S={S}, query_bits={query['q_planes'].shape[1]}")

    # ---- Bit-exact unit test of the b1 MMA kernel --------------------
    print("[smoke] unit test: b1 MMA matches torch popcount-AND on small input")
    rng = np.random.default_rng(42)
    A_np = rng.integers(low=-(2**31), high=2**31, size=(64, 4), dtype=np.int32)
    B_np = rng.integers(low=-(2**31), high=2**31, size=(8, 4), dtype=np.int32)
    A_t = torch.from_numpy(A_np).to(device)
    B_t = torch.from_numpy(B_np).to(device)
    C_mma = ext.mma_b1(A_t, B_t).cpu().numpy()
    C_ref = torch_popc_and(A_t, B_t).cpu().numpy()
    if not np.array_equal(C_mma, C_ref):
        diffs = np.argwhere(C_mma != C_ref)
        print(f"[smoke] FAIL: b1 MMA mismatched on {len(diffs)} entries")
        print(f"  first 5 mismatches: {diffs[:5]}")
        for r, c in diffs[:5]:
            print(f"    [{r},{c}] mma={C_mma[r,c]} ref={C_ref[r,c]}")
        sys.exit(1)
    print("[smoke]   OK — b1 MMA bit-exact")

    # ---- Build the (B*T, n_words) packed pos / neg planes ------------
    sign4 = payload["sign"].view(n_docs * t_max, n_words).contiguous()  # (B*T, 4)
    nz4 = payload["nz"].view(n_docs * t_max, n_words).contiguous()
    pos_doc = (sign4 & nz4).contiguous()
    neg_doc = ((~sign4) & nz4).contiguous()

    # Pad B*T to a multiple of 8 for the m8n8k128 tile (drop padding rows
    # in the epilogue via the mask).
    n_rows = n_docs * t_max
    pad_rows = (8 - n_rows % 8) % 8
    if pad_rows:
        zero_pad = torch.zeros(pad_rows, n_words, dtype=torch.int32, device=device)
        pos_doc = torch.cat([pos_doc, zero_pad], dim=0)
        neg_doc = torch.cat([neg_doc, zero_pad], dim=0)
    M = pos_doc.shape[0]

    # Pack q_planes into (S * query_bits, n_words). For each query token i and
    # bit-plane k, take q_planes[i, k, :] and stack. Pad to multiple of 8.
    query_bits = q_planes_t.shape[1]
    qp_flat = q_planes_t.permute(0, 1, 2).reshape(S * query_bits, n_words).contiguous()
    pad_n = (8 - (S * query_bits) % 8) % 8
    if pad_n:
        qp_flat = torch.cat(
            [qp_flat, torch.zeros(pad_n, n_words, dtype=torch.int32, device=device)],
            dim=0,
        )
    N = qp_flat.shape[0]
    print(f"[smoke]   b1 MMA matrix: A=({M}, 128 bits), B=({N}, 128 bits)")

    # ---- One-shot full b1 MMA path  -------------------------------------
    torch.cuda.synchronize()
    m_sums = ext.mma_b1(pos_doc, qp_flat)  # (M, N) int32
    c_sums = ext.mma_b1(neg_doc, qp_flat)
    torch.cuda.synchronize()

    # Trim padding back to (B*T, S*query_bits)
    m_sums = m_sums[: n_docs * t_max, : S * query_bits]
    c_sums = c_sums[: n_docs * t_max, : S * query_bits]
    m_sums = m_sums.view(n_docs * t_max, S, query_bits)
    c_sums = c_sums.view(n_docs * t_max, S, query_bits)

    # Doc-side popcount sums (one-time per corpus; not in hot path)
    pos_popc = (sign4 & nz4)
    pos_popc = sum(_popcount_int32(pos_popc[..., w].to(torch.int64))
                   for w in range(n_words)).to(torch.int32)
    neg_popc = ((~sign4) & nz4)
    neg_popc = sum(_popcount_int32(neg_popc[..., w].to(torch.int64))
                   for w in range(n_words)).to(torch.int32)

    scores_b1 = b1_epilogue(
        m_sums, c_sums, pos_popc, neg_popc,
        payload["scl"], payload["cid"], payload["cosn"], payload["sinn"],
        payload["mask"], q_meta_t, qc_table_t,
        B=n_docs, T=t_max, S=S, query_bits=query_bits,
    )
    print(f"[smoke]   scores_b1 shape={tuple(scores_b1.shape)}")

    # ---- Fused b1 MMA kernel (no intermediate materialise) -------------
    pos_doc_3d = (payload["sign"] & payload["nz"]).contiguous()    # (B, T, 4)
    neg_doc_3d = ((~payload["sign"]) & payload["nz"]).contiguous()
    scores_fused = ext.fused_b1_rroq158(
        pos_doc_3d, neg_doc_3d,
        q_planes_t, q_meta_t, qc_table_t,
        payload["scl"][..., 0].contiguous(),
        payload["cid"], payload["cosn"], payload["sinn"], payload["mask"],
    )
    print(f"[smoke]   scores_fused shape={tuple(scores_fused.shape)}")

    # ---- Triton baseline -----------------------------------------------
    docs_sign = payload["sign"]   # (B, T, n_words) int32
    docs_nz = payload["nz"]
    docs_scl = payload["scl"]
    docs_cid = payload["cid"]
    docs_cosn = payload["cosn"]
    docs_sinn = payload["sinn"]
    docs_mask = payload["mask"]

    q_planes_b = q_planes_t.unsqueeze(0)        # (1, S, qb, n_words)
    q_meta_b = q_meta_t.unsqueeze(0)            # (1, S, 2)
    qc_table_b = qc_table_t.unsqueeze(0)        # (1, S, K)

    # Warmup Triton autotune
    for _ in range(3):
        scores_triton = roq_maxsim_rroq158(
            q_planes_b, q_meta_b, qc_table_b,
            docs_cid, docs_cosn, docs_sinn,
            docs_sign, docs_nz, docs_scl,
            documents_mask=docs_mask,
        )
    torch.cuda.synchronize()
    scores_triton = scores_triton.squeeze(0)
    print(f"[smoke]   scores_triton shape={tuple(scores_triton.shape)}")

    # ---- Parity checks --------------------------------------------------
    abs_err_unfused = (scores_b1 - scores_triton).abs()
    rel_err_unfused = abs_err_unfused / (scores_triton.abs() + 1e-6)
    abs_err_fused = (scores_fused - scores_triton).abs()
    rel_err_fused = abs_err_fused / (scores_triton.abs() + 1e-6)
    print(f"[smoke]   parity unfused: max abs err = {abs_err_unfused.max().item():.4e}, "
          f"max rel err = {rel_err_unfused.max().item():.4e}")
    print(f"[smoke]   parity FUSED:   max abs err = {abs_err_fused.max().item():.4e}, "
          f"max rel err = {rel_err_fused.max().item():.4e}")
    fused_parity_ok = rel_err_fused.max().item() <= 5e-3
    if fused_parity_ok:
        print(f"[smoke]   OK — FUSED parity within 5e-3 relative tolerance")
    else:
        print(f"[smoke]   WARN: FUSED parity outside 5e-3 — investigate epilogue")

    # ---- Microbench ----------------------------------------------------
    print("[smoke] microbench (kernel-only, 100 iter, 5 warmup)...")

    def bench_triton(iters: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            roq_maxsim_rroq158(
                q_planes_b, q_meta_b, qc_table_b,
                docs_cid, docs_cosn, docs_sinn,
                docs_sign, docs_nz, docs_scl,
                documents_mask=docs_mask,
            )
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3  # ms

    def bench_b1(iters: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            ext.mma_b1(pos_doc, qp_flat)
            ext.mma_b1(neg_doc, qp_flat)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3  # ms

    def bench_b1_with_epilogue(iters: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m = ext.mma_b1(pos_doc, qp_flat)[: n_docs * t_max, : S * query_bits]
            c = ext.mma_b1(neg_doc, qp_flat)[: n_docs * t_max, : S * query_bits]
            m = m.view(n_docs * t_max, S, query_bits)
            c = c.view(n_docs * t_max, S, query_bits)
            b1_epilogue(
                m, c, pos_popc, neg_popc,
                payload["scl"], payload["cid"], payload["cosn"], payload["sinn"],
                payload["mask"], q_meta_t, qc_table_t,
                B=n_docs, T=t_max, S=S, query_bits=query_bits,
            )
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    def bench_fused(iters: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            ext.fused_b1_rroq158(
                pos_doc_3d, neg_doc_3d,
                q_planes_t, q_meta_t, qc_table_t,
                payload["scl"][..., 0].contiguous(),
                payload["cid"], payload["cosn"], payload["sinn"], payload["mask"],
            )
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    # warmup
    bench_triton(5)
    bench_b1(5)
    bench_b1_with_epilogue(5)
    bench_fused(5)

    triton_ms = bench_triton(100)
    b1_only_ms = bench_b1(100)
    b1_full_ms = bench_b1_with_epilogue(100)
    fused_ms = bench_fused(100)

    # ---- Memory footprint of intermediate ------------------------------
    intermediate_bytes = (M * N + M * N) * 4  # m_sums + c_sums int32
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  shape: B={n_docs}, T={t_max}, S={S}, query_bits={query_bits}, "
          f"dim={dim}")
    print(f"  Triton popc kernel (current production):     {triton_ms:8.3f} ms / query")
    print(f"  CUDA b1 MMA only (2x mma_b1 calls, unfused): {b1_only_ms:8.3f} ms / query")
    print(f"  CUDA b1 MMA + py epilogue (unfused full):    {b1_full_ms:8.3f} ms / query")
    print(f"  CUDA FUSED b1 + epilogue (single kernel):    {fused_ms:8.3f} ms / query")
    print(f"  speedup b1-MMA-only vs Triton:               {triton_ms / b1_only_ms:8.2f}x")
    print(f"  speedup b1-MMA + py epilogue vs Triton:      {triton_ms / b1_full_ms:8.2f}x")
    print(f"  speedup FUSED vs Triton:                     {triton_ms / fused_ms:8.2f}x")
    print(f"  intermediate buffer (unfused): {intermediate_bytes / 1024**2:.1f} MB "
          f"(2 x ({M} x {N}) int32)")
    print(f"  intermediate buffer (FUSED):              0 MB  (registers + (B,) only)")
    print(f"  parity unfused max rel err: {rel_err_unfused.max().item():.4e}")
    print(f"  parity FUSED    max rel err: {rel_err_fused.max().item():.4e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
