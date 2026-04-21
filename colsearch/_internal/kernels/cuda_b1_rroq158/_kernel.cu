// Fused b1 (binary tensor-core) MaxSim kernel for the rroq158 codec.
//
// Companion to ``voyager_index/_internal/kernels/triton_roq_rroq158.py``.
// Replaces the Triton popcount path with a single fused CUDA kernel
// that exploits ``mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc``
// (binary tensor cores, sm_75+; we validate sm_90+).
//
// Layout & contract
// -----------------
// Inputs are the SAME tensors the production code already passes to the
// Triton kernel — we don't materialise any intermediate buffer:
//
//   docs_sign  : (B, T, 4) i32   — sign-bit plane (n_words=4 ≡ dim=128)
//   docs_nz    : (B, T, 4) i32   — non-zero-bit plane
//   q_planes   : (S, 4, 4) i32   — (S, query_bits=4, n_words=4)
//   q_meta     : (S, 2)  f32     — [scale, offset] per query token
//   qc_table   : (S, K)  f32     — query·centroid table
//   scl        : (B, T)  f32     — per-token scale (n_groups=1)
//   cid        : (B, T)  i32     — per-token centroid id
//   cosn,sinn  : (B, T)  f32     — per-token cos/sin norm
//   mask       : (B, T)  f32     — per-token doc mask (1 active, 0 pad)
//   scores     : (B,)    f32     — output per-doc MaxSim score
//
// Hard-specialisation
// -------------------
// dim=128 (n_words=4), n_groups=1 (group_words=4), query_bits=4, S=32.
// Other shapes MUST fall back to Triton — gated in Python by
// ``cuda_b1_rroq158._shape_supported``.
//
// Tile scheme
// -----------
// 1 warp = 1 block = 8 docs × all 32 query tokens × all T j-tokens.
// Per j: 32 m8n8k128 mma instructions (4 i-tiles × 4 query bit-planes
// × 2 sign/neg). Per (i_tile, lane): epilogue reduces directly into the
// running max in registers. After the j-loop a 4-way warp reduction
// produces the final per-doc score, written once to global memory.
//
// Bug-of-record
// -------------
// An earlier prototype used ``if (mask_j <= 0) continue;`` on the
// per-thread mask — this caused divergent control flow in a warp where
// different docs have different mask patterns, which corrupted the
// warp-collective ``__shfl_xor_sync`` and ``mma.sync`` instructions.
// The current implementation uses warp-uniform ``__ballot_sync`` to
// skip j only when ALL 8 docs in the tile are masked, and gates the
// per-thread max-update on the per-thread mask. Parity verified
// bit-exact against the Triton kernel in
// ``benchmarks/_smoke_b1_mma.py``.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>


extern "C" __global__ void __launch_bounds__(32)
fused_b1_rroq158_kernel(
    const uint32_t* __restrict__ docs_sign,  // (B, T, 4)
    const uint32_t* __restrict__ docs_nz,    // (B, T, 4)
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

    const int row_base = (d * T) * W;        // docs_sign[d, 0, 0]
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

        // Load doc bits for this j (1 uint32 per thread = 32 K-bits;
        // 8 docs × 4 words = 32 fragments per warp = a single coalesced
        // 128-byte transaction per sign and per nz). Compute pos/neg
        // per-thread so the caller doesn't need to materialise (B, T, 4)
        // pos/neg buffers.
        uint32_t sign_frag = docs_sign[row_base + j * W + word];
        uint32_t nz_frag = docs_nz[row_base + j * W + word];
        uint32_t pos_frag = sign_frag & nz_frag;
        uint32_t neg_frag = (~sign_frag) & nz_frag;

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
            int m_pc[QB][2];
            int c_pc[QB][2];

            #pragma unroll
            for (int k = 0; k < QB; k++) {
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
    torch::Tensor docs_sign,  // (B, T, 4) i32
    torch::Tensor docs_nz,    // (B, T, 4) i32
    torch::Tensor q_planes,   // (S, QB, 4) i32
    torch::Tensor q_meta,     // (S, 2)
    torch::Tensor qc_table,   // (S, K)
    torch::Tensor scl,        // (B, T)  -- n_groups=1
    torch::Tensor cid,        // (B, T)  i32
    torch::Tensor cosn,       // (B, T)
    torch::Tensor sinn,       // (B, T)
    torch::Tensor mask) {     // (B, T)
    TORCH_CHECK(docs_sign.is_cuda() && docs_sign.dtype() == torch::kInt32,
                "docs_sign must be CUDA int32");
    TORCH_CHECK(docs_nz.is_cuda() && docs_nz.dtype() == torch::kInt32,
                "docs_nz must be CUDA int32");
    TORCH_CHECK(q_planes.is_cuda() && q_planes.dtype() == torch::kInt32,
                "q_planes must be CUDA int32");
    TORCH_CHECK(docs_sign.dim() == 3 && docs_sign.size(2) == 4,
                "docs_sign must be (B, T, 4)");
    TORCH_CHECK(q_planes.dim() == 3 && q_planes.size(0) == 32 &&
                q_planes.size(1) == 4 && q_planes.size(2) == 4,
                "q_planes must be (32, 4, 4)");
    int B = docs_sign.size(0);
    int T = docs_sign.size(1);
    int K = qc_table.size(1);
    TORCH_CHECK(B % 8 == 0, "B must be a multiple of 8 (caller pads)");

    auto scores = torch::zeros({B},
        torch::dtype(torch::kFloat32).device(docs_sign.device()));

    dim3 grid(B / 8);
    dim3 block(32);
    fused_b1_rroq158_kernel<<<grid, block>>>(
        reinterpret_cast<const uint32_t*>(docs_sign.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(docs_nz.data_ptr<int32_t>()),
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
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error in fused_b1_rroq158: ", cudaGetErrorString(err));
    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_b1_rroq158", &fused_b1_rroq158,
          "Fused b1 MMA + epilogue rroq158 MaxSim kernel (sm_90+).");
}
