
import torch
import triton
import triton.language as tl


@triton.jit
def _popc(x):
    return tl.inline_asm_elementwise("popc.b32 $0, $1;", "=r,r", [x], dtype=tl.int32, is_pure=True, pack=1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 128}, num_warps=8, num_stages=2),
    ],
    key=['n_d_tokens', 'dim']
)
@triton.jit
def _roq_maxsim_1bit_kernel(
    Q_PTR, D_PTR,
    OUTPUT_PTR,
    q_batch_stride, dim, # q_token_stride is dim * 4
    d_batch_stride, d_token_stride,
    output_q_stride, output_d_stride,
    n_queries, n_docs,
    n_q_tokens, n_d_tokens,
    n_bytes: tl.constexpr,          # dim / 8
    BLOCK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """
    1-bit RoQ MaxSim Kernel (Symmetric).
    """
    q_idx = tl.program_id(0)
    d_idx = tl.program_id(1)

    q_ptr_base = Q_PTR + q_idx * q_batch_stride
    d_ptr_base = D_PTR + d_idx * d_batch_stride

    total_score = 0.0

    # Calculate number of int32s per vector (n_bytes / 4)
    # Assumes n_bytes is multiple of 4 (dim 128 -> 16 bytes -> 4 ints)
    n_bytes // 4
    tl.arange(0, 4) # Hardcoded for 128 dim (4 ints)

    # Cosine correction scale
    scaled_dim = dim.to(tl.float32)

    # REVERTING TO ORIGINAL SYMMETRIC LOGIC CAREFULLY
    # q_token_stride was computed as 16 in launcher?
    # Let's assume standard packed layout.
    q_stride_bytes = n_bytes

    # Offsets for byte-wise loading
    byte_offsets = tl.arange(0, BLOCK_DIM)

    # Loop over query tokens
    for i in range(n_q_tokens):
        q_off = i * q_stride_bytes
        q_ptr = q_ptr_base + q_off
        # q_ptr_i32 = q_ptr.to(tl.pointer_type(tl.int32)) # Not used in new logic
        # q_vec = tl.load(q_ptr_i32 + n_ints_offsets) # Original load

        # Load Q Vector (byte-wise)
        q_vec_bytes = tl.load(q_ptr + byte_offsets, mask=byte_offsets < n_bytes, other=0)
        # Mask to ensure unsigned for popc
        q_vec_bytes = q_vec_bytes.to(tl.int32) & 0xFF

        max_sim = -1.0e9

        for j_start in range(0, n_d_tokens, BLOCK_D):
            j_offsets = j_start + tl.arange(0, BLOCK_D)
            j_mask = j_offsets < n_d_tokens

            # Doc ptr
            # d_off = j_offsets[:, None] * n_bytes # [BLOCK_D, 1] # Original d_off
            # d_ptr = d_ptr_base + d_off # Original d_ptr
            # d_ptr_i32 = d_ptr.to(tl.pointer_type(tl.int32)) # Original d_ptr_i32
            # d_ptrs = d_ptr_i32 + n_ints_offsets[None, :] # [BLOCK_D, n_ints] # Original d_ptrs
            # d_vecs = tl.load(d_ptrs, mask=j_mask[:, None], other=0) # Original d_vecs

            # Load D Vectors [BLOCK_D, n_bytes]
            d_ptr_offs = j_offsets[:, None] * n_bytes + byte_offsets[None, :]
            # d_ptr_offs assumed within int32 range

            d_vecs_bytes = tl.load(d_ptr_base + d_ptr_offs,
                                   mask=(j_mask[:, None] & (byte_offsets[None, :] < n_bytes)), other=0)
            d_vecs_bytes = d_vecs_bytes.to(tl.int32) & 0xFF

            # XOR
            # q: [BLOCK_DIM], d: [BLOCK_D, BLOCK_DIM]
            xor_res = q_vec_bytes[None, :] ^ d_vecs_bytes
            bits_diff = _popc(xor_res) # [BLOCK_D, BLOCK_DIM]
            dist_int = tl.sum(bits_diff, axis=1)
            dist_float = dist_int.to(tl.float32)

            # Cosine Correction
            # sim = D * cos(pi * h / D)
            angle = 3.14159265 * dist_float / scaled_dim
            sim = scaled_dim * tl.cos(angle)

            sim = tl.where(j_mask, sim, -1.0e9)
            block_max = tl.max(sim, axis=0)
            if block_max > max_sim:
                max_sim = block_max

        total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
    ],
    key=['n_d_tokens', 'dim']
)
@triton.jit
def _roq_maxsim_1bit_asymmetric_kernel(
    Q_PLANES_PTR, D_CODES_PTR,
    Q_META_PTR,
    Q_MASK_PTR, D_MASK_PTR,
    OUTPUT_PTR,
    q_batch_stride, q_token_stride, q_plane_stride, q_word_stride,
    d_batch_stride, d_token_stride, d_word_stride,
    q_meta_token_stride,
    q_mask_batch_stride, q_mask_token_stride,
    d_mask_batch_stride, d_mask_token_stride,
    output_q_stride, output_d_stride,
    n_queries, n_docs,
    n_q_tokens, n_d_tokens,
    dim,
    N_WORDS: tl.constexpr,
    QUERY_BITS: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """
    Asymmetric 1-bit RoQ MaxSim kernel.

    Documents are stored as packed sign bits.
    Queries are stored as packed bit-planes over the scalar query codes.
    """
    q_idx = tl.program_id(0)
    d_idx = tl.program_id(1)

    q_ptr_base = Q_PLANES_PTR + q_idx * q_batch_stride
    d_ptr_base = D_CODES_PTR + d_idx * d_batch_stride
    q_meta_ptr_base = Q_META_PTR + q_idx * n_q_tokens * q_meta_token_stride

    total_score = 0.0

    for i in range(n_q_tokens):
        q_token_active = tl.load(
            Q_MASK_PTR + q_idx * q_mask_batch_stride + i * q_mask_token_stride
        ) > 0
        if q_token_active:
            q_meta_ptr = q_meta_ptr_base + i * q_meta_token_stride
            q_scale = tl.load(q_meta_ptr + 0)
            q_offset = tl.load(q_meta_ptr + 1)
            q_sum = tl.load(q_meta_ptr + 2)
            q_affine_sum = dim * q_offset + q_scale * q_sum

            max_sim = -1.0e9
            for j_start in range(0, n_d_tokens, BLOCK_D):
                j_offsets = j_start + tl.arange(0, BLOCK_D)
                j_mask = j_offsets < n_d_tokens
                d_token_active = tl.load(
                    D_MASK_PTR + d_idx * d_mask_batch_stride + j_offsets * d_mask_token_stride,
                    mask=j_mask,
                    other=0,
                ) > 0
                d_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
                dot_val = tl.zeros([BLOCK_D], dtype=tl.float32)
                q_token_base = q_ptr_base + i * q_token_stride
                d_token_base = d_ptr_base + j_offsets * d_token_stride
                for word_idx in range(N_WORDS):
                    d_word = tl.load(
                        d_token_base + word_idx * d_word_stride,
                        mask=j_mask,
                        other=0,
                    ).to(tl.int32)
                    d_sum += _popc(d_word).to(tl.float32)

                    q_plane0 = tl.load(q_token_base + 0 * q_plane_stride + word_idx * q_word_stride).to(tl.int32)
                    dot_val += _popc(d_word & q_plane0).to(tl.float32)
                    if QUERY_BITS > 1:
                        q_plane1 = tl.load(q_token_base + 1 * q_plane_stride + word_idx * q_word_stride).to(tl.int32)
                        dot_val += 2.0 * _popc(d_word & q_plane1).to(tl.float32)
                    if QUERY_BITS > 2:
                        q_plane2 = tl.load(q_token_base + 2 * q_plane_stride + word_idx * q_word_stride).to(tl.int32)
                        dot_val += 4.0 * _popc(d_word & q_plane2).to(tl.float32)
                    if QUERY_BITS > 3:
                        q_plane3 = tl.load(q_token_base + 3 * q_plane_stride + word_idx * q_word_stride).to(tl.int32)
                        dot_val += 8.0 * _popc(d_word & q_plane3).to(tl.float32)
                    if QUERY_BITS > 4:
                        q_plane4 = tl.load(q_token_base + 4 * q_plane_stride + word_idx * q_word_stride).to(tl.int32)
                        dot_val += 16.0 * _popc(d_word & q_plane4).to(tl.float32)
                    if QUERY_BITS > 5:
                        q_plane5 = tl.load(q_token_base + 5 * q_plane_stride + word_idx * q_word_stride).to(tl.int32)
                        dot_val += 32.0 * _popc(d_word & q_plane5).to(tl.float32)
                    if QUERY_BITS > 6:
                        q_plane6 = tl.load(q_token_base + 6 * q_plane_stride + word_idx * q_word_stride).to(tl.int32)
                        dot_val += 64.0 * _popc(d_word & q_plane6).to(tl.float32)
                    if QUERY_BITS > 7:
                        q_plane7 = tl.load(q_token_base + 7 * q_plane_stride + word_idx * q_word_stride).to(tl.int32)
                        dot_val += 128.0 * _popc(d_word & q_plane7).to(tl.float32)
                est_dot = (2.0 * (q_offset * d_sum + q_scale * dot_val)) - q_affine_sum
                sim = tl.where(j_mask & d_token_active, est_dot, -1.0e9)
                block_max = tl.max(sim, axis=0)
                if block_max > max_sim:
                    max_sim = block_max

            total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


def roq_maxsim_1bit(queries_codes, docs_codes, queries_meta=None, queries_mask=None, documents_mask=None):
    """
    Args:
        queries_codes: symmetric path -> (A, S, n_bytes) packed uint8
                       asymmetric path -> (A, S, query_bits, n_words) int32 query bit-planes
        queries_meta: required for asymmetric path, shape (A, S, 4)
        docs_codes: symmetric path -> (B, T, n_bytes) uint8
                    asymmetric path -> (B, T, n_words) int32
    Returns:
        scores: (A, B) float32
    """
    A, S = queries_codes.shape[:2]
    B, T = docs_codes.shape[:2]
    scores = torch.empty((A, B), dtype=torch.float32, device=queries_codes.device)

    if queries_meta is None:
        NB = docs_codes.shape[-1]
        NB_q = queries_codes.shape[-1]
        assert NB_q == NB
        dim = NB * 8
        block_dim = 32
        while block_dim < NB:
            block_dim *= 2
        grid = (A, B)
        _roq_maxsim_1bit_kernel[grid](
            queries_codes, docs_codes,
            scores,
            queries_codes.stride(0), dim,
            docs_codes.stride(0), docs_codes.stride(1),
            scores.stride(0), scores.stride(1),
            A, B, S, T, NB,
            BLOCK_DIM=block_dim,
        )
        return scores

    query_bits = queries_codes.shape[2]
    n_words = docs_codes.shape[-1]
    dim = n_words * 32
    if queries_mask is None:
        queries_mask = torch.ones((A, S), dtype=torch.float32, device=queries_codes.device)
    else:
        queries_mask = queries_mask.to(device=queries_codes.device, dtype=torch.float32)
    if documents_mask is None:
        documents_mask = torch.ones((B, T), dtype=torch.float32, device=docs_codes.device)
    else:
        documents_mask = documents_mask.to(device=docs_codes.device, dtype=torch.float32)

    grid = (A, B)
    _roq_maxsim_1bit_asymmetric_kernel[grid](
        Q_PLANES_PTR=queries_codes,
        D_CODES_PTR=docs_codes,
        Q_META_PTR=queries_meta,
        Q_MASK_PTR=queries_mask,
        D_MASK_PTR=documents_mask,
        OUTPUT_PTR=scores,
        q_batch_stride=queries_codes.stride(0),
        q_token_stride=queries_codes.stride(1),
        q_plane_stride=queries_codes.stride(2),
        q_word_stride=queries_codes.stride(3),
        d_batch_stride=docs_codes.stride(0),
        d_token_stride=docs_codes.stride(1),
        d_word_stride=docs_codes.stride(2),
        q_meta_token_stride=queries_meta.stride(1),
        q_mask_batch_stride=queries_mask.stride(0),
        q_mask_token_stride=queries_mask.stride(1),
        d_mask_batch_stride=documents_mask.stride(0),
        d_mask_token_stride=documents_mask.stride(1),
        output_q_stride=scores.stride(0),
        output_d_stride=scores.stride(1),
        n_queries=A,
        n_docs=B,
        n_q_tokens=S,
        n_d_tokens=T,
        dim=dim,
        N_WORDS=n_words,
        QUERY_BITS=query_bits,
    )
    return scores


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ],
    key=['n_d_tokens', 'dim']
)
@triton.jit
def _roq_maxsim_8bit_kernel(
    Q_CODES_PTR, D_CODES_PTR,
    Q_META_PTR, D_META_PTR, # [scale, offset, code_sum, norm_sq] per vector
    OUTPUT_PTR,
    q_batch_stride, q_token_stride,
    d_batch_stride, d_token_stride,
    output_q_stride, output_d_stride,
    n_queries, n_docs,
    n_q_tokens, n_d_tokens,
    dim,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DIM: tl.constexpr
):
    """
    8-bit RoQ MaxSim Kernel with Affine Correction.
    Sim(x, y) = <x, y> = (||x||^2 + ||y||^2 - ||x-y||^2) / 2
    But for RoQ, we estimate <x, y> directly:
    <x, q> approx D*l_x*l_q + l_x*d_q*sum(c_q) + d_x*l_q*sum(c_x) + d_x*d_q*<c_x, c_q>

    Meta layout: [scale, offset, code_sum, norm_sq] (floats)
    """
    q_idx = tl.program_id(0)
    d_idx = tl.program_id(1)

    q_ptr_base = Q_CODES_PTR + q_idx * q_batch_stride
    d_ptr_base = D_CODES_PTR + d_idx * d_batch_stride

    q_meta_base = Q_META_PTR + q_idx * n_q_tokens * 4
    d_meta_base = D_META_PTR + d_idx * n_d_tokens * 4

    # dim is passed as scalar (int32).
    # To use in float calc, we can just use it (promotes) or cast.
    # float(dim) fails on JIT types.
    dim_float = dim # Implicit promotion

    # Offsets for dimension
    dim_offsets = tl.arange(0, BLOCK_DIM)

    total_score = 0.0

    # Loop over query tokens
    for i in range(n_q_tokens):
        # Load Query Meta (scalar per token)
        q_meta_ptr = q_meta_base + i * 4
        q_scale = tl.load(q_meta_ptr + 0)
        q_offset = tl.load(q_meta_ptr + 1)
        q_sum = tl.load(q_meta_ptr + 2)

        # Load Query Code Vector (1, Dim)
        # We need this repeated for BLOCK_D, or broadcast
        q_code_ptr = q_ptr_base + i * q_token_stride + dim_offsets
        q_code_mask = dim_offsets < dim
        q_vec = tl.load(q_code_ptr, mask=q_code_mask, other=0.0).to(tl.float32)

        # Constant parts for this query token
        # T1 coeff: D * q_offset
        t1_q_part = dim_float * q_offset
        # T3 coeff: q_scale * q_sum
        t3_coeff = q_scale * q_sum

        max_sim = -1.0e9 # Float32 min

        # Loop over doc tokens in blocks
        for j_start in range(0, n_d_tokens, BLOCK_D):
            # Block offsets
            j_offsets = j_start + tl.arange(0, BLOCK_D)
            j_mask = j_offsets < n_d_tokens

            # Load Doc Meta [BLOCK_D]
            # Layout: (B, T, 4) -> (T, 4) contiguous since B is handled by grid
            # d_meta_base points to start of docs for this batch item
            d_meta_ptr = d_meta_base + j_offsets * 4

            # Load scalars vectorized [BLOCK_D]
            d_scale = tl.load(d_meta_ptr + 0, mask=j_mask, other=0.0)
            d_offset = tl.load(d_meta_ptr + 1, mask=j_mask, other=0.0)
            d_sum = tl.load(d_meta_ptr + 2, mask=j_mask, other=0.0)

            # Load Doc Codes [BLOCK_D, DIM]
            # Pointer arithmetic: Base + T_offset * stride + DIM_offset
            d_code_ptrs = d_ptr_base + (j_offsets[:, None] * d_token_stride) + dim_offsets[None, :]
            d_code_mask = (j_offsets[:, None] < n_d_tokens) & (dim_offsets[None, :] < dim)

            d_vecs = tl.load(d_code_ptrs, mask=d_code_mask, other=0.0).to(tl.float32)

            # Dot Product <c_x, c_q> [BLOCK_D]
            # q_vec is [DIM], d_vecs is [BLOCK_D, DIM]
            # Broadcast q_vec to [1, DIM] implicitly
            prod = q_vec[None, :] * d_vecs
            dot_val = tl.sum(prod, axis=1)

            # Affine Correction Vectorized
            # T1: D * q_offset * d_offset
            t1 = t1_q_part * d_offset

            # T2: d_scale * d_sum * q_offset
            t2 = d_scale * d_sum * q_offset

            # T3: q_scale * q_sum * d_offset (t3_coeff computed outside)
            t3 = t3_coeff * d_offset

            # T4: q_scale * d_scale * dot_val
            t4 = q_scale * d_scale * dot_val

            # Final sim [BLOCK_D]
            sim = t1 + t2 + t3 + t4

            # Apply mask (set invalid to min)
            sim = tl.where(j_mask, sim, -1.0e9)

            # Max reduction across block
            block_max = tl.max(sim, axis=0)
            if block_max > max_sim:
                max_sim = block_max

        total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


def roq_maxsim_8bit(queries_codes, queries_meta, docs_codes, docs_meta):
    """
    Args:
        queries_codes: (A, S, Dim) uint8
        queries_meta: (A, S, 4) float32 [scale, offset, sum, norm_sq]
        docs_codes: (B, T, Dim) uint8
        docs_meta: (B, T, 4) float32
    Returns:
        scores: (A, B) float32
    """
    A, S, Dim = queries_codes.shape
    B, T, Dim2 = docs_codes.shape
    assert Dim == Dim2

    scores = torch.empty((A, B), dtype=torch.float32, device=queries_codes.device)

    grid = (A, B)
    _roq_maxsim_8bit_kernel[grid](
        queries_codes, docs_codes,
        queries_meta, docs_meta,
        scores,
        queries_codes.stride(0), queries_codes.stride(1),
        docs_codes.stride(0), docs_codes.stride(1),
        scores.stride(0), scores.stride(1),
        A, B, S, T, Dim,
        BLOCK_Q=1,
        BLOCK_DIM=128 # Must be constexpr
    )
    return scores


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
    ],
    key=['n_d_tokens', 'dim']
)
@triton.jit
def _roq_maxsim_4bit_kernel(
    Q_CODES_PTR, D_CODES_PTR,
    Q_META_PTR, D_META_PTR,
    Q_MASK_PTR, D_MASK_PTR,
    OUTPUT_PTR,
    q_batch_stride, q_token_stride,
    d_batch_stride, d_token_stride,
    q_meta_token_stride, d_meta_token_stride,
    q_mask_batch_stride, q_mask_token_stride,
    d_mask_batch_stride, d_mask_token_stride,
    output_q_stride, output_d_stride,
    n_queries, n_docs,
    n_q_tokens, n_d_tokens,
    dim, # original dim
    n_bytes: tl.constexpr, # dim / 2
    BLOCK_DIM: tl.constexpr, # unused? we iterate bytes
    BLOCK_D: tl.constexpr
):
    """
    4-bit RoQ MaxSim Kernel.
    Packed: 2 values per byte (High, Low).
    """
    q_idx = tl.program_id(0)
    d_idx = tl.program_id(1)

    q_ptr_base = Q_CODES_PTR + q_idx * q_batch_stride
    d_ptr_base = D_CODES_PTR + d_idx * d_batch_stride

    q_meta_ptr_base = Q_META_PTR + q_idx * n_q_tokens * q_meta_token_stride
    d_meta_ptr_base = D_META_PTR + d_idx * n_d_tokens * d_meta_token_stride

    total_score = 0.0

    # Offsets for bytes loop
    byte_offsets = tl.arange(0, BLOCK_DIM)

    for i in range(n_q_tokens):
        q_token_active = tl.load(
            Q_MASK_PTR + q_idx * q_mask_batch_stride + i * q_mask_token_stride
        ) > 0
        if q_token_active:
            # Load Q Meta
            q_meta_ptr = q_meta_ptr_base + i * q_meta_token_stride

            q_scale = tl.load(q_meta_ptr + 0)
            q_offset = tl.load(q_meta_ptr + 1)
            q_sum = tl.load(q_meta_ptr + 2)
            q_affine_offset = dim * q_offset + q_scale * q_sum

            # Load Q Vector (Packed Bytes)
            q_off = i * q_token_stride
            q_ptr = q_ptr_base + q_off
            q_vec_bytes = tl.load(q_ptr + byte_offsets, mask=byte_offsets < n_bytes, other=0)

            # Keep nibble arithmetic in int32, then cast once after reduction.
            q_high = (q_vec_bytes >> 4).to(tl.int32)
            q_low = (q_vec_bytes & 0x0F).to(tl.int32)

            max_sim = -1.0e9

            for j_start in range(0, n_d_tokens, BLOCK_D):
                j_offsets = j_start + tl.arange(0, BLOCK_D)
                j_mask = j_offsets < n_d_tokens
                d_meta_ptrs = d_meta_ptr_base + j_offsets * d_meta_token_stride
                d_scale = tl.load(d_meta_ptrs + 0, mask=j_mask, other=0.0)
                d_offset = tl.load(d_meta_ptrs + 1, mask=j_mask, other=0.0)
                d_sum = tl.load(d_meta_ptrs + 2, mask=j_mask, other=0.0)

                d_ptr_offs = j_offsets[:, None] * d_token_stride + byte_offsets[None, :]
                d_vecs_bytes = tl.load(
                    d_ptr_base + d_ptr_offs,
                    mask=(j_mask[:, None] & (byte_offsets[None, :] < n_bytes)),
                    other=0,
                )
                d_token_active = tl.load(
                    D_MASK_PTR + d_idx * d_mask_batch_stride + j_offsets * d_mask_token_stride,
                    mask=j_mask,
                    other=0,
                ) > 0

                d_high = (d_vecs_bytes >> 4).to(tl.int32)
                d_low = (d_vecs_bytes & 0x0F).to(tl.int32)

                dot_val = tl.sum(q_high[None, :] * d_high + q_low[None, :] * d_low, axis=1).to(tl.float32)

                est_dot = (d_offset * q_affine_offset) + (d_scale * (q_offset * d_sum + q_scale * dot_val))
                sim = tl.where(j_mask & d_token_active, est_dot, -1.0e9)
                block_max = tl.max(sim, axis=0)
                if block_max > max_sim:
                    max_sim = block_max

            total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32, 'DOCS_PER_PROG': 1}, num_warps=2),
        triton.Config({'BLOCK_D': 64, 'DOCS_PER_PROG': 1}, num_warps=4),
        triton.Config({'BLOCK_D': 128, 'DOCS_PER_PROG': 1}, num_warps=4),
        triton.Config({'BLOCK_D': 256, 'DOCS_PER_PROG': 1}, num_warps=8),
        triton.Config({'BLOCK_D': 64, 'DOCS_PER_PROG': 4}, num_warps=4),
        triton.Config({'BLOCK_D': 128, 'DOCS_PER_PROG': 4}, num_warps=4),
        triton.Config({'BLOCK_D': 128, 'DOCS_PER_PROG': 8}, num_warps=4),
    ],
    key=['n_d_tokens', 'n_bytes', 'n_docs'],
)
@triton.jit
def _roq_maxsim_4bit_v2_kernel(
    Q_CODES_PTR, D_CODES_PTR,
    Q_META_PTR, D_META_PTR,
    D_MASK_PTR,
    OUTPUT_PTR,
    q_batch_stride, q_token_stride,
    d_batch_stride, d_token_stride,
    d_meta_token_stride,
    d_mask_batch_stride, d_mask_token_stride,
    output_q_stride, output_d_stride,
    n_queries, n_docs,
    n_q_tokens, n_d_tokens,
    dim,
    n_bytes: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_D_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOCS_PER_PROG: tl.constexpr,
):
    """
    Optimized 4-bit RoQ MaxSim kernel.
    - Document tiling: each program processes DOCS_PER_PROG documents
    - FP32 dot path with fused nibble unpack
    - Optional document mask (HAS_D_MASK constexpr eliminates branch when unused)
    - Wider autotune configs including BLOCK_D=256
    - Query codes loaded once per query token, reused across all docs in tile
    """
    q_idx = tl.program_id(0)
    d_tile_idx = tl.program_id(1)

    q_ptr_base = Q_CODES_PTR + q_idx * q_batch_stride
    q_meta_ptr_base = Q_META_PTR + q_idx * n_q_tokens * 4

    byte_offsets = tl.arange(0, BLOCK_DIM)

    for doc_in_tile in range(DOCS_PER_PROG):
        d_idx = d_tile_idx * DOCS_PER_PROG + doc_in_tile
        if d_idx < n_docs:
            d_ptr_base = D_CODES_PTR + d_idx * d_batch_stride
            d_meta_ptr_base = D_META_PTR + d_idx * n_d_tokens * d_meta_token_stride

            total_score = 0.0

            for i in range(n_q_tokens):
                q_meta_ptr = q_meta_ptr_base + i * 4
                q_scale = tl.load(q_meta_ptr + 0)
                q_offset = tl.load(q_meta_ptr + 1)
                q_sum = tl.load(q_meta_ptr + 2)
                q_affine_offset = dim * q_offset + q_scale * q_sum

                q_vec_bytes = tl.load(q_ptr_base + i * q_token_stride + byte_offsets,
                                      mask=byte_offsets < n_bytes, other=0)
                q_high = (q_vec_bytes >> 4).to(tl.float32)
                q_low = (q_vec_bytes & 0x0F).to(tl.float32)

                max_sim = -1.0e9

                for j_start in range(0, n_d_tokens, BLOCK_D):
                    j_offsets = j_start + tl.arange(0, BLOCK_D)
                    j_mask = j_offsets < n_d_tokens

                    d_meta_ptrs = d_meta_ptr_base + j_offsets * d_meta_token_stride
                    d_scale = tl.load(d_meta_ptrs + 0, mask=j_mask, other=0.0)
                    d_offset = tl.load(d_meta_ptrs + 1, mask=j_mask, other=0.0)
                    d_sum = tl.load(d_meta_ptrs + 2, mask=j_mask, other=0.0)

                    d_ptr_offs = j_offsets[:, None] * d_token_stride + byte_offsets[None, :]
                    d_vecs_bytes = tl.load(
                        d_ptr_base + d_ptr_offs,
                        mask=(j_mask[:, None] & (byte_offsets[None, :] < n_bytes)),
                        other=0,
                    )
                    d_high = (d_vecs_bytes >> 4).to(tl.float32)
                    d_low = (d_vecs_bytes & 0x0F).to(tl.float32)

                    dot_val = tl.sum(q_high[None, :] * d_high + q_low[None, :] * d_low, axis=1)

                    est_dot = (d_offset * q_affine_offset) + (d_scale * (q_offset * d_sum + q_scale * dot_val))

                    if HAS_D_MASK:
                        d_token_active = tl.load(
                            D_MASK_PTR + d_idx * d_mask_batch_stride + j_offsets * d_mask_token_stride,
                            mask=j_mask, other=0,
                        ) > 0
                        sim = tl.where(j_mask & d_token_active, est_dot, -1.0e9)
                    else:
                        sim = tl.where(j_mask, est_dot, -1.0e9)

                    block_max = tl.max(sim, axis=0)
                    if block_max > max_sim:
                        max_sim = block_max

                total_score += max_sim

            out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
            tl.store(out_ptr, total_score)


def roq_maxsim_4bit(queries_codes, queries_meta, docs_codes, docs_meta, queries_mask=None, documents_mask=None):
    """
    Args:
        queries_codes: (A, S, NB) uint8 (Packed)
        queries_meta: (A, S, 4) float32
        docs_codes: (B, T, NB) uint8 (Packed)
        docs_meta: (B, T, 4) float32
    """
    A, S, NB = queries_codes.shape
    B, T, NB2 = docs_codes.shape
    assert NB == NB2

    scores = torch.empty((A, B), dtype=torch.float32, device=queries_codes.device)
    dim = NB * 2
    block_dim = 32
    while block_dim < NB:
        block_dim *= 2

    has_q_mask = queries_mask is not None
    has_d_mask = documents_mask is not None

    if not has_q_mask:
        if has_d_mask:
            documents_mask = documents_mask.to(device=docs_codes.device, dtype=torch.float32)
        else:
            documents_mask = torch.empty((1, 1), dtype=torch.float32, device=docs_codes.device)

        grid = lambda META: (A, (B + META['DOCS_PER_PROG'] - 1) // META['DOCS_PER_PROG'])
        _roq_maxsim_4bit_v2_kernel[grid](
            Q_CODES_PTR=queries_codes,
            D_CODES_PTR=docs_codes,
            Q_META_PTR=queries_meta,
            D_META_PTR=docs_meta,
            D_MASK_PTR=documents_mask,
            OUTPUT_PTR=scores,
            q_batch_stride=queries_codes.stride(0),
            q_token_stride=queries_codes.stride(1),
            d_batch_stride=docs_codes.stride(0),
            d_token_stride=docs_codes.stride(1),
            d_meta_token_stride=docs_meta.stride(1),
            d_mask_batch_stride=documents_mask.stride(0),
            d_mask_token_stride=documents_mask.stride(1),
            output_q_stride=scores.stride(0),
            output_d_stride=scores.stride(1),
            n_queries=A,
            n_docs=B,
            n_q_tokens=S,
            n_d_tokens=T,
            dim=dim,
            n_bytes=NB,
            BLOCK_DIM=block_dim,
            HAS_D_MASK=has_d_mask,
        )
        return scores

    queries_mask = queries_mask.to(device=queries_codes.device, dtype=torch.float32)
    if documents_mask is None:
        documents_mask = torch.ones((B, T), dtype=torch.float32, device=docs_codes.device)
    else:
        documents_mask = documents_mask.to(device=docs_codes.device, dtype=torch.float32)

    grid = (A, B)
    _roq_maxsim_4bit_kernel[grid](
        Q_CODES_PTR=queries_codes,
        D_CODES_PTR=docs_codes,
        Q_META_PTR=queries_meta,
        D_META_PTR=docs_meta,
        Q_MASK_PTR=queries_mask,
        D_MASK_PTR=documents_mask,
        OUTPUT_PTR=scores,
        q_batch_stride=queries_codes.stride(0),
        q_token_stride=queries_codes.stride(1),
        d_batch_stride=docs_codes.stride(0),
        d_token_stride=docs_codes.stride(1),
        q_meta_token_stride=queries_meta.stride(1),
        d_meta_token_stride=docs_meta.stride(1),
        q_mask_batch_stride=queries_mask.stride(0),
        q_mask_token_stride=queries_mask.stride(1),
        d_mask_batch_stride=documents_mask.stride(0),
        d_mask_token_stride=documents_mask.stride(1),
        output_q_stride=scores.stride(0),
        output_d_stride=scores.stride(1),
        n_queries=A,
        n_docs=B,
        n_q_tokens=S,
        n_d_tokens=T,
        dim=dim,
        n_bytes=NB,
        BLOCK_DIM=block_dim,
    )
    return scores


@triton.jit
def _roq_maxsim_2bit_kernel(
    Q_PTR, D_PTR,
    Q_META_PTR, D_META_PTR,
    Q_MASK_PTR, D_MASK_PTR,
    OUTPUT_PTR,
    stride_q_batch, stride_q_tokens,
    stride_d_batch, stride_d_tokens,
    q_meta_token_stride, d_meta_token_stride,
    q_mask_batch_stride, q_mask_token_stride,
    d_mask_batch_stride, d_mask_token_stride,
    output_q_stride, output_d_stride,
    A, B,
    n_q_tokens, n_d_tokens,
    dim: tl.constexpr, # full dim
    n_bytes: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr = 128
):
    q_idx = tl.program_id(0) # query batch idx
    d_idx = tl.program_id(1) # doc batch idx

    # Pointers
    q_ptr_base = Q_PTR + q_idx * stride_q_batch
    d_ptr_base = D_PTR + d_idx * stride_d_batch

    q_meta_ptr_base = Q_META_PTR + q_idx * n_q_tokens * q_meta_token_stride
    d_meta_ptr_base = D_META_PTR + d_idx * n_d_tokens * d_meta_token_stride

    total_score = 0.0

    byte_offsets = tl.arange(0, BLOCK_DIM) # n_bytes is small (dim/4)
    # Actually we just load everything

    for i in range(n_q_tokens):
        q_token_active = tl.load(
            Q_MASK_PTR + q_idx * q_mask_batch_stride + i * q_mask_token_stride
        ) > 0
        if q_token_active:
            q_meta_ptrs = q_meta_ptr_base + i * q_meta_token_stride
            q_scale = tl.load(q_meta_ptrs + 0)
            q_offset = tl.load(q_meta_ptrs + 1)
            q_sum = tl.load(q_meta_ptrs + 2)

            q_vec_bytes = tl.load(q_ptr_base + i * stride_q_tokens + byte_offsets, mask=byte_offsets < n_bytes, other=0)
            q0 = (q_vec_bytes >> 6).to(tl.float32)
            q1 = ((q_vec_bytes >> 4) & 3).to(tl.float32)
            q2 = ((q_vec_bytes >> 2) & 3).to(tl.float32)
            q3 = (q_vec_bytes & 3).to(tl.float32)

            max_sim = -1.0e9

            for j_start in range(0, n_d_tokens, BLOCK_D):
                j_offsets = j_start + tl.arange(0, BLOCK_D)
                j_mask = j_offsets < n_d_tokens

                d_meta_ptrs = d_meta_ptr_base + j_offsets * d_meta_token_stride
                d_scale = tl.load(d_meta_ptrs + 0, mask=j_mask, other=0.0)
                d_offset = tl.load(d_meta_ptrs + 1, mask=j_mask, other=0.0)
                d_sum = tl.load(d_meta_ptrs + 2, mask=j_mask, other=0.0)

                d_ptr_offs = j_offsets[:, None] * stride_d_tokens + byte_offsets[None, :]
                d_vecs_bytes = tl.load(
                    d_ptr_base + d_ptr_offs,
                    mask=(j_mask[:, None] & (byte_offsets[None, :] < n_bytes)),
                    other=0,
                )
                d_token_active = tl.load(
                    D_MASK_PTR + d_idx * d_mask_batch_stride + j_offsets * d_mask_token_stride,
                    mask=j_mask,
                    other=0,
                ) > 0

                d0 = (d_vecs_bytes >> 6).to(tl.float32)
                d1 = ((d_vecs_bytes >> 4) & 3).to(tl.float32)
                d2 = ((d_vecs_bytes >> 2) & 3).to(tl.float32)
                d3 = (d_vecs_bytes & 3).to(tl.float32)

                dot_val = tl.sum(
                    q0[None, :] * d0
                    + q1[None, :] * d1
                    + q2[None, :] * d2
                    + q3[None, :] * d3,
                    axis=1,
                )

                term1 = dim * d_offset * q_offset
                term2 = d_offset * q_scale * q_sum
                term3 = q_offset * d_scale * d_sum
                term4 = d_scale * q_scale * dot_val

                est_dot = term1 + term2 + term3 + term4
                sim = tl.where(j_mask & d_token_active, est_dot, -1.0e9)
                block_max = tl.max(sim, axis=0)
                if block_max > max_sim:
                    max_sim = block_max

            total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


def roq_maxsim_2bit(queries_codes, queries_meta, docs_codes, docs_meta, queries_mask=None, documents_mask=None):
    scores = torch.empty((queries_codes.shape[0], docs_codes.shape[0]), dtype=torch.float32, device=queries_codes.device)
    A, S, NB = queries_codes.shape
    B, T, NB2 = docs_codes.shape
    assert NB == NB2
    if queries_mask is None:
        queries_mask = torch.ones((A, S), dtype=torch.float32, device=queries_codes.device)
    else:
        queries_mask = queries_mask.to(device=queries_codes.device, dtype=torch.float32)
    if documents_mask is None:
        documents_mask = torch.ones((B, T), dtype=torch.float32, device=docs_codes.device)
    else:
        documents_mask = documents_mask.to(device=docs_codes.device, dtype=torch.float32)
    dim = NB * 4 #    dim = NB * 8

    # BLOCK_DIM must cover n_bytes (NB)
    # Find next power of 2 >= NB
    block_dim = 32
    while block_dim < NB:
        block_dim *= 2

    grid = (A, B)
    _roq_maxsim_2bit_kernel[grid]( # Changed from _roq_maxsim_1bit_kernel to _roq_maxsim_2bit_kernel to match context
        queries_codes, docs_codes, # Changed from queries, docs to queries_codes, docs_codes to match function signature
        queries_meta, docs_meta, # Added back queries_meta, docs_meta
        queries_mask, documents_mask,
        scores,
        queries_codes.stride(0), queries_codes.stride(1),
        docs_codes.stride(0), docs_codes.stride(1),
        queries_meta.stride(1), docs_meta.stride(1),
        queries_mask.stride(0), queries_mask.stride(1),
        documents_mask.stride(0), documents_mask.stride(1),
        scores.stride(0), scores.stride(1),
        A, B,
        S, T,
        dim,
        NB,
        BLOCK_DIM=block_dim,
        BLOCK_D=128
    )
    return scores
