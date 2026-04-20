"""
MaxSim scorer: thin wrapper over voyager-index Triton kernels.

Supports two paths:
- score_all_docs_topk: batched single-call scoring (fast path)
- score_shards_and_topk: per-shard scoring with GPU top-k merge (legacy)
- score_and_topk: legacy padded-tensor interface
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch

from .profiler import Timer

if TYPE_CHECKING:
    from .shard_store import ShardStore

logger = logging.getLogger(__name__)

_maxsim_fn = None
_warmup_done_lock = __import__("threading").Lock()
_warmup_done: set = set()


def _get_maxsim():
    global _maxsim_fn
    if _maxsim_fn is not None:
        return _maxsim_fn
    try:
        from voyager_index._internal.kernels.maxsim import fast_colbert_scores

        _maxsim_fn = fast_colbert_scores
        logger.info("Using voyager-index Triton MaxSim kernel")
    except ImportError:
        _maxsim_fn = _fallback_maxsim
        logger.warning("Triton MaxSim unavailable, using PyTorch fallback")
    return _maxsim_fn


def warmup_maxsim(dim: int, doc_token_counts: List[int], device: str = "cuda") -> None:
    """Pre-warm Triton autotune for the exact (S, T, H) shapes that will appear.

    Triton autotune keys on (NUM_Q_TOKENS, NUM_D_TOKENS, EMBED_DIM) only — B
    (batch size) is NOT a key.  We use B=4 for minimal memory and warm only
    the unique (S, T, H) combinations.
    """
    global _warmup_done
    if not torch.cuda.is_available() and "cuda" in device:
        return
    maxsim = _get_maxsim()
    dev = torch.device(device)

    q_tokens_list = [32, 64, 128, 256]
    d_tokens_set: set = set()
    for tc in doc_token_counts:
        d_tokens_set.add(_next_pow2(tc, 32))
    if not d_tokens_set:
        d_tokens_set = {32, 64, 128, 256, 512, 1024, 2048}

    n_warmup_docs = 4
    for qt in q_tokens_list:
        for dt in sorted(d_tokens_set):
            key = (qt, dt, dim)
            with _warmup_done_lock:
                if key in _warmup_done:
                    continue
            q = torch.zeros(1, qt, dim, dtype=torch.float16, device=dev)
            d = torch.zeros(n_warmup_docs, dt, dim, dtype=torch.float16, device=dev)
            m = torch.ones(n_warmup_docs, dt, dtype=torch.float32, device=dev)
            maxsim(q, d, documents_mask=m)
            with _warmup_done_lock:
                _warmup_done.add(key)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info("MaxSim kernel warmup done for %d (S,T,H) combos", len(_warmup_done))


def _next_pow2(v: int, minimum: int = 1) -> int:
    v = max(v, minimum)
    p = 1
    while p < v:
        p <<= 1
    return p


def _fallback_maxsim(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask=None,
    documents_mask=None,
    **kwargs,
) -> torch.Tensor:
    """Pure PyTorch MaxSim for environments without Triton."""
    Q = queries_embeddings.float()
    D = documents_embeddings.float()
    sim = torch.einsum("ash,bth->abst", Q, D)
    if documents_mask is not None:
        dm = documents_mask.bool()
        sim = sim.masked_fill(~dm[None, :, None, :], float("-inf"))
    max_sim = sim.max(dim=-1).values
    if queries_mask is not None:
        qm = queries_mask.float()
        max_sim = max_sim * qm[:, None, :]
    max_sim = torch.nan_to_num(max_sim, neginf=0.0)
    return max_sim.sum(dim=-1)


ShardChunk = Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]


# ------------------------------------------------------------------
# Centroid proxy pre-scoring (Track C optimisation)
# ------------------------------------------------------------------


def proxy_score_candidates(
    query: torch.Tensor,
    doc_means: torch.Tensor,
    candidate_doc_ids: List[int],
    doc_id_to_idx: Dict[int, int],
    n_full_scores: int,
    device: torch.device = None,
) -> List[int]:
    """Cheap proxy scoring to prune candidates before exact MaxSim.

    Uses mean-pooled document embeddings (one vector per doc) as a proxy.
    Score = max over query tokens of (q_token dot d_mean).  This is orders
    of magnitude cheaper than full MaxSim because each doc is one vector
    instead of ~120 token vectors.

    Returns the top *n_full_scores* doc IDs from proxy-scored candidates,
    plus any candidate IDs that could not be proxy-scored (e.g. newly added
    docs whose means are not yet indexed).  The output may therefore exceed
    *n_full_scores* when unknown candidates exist.
    """
    if not candidate_doc_ids or doc_means is None:
        return candidate_doc_ids

    if len(candidate_doc_ids) <= n_full_scores:
        return candidate_doc_ids

    if device is None:
        device = doc_means.device

    valid: List[Tuple[int, int]] = []
    unknown: List[int] = []
    for did in candidate_doc_ids:
        idx = doc_id_to_idx.get(did)
        if idx is not None:
            valid.append((did, idx))
        else:
            unknown.append(did)

    if not valid:
        return candidate_doc_ids

    # Budget for proxy-scored docs — reserve slots for unknowns that
    # cannot be scored (they pass through unconditionally).
    budget = max(n_full_scores - len(unknown), 1)
    if len(valid) <= budget:
        return [did for did, _ in valid] + unknown

    v_ids, v_indices = zip(*valid)
    idx_t = torch.tensor(v_indices, dtype=torch.long, device=device)
    D_proxy = doc_means[idx_t]  # (N, dim)

    q = query.to(device, dtype=torch.float16)
    if q.dim() == 3:
        q = q.squeeze(0)
    # (S, dim) @ (dim, N) → (S, N), max over S → (N,)
    scores = (q.float() @ D_proxy.float().T).max(dim=0).values

    top_n = min(budget, len(v_ids))
    _, top_idx = scores.topk(top_n)
    result = [v_ids[i] for i in top_idx.cpu().tolist()]
    result.extend(unknown)
    return result


# ------------------------------------------------------------------
# Batched scoring (Fix 1) — single kernel call for all docs
# ------------------------------------------------------------------


def score_all_docs_topk(
    query: torch.Tensor,
    shard_chunks: List[ShardChunk],
    k: int = 10,
    device: torch.device = None,
    quantization_mode: str = "",
    variable_length_strategy: str = "bucketed",
    return_stats: bool = False,
) -> Tuple[List[int], List[float]] | Tuple[List[int], List[float], dict]:
    """Score all fetched docs in one kernel call.

    Concatenates per-shard tensors (not per-doc slices) to avoid O(n_docs)
    torch.cat overhead, then reshapes for the Triton MaxSim kernel.

    When *quantization_mode* is set (e.g. ``"int8"``), the flag is forwarded
    to the Triton kernel so it can use reduced-precision accumulation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    maxsim = _get_maxsim()

    q = query.to(device, dtype=torch.float16)
    if q.dim() == 2:
        q = q.unsqueeze(0)

    shard_embs: List[torch.Tensor] = []
    all_doc_ids: List[int] = []
    tok_per_doc: int = 0
    is_uniform = True

    for flat_emb, offsets, doc_ids in shard_chunks:
        if not doc_ids:
            continue
        shard_embs.append(flat_emb)
        all_doc_ids.extend(doc_ids)
        for s, e in offsets:
            tlen = e - s
            if tok_per_doc == 0:
                tok_per_doc = tlen
            elif tlen != tok_per_doc:
                is_uniform = False

    if not all_doc_ids:
        empty_stats = {
            "score_mode": "none",
            "prepare_ms": 0.0,
            "h2d_ms": 0.0,
            "maxsim_ms": 0.0,
            "topk_ms": 0.0,
            "exact_ms": 0.0,
            "n_buckets": 0,
        }
        if return_stats:
            return [], [], empty_stats
        return [], []

    n_docs = len(all_doc_ids)
    dim = shard_embs[0].shape[1]

    quant_kwargs: dict = {}
    if quantization_mode:
        quant_kwargs["use_quantization"] = True
        quant_kwargs["quantization_mode"] = quantization_mode

    sync_cuda = device.type == "cuda"
    stats = {
        "score_mode": "uniform" if is_uniform and tok_per_doc > 0 else variable_length_strategy,
        "prepare_ms": 0.0,
        "h2d_ms": 0.0,
        "maxsim_ms": 0.0,
        "topk_ms": 0.0,
        "exact_ms": 0.0,
        "n_buckets": 0,
    }

    def _finalize_scores(scores: torch.Tensor, ordered_doc_ids: List[int]) -> Tuple[List[int], List[float]]:
        final_k = min(k, len(ordered_doc_ids))
        if final_k <= 0:
            return [], []
        with Timer(sync_cuda=sync_cuda) as t_topk:
            top_sc, top_idx = scores.topk(final_k)
        stats["topk_ms"] += t_topk.elapsed_ms
        idx_list = top_idx.cpu().tolist()
        return [ordered_doc_ids[i] for i in idx_list], top_sc.cpu().tolist()

    with Timer(sync_cuda=sync_cuda) as t_exact:
        if is_uniform and tok_per_doc > 0:
            flat = torch.cat(shard_embs, dim=0)
            if device.type == "cuda":
                with Timer(sync_cuda=True) as t_h2d:
                    D_dev = flat.view(n_docs, tok_per_doc, dim).to(device, dtype=torch.float16)
                stats["h2d_ms"] += t_h2d.elapsed_ms
            else:
                D_dev = flat.view(n_docs, tok_per_doc, dim).to(device, dtype=torch.float16)
            with Timer(sync_cuda=sync_cuda) as t_maxsim:
                scores = maxsim(
                    queries_embeddings=q,
                    documents_embeddings=D_dev,
                    documents_mask=None,
                    **quant_kwargs,
                ).squeeze(0)
            stats["maxsim_ms"] += t_maxsim.elapsed_ms
            result_ids, result_scores = _finalize_scores(scores, all_doc_ids)
        elif variable_length_strategy == "padded":
            lengths: List[int] = []
            all_slices: List[torch.Tensor] = []
            for flat_emb, offsets, doc_ids in shard_chunks:
                if not doc_ids:
                    continue
                for s, e in offsets:
                    all_slices.append(flat_emb[s:e])
                    lengths.append(e - s)
            max_tok = max(lengths)
            with Timer(sync_cuda=False) as t_prepare:
                D = np.zeros((n_docs, max_tok, dim), dtype=np.float16)
                M = np.zeros((n_docs, max_tok), dtype=np.float32)
                for i, sl in enumerate(all_slices):
                    tok = sl.shape[0]
                    D[i, :tok] = sl.cpu().numpy()
                    M[i, :tok] = 1.0
            stats["prepare_ms"] += t_prepare.elapsed_ms
            if device.type == "cuda":
                with Timer(sync_cuda=True) as t_h2d:
                    D_dev = torch.from_numpy(D).to(device)
                    M_dev = torch.from_numpy(M).to(device)
                stats["h2d_ms"] += t_h2d.elapsed_ms
            else:
                D_dev = torch.from_numpy(D).to(device)
                M_dev = torch.from_numpy(M).to(device)
            with Timer(sync_cuda=sync_cuda) as t_maxsim:
                scores = maxsim(
                    queries_embeddings=q,
                    documents_embeddings=D_dev,
                    documents_mask=M_dev,
                    **quant_kwargs,
                ).squeeze(0)
            stats["maxsim_ms"] += t_maxsim.elapsed_ms
            result_ids, result_scores = _finalize_scores(scores, all_doc_ids)
        else:
            buckets: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
            for flat_emb, offsets, doc_ids in shard_chunks:
                if not doc_ids:
                    continue
                for doc_id, (s, e) in zip(doc_ids, offsets):
                    tok = int(e - s)
                    bucket_key = _next_pow2(tok, 32)
                    buckets.setdefault(bucket_key, []).append((doc_id, flat_emb[s:e]))

            stats["n_buckets"] = len(buckets)
            bucket_scores: List[torch.Tensor] = []
            bucket_doc_ids: List[int] = []

            for bucket_key in sorted(buckets):
                entries = buckets[bucket_key]
                with Timer(sync_cuda=False) as t_prepare:
                    pieces = [piece for _doc_id, piece in entries]
                    doc_ids = [doc_id for doc_id, _piece in entries]
                    offsets: List[Tuple[int, int]] = []
                    pos = 0
                    for piece in pieces:
                        tok = int(piece.shape[0])
                        offsets.append((pos, pos + tok))
                        pos += tok
                    flat = torch.cat(pieces, dim=0)
                stats["prepare_ms"] += t_prepare.elapsed_ms

                if device.type == "cuda" and flat.device != device:
                    with Timer(sync_cuda=True) as t_h2d:
                        flat = flat.to(device, dtype=torch.float16, non_blocking=True)
                    stats["h2d_ms"] += t_h2d.elapsed_ms
                elif flat.device != device:
                    flat = flat.to(device, dtype=torch.float16)

                with Timer(sync_cuda=sync_cuda) as t_prepare:
                    doc_emb, doc_mask = _pad_flat_embeddings(flat, offsets, device)
                stats["prepare_ms"] += t_prepare.elapsed_ms

                with Timer(sync_cuda=sync_cuda) as t_maxsim:
                    scores = maxsim(
                        queries_embeddings=q,
                        documents_embeddings=doc_emb,
                        documents_mask=None if doc_mask.shape[1] and bool(doc_mask[:, -1].all().item()) else doc_mask,
                        **quant_kwargs,
                    ).squeeze(0)
                stats["maxsim_ms"] += t_maxsim.elapsed_ms
                bucket_scores.append(scores)
                bucket_doc_ids.extend(doc_ids)

            if bucket_scores:
                scores = torch.cat(bucket_scores, dim=0)
                result_ids, result_scores = _finalize_scores(scores, bucket_doc_ids)
            else:
                result_ids, result_scores = [], []

    stats["exact_ms"] = t_exact.elapsed_ms
    if return_stats:
        return result_ids, result_scores, stats
    return result_ids, result_scores


# ------------------------------------------------------------------
# Pre-loaded GPU corpus (GEM pattern)
# ------------------------------------------------------------------


class PreloadedGpuCorpus:
    """Pre-pad all docs into a contiguous GPU tensor once at startup.

    At query time, D_gpu[candidate_indices] is a zero-allocation GPU gather.
    """

    def __init__(
        self,
        doc_vecs: list,
        doc_ids: List[int],
        dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        n_docs = len(doc_vecs)
        max_tok = max(v.shape[0] for v in doc_vecs) if doc_vecs else 1
        logger.info(
            "PreloadedGpuCorpus: %d docs, max_tok=%d, dim=%d, %.1f GB %s",
            n_docs,
            max_tok,
            dim,
            n_docs * max_tok * dim * (2 if dtype == torch.float16 else 4) / 1e9,
            dtype,
        )
        self.D = torch.zeros((n_docs, max_tok, dim), dtype=dtype, device=device)
        self.M = torch.zeros((n_docs, max_tok), dtype=torch.float32, device=device)
        self.doc_ids = list(doc_ids)
        self.doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        self.max_tok = max_tok
        self.dim = dim
        self._device = device

        for i, v in enumerate(doc_vecs):
            tok = v.shape[0]
            self.D[i, :tok] = torch.from_numpy(v).to(dtype)
            self.M[i, :tok] = 1.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("PreloadedGpuCorpus ready on %s", device)

    def refresh(self, doc_vecs: list, doc_ids: List[int]) -> None:
        """Rebuild GPU tensors from a new sealed snapshot."""
        n_docs = len(doc_vecs)
        max_tok = max(v.shape[0] for v in doc_vecs) if doc_vecs else 1
        self.D = torch.zeros((n_docs, max_tok, self.dim), dtype=self.D.dtype, device=self._device)
        self.M = torch.zeros((n_docs, max_tok), dtype=torch.float32, device=self._device)
        self.doc_ids = list(doc_ids)
        self.doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        self.max_tok = max_tok
        for i, v in enumerate(doc_vecs):
            tok = v.shape[0]
            self.D[i, :tok] = torch.from_numpy(v).to(self.D.dtype)
            self.M[i, :tok] = 1.0

    def score_candidates(
        self,
        query: torch.Tensor,
        candidate_doc_ids: List[int],
        k: int = 10,
        return_stats: bool = False,
    ) -> Tuple[List[int], List[float]] | Tuple[List[int], List[float], dict]:
        maxsim = _get_maxsim()
        q = query.to(self._device, dtype=torch.float16)
        if q.dim() == 2:
            q = q.unsqueeze(0)

        valid_ids = [did for did in candidate_doc_ids if did in self.doc_id_to_idx]
        if not valid_ids:
            empty_stats = {
                "score_mode": "gpu_corpus",
                "prepare_ms": 0.0,
                "h2d_ms": 0.0,
                "maxsim_ms": 0.0,
                "topk_ms": 0.0,
                "exact_ms": 0.0,
                "n_buckets": 0,
            }
            if return_stats:
                return [], [], empty_stats
            return [], []

        indices = torch.tensor(
            [self.doc_id_to_idx[did] for did in valid_ids],
            dtype=torch.long,
            device=self._device,
        )

        stats = {
            "score_mode": "gpu_corpus",
            "prepare_ms": 0.0,
            "h2d_ms": 0.0,
            "maxsim_ms": 0.0,
            "topk_ms": 0.0,
            "exact_ms": 0.0,
            "n_buckets": 1,
        }

        with Timer(sync_cuda=True) as t_exact:
            with Timer(sync_cuda=True) as t_prepare:
                D_slice = self.D[indices]
                M_slice = self.M[indices]
                has_padding = (M_slice[:, -1] == 0).any()
                if has_padding:
                    actual_max = int(M_slice.sum(dim=1).max().item())
                    D_slice = D_slice[:, :actual_max].contiguous()
                    M_slice = M_slice[:, :actual_max].contiguous()
            stats["prepare_ms"] += t_prepare.elapsed_ms

            with Timer(sync_cuda=True) as t_maxsim:
                scores = maxsim(
                    queries_embeddings=q,
                    documents_embeddings=D_slice,
                    documents_mask=M_slice if has_padding else None,
                ).squeeze(0)
            stats["maxsim_ms"] = t_maxsim.elapsed_ms

            n = len(valid_ids)
            final_k = min(k, n)
            with Timer(sync_cuda=True) as t_topk:
                top_sc, top_idx = scores.topk(final_k)
            stats["topk_ms"] = t_topk.elapsed_ms

        stats["exact_ms"] = t_exact.elapsed_ms
        idx_list = top_idx.cpu().tolist()
        result = ([valid_ids[i] for i in idx_list], top_sc.cpu().tolist())
        if return_stats:
            return result[0], result[1], stats
        return result

    @staticmethod
    def fits_on_gpu(n_docs: int, max_tok: int, dim: int, dtype=torch.float16) -> bool:
        dtype_itemsize = dtype.itemsize
        bytes_needed = int(n_docs * max_tok * dim * dtype_itemsize)
        bytes_needed += n_docs * max_tok * 4  # mask
        if not torch.cuda.is_available():
            return False
        free, _total = torch.cuda.mem_get_info()
        return bytes_needed < free * 0.85

    @staticmethod
    def estimate_gpu_bytes(n_docs: int, max_tok: int, dim: int, dtype=torch.float16) -> int:
        return int(n_docs * max_tok * dim * dtype.itemsize)

    @staticmethod
    def estimate_cpu_staging_bytes(n_docs: int, max_tok: int, dim: int, dtype=np.float32) -> int:
        return int(n_docs * max_tok * dim * np.dtype(dtype).itemsize)

    @staticmethod
    def estimate_streaming_cpu_bytes(
        chunk_docs: int,
        max_tok: int,
        dim: int,
        dtype=np.float16,
    ) -> int:
        emb_bytes = int(chunk_docs * max_tok * dim * np.dtype(dtype).itemsize)
        mask_bytes = int(chunk_docs * max_tok * np.dtype(np.float32).itemsize)
        return emb_bytes + mask_bytes

    @classmethod
    def suggest_streaming_chunk_docs(
        cls,
        max_tok: int,
        dim: int,
        dtype=np.float16,
        max_host_bytes: int = 256 * 1024**2,
    ) -> int:
        bytes_per_doc = cls.estimate_streaming_cpu_bytes(1, max_tok, dim, dtype=dtype)
        if bytes_per_doc <= 0:
            return 1
        return max(1, min(4096, int(max_host_bytes // bytes_per_doc)))

    @staticmethod
    def available_cpu_bytes() -> int | None:
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
        except (AttributeError, ValueError, OSError):
            return None

    @classmethod
    def fits_cpu_staging(
        cls,
        n_docs: int,
        max_tok: int,
        dim: int,
        dtype=np.float32,
        hard_cap_bytes: int = 8 * 1024**3,
    ) -> bool:
        bytes_needed = cls.estimate_cpu_staging_bytes(n_docs, max_tok, dim, dtype=dtype)
        if bytes_needed > hard_cap_bytes:
            return False
        avail = cls.available_cpu_bytes()
        if avail is None:
            return True
        # Leave generous headroom for Python object overhead and transient copies.
        return bytes_needed < avail * 0.45

    @classmethod
    def fits_cpu_streaming(
        cls,
        max_tok: int,
        dim: int,
        chunk_docs: int | None = None,
        dtype=np.float16,
        hard_cap_bytes: int = 1024**3,
        max_host_bytes: int = 256 * 1024**2,
    ) -> bool:
        if chunk_docs is None:
            chunk_docs = cls.suggest_streaming_chunk_docs(
                max_tok,
                dim,
                dtype=dtype,
                max_host_bytes=max_host_bytes,
            )
        bytes_needed = cls.estimate_streaming_cpu_bytes(
            chunk_docs,
            max_tok,
            dim,
            dtype=dtype,
        )
        if bytes_needed > hard_cap_bytes:
            return False
        avail = cls.available_cpu_bytes()
        if avail is None:
            return True
        return bytes_needed < avail * 0.45

    @classmethod
    def from_merged_streaming(
        cls,
        store: "ShardStore",
        dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        chunk_docs: int | None = None,
    ) -> "PreloadedGpuCorpus":
        doc_ids_arr, offsets, merged_dim, _total_tokens = store.load_merged_layout()
        if int(merged_dim) != int(dim):
            raise ValueError(f"merged dim {merged_dim} != expected dim {dim}")
        max_tok = int(np.max(offsets[1:] - offsets[:-1])) if len(doc_ids_arr) else 1
        if chunk_docs is None:
            chunk_docs = cls.suggest_streaming_chunk_docs(max_tok, dim)

        n_docs = int(len(doc_ids_arr))
        self = cls.__new__(cls)
        self.D = torch.zeros((n_docs, max_tok, dim), dtype=dtype, device=device)
        self.M = torch.zeros((n_docs, max_tok), dtype=torch.float32, device=device)
        self.doc_ids = [int(did) for did in doc_ids_arr.tolist()]
        self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
        self.max_tok = max_tok
        self.dim = dim
        self._device = device

        pos = 0
        for ids_chunk, emb_chunk, mask_chunk, chunk_max_tok, _global_max_tok in store.iter_merged_doc_chunks(
            chunk_docs
        ):
            n_chunk = int(len(ids_chunk))
            if n_chunk == 0:
                continue
            emb_t = torch.from_numpy(emb_chunk)
            mask_t = torch.from_numpy(mask_chunk)
            if device != "cpu":
                emb_t = emb_t.to(device=device, dtype=dtype, non_blocking=True)
                mask_t = mask_t.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                emb_t = emb_t.to(device=device, dtype=dtype)
                mask_t = mask_t.to(device=device, dtype=torch.float32)
            self.D[pos : pos + n_chunk, :chunk_max_tok] = emb_t[:, :chunk_max_tok]
            self.M[pos : pos + n_chunk, :chunk_max_tok] = mask_t[:, :chunk_max_tok]
            pos += n_chunk

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(
            "PreloadedGpuCorpus ready from merged mmap: %d docs, max_tok=%d, dim=%d on %s",
            n_docs,
            max_tok,
            dim,
            device,
        )
        return self


# ------------------------------------------------------------------
# ROQ 4-bit scoring (Step 5)
# ------------------------------------------------------------------

_roq_fn = None


def _get_roq_maxsim():
    global _roq_fn
    if _roq_fn is not None:
        return _roq_fn
    try:
        from voyager_index._internal.kernels.roq import roq_maxsim_4bit

        _roq_fn = roq_maxsim_4bit
        logger.info("Using voyager-index ROQ 4-bit MaxSim kernel")
    except ImportError:
        _roq_fn = None
        logger.warning("ROQ 4-bit MaxSim kernel unavailable")
    return _roq_fn


def score_roq4_topk(
    query_codes: torch.Tensor,
    query_meta: torch.Tensor,
    doc_codes: torch.Tensor,
    doc_meta: torch.Tensor,
    doc_ids: List[int],
    k: int = 10,
    documents_mask: torch.Tensor = None,
    device: torch.device = None,
) -> Tuple[List[int], List[float]]:
    """Score documents using ROQ 4-bit MaxSim kernel.

    query_codes: (1, S, NB) uint8
    query_meta:  (1, S, 4) float32
    doc_codes:   (N, T, NB) uint8
    doc_meta:    (N, T, 4) float32
    """
    roq = _get_roq_maxsim()
    if roq is None:
        return [], []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu" if isinstance(device, torch.device) else device == "cpu":
        logger.warning("ROQ4 scoring requires CUDA; falling back to empty results")
        return [], []

    qc = query_codes.to(device)
    qm = query_meta.to(device)
    dc = doc_codes.to(device)
    dm = doc_meta.to(device)

    n_docs = dc.shape[0]
    if n_docs == 0:
        return [], []

    kwargs = {}
    if documents_mask is not None:
        kwargs["documents_mask"] = documents_mask.to(device)

    scores = roq(qc, qm, dc, dm, **kwargs).squeeze(0)
    final_k = min(k, n_docs)
    top_sc, top_idx = scores.topk(final_k)
    idx_list = top_idx.cpu().tolist()
    return [doc_ids[i] for i in idx_list], top_sc.cpu().tolist()


# ------------------------------------------------------------------
# RROQ158 (Riemannian 1.58-bit) scoring
# ------------------------------------------------------------------

_rroq158_fn = None


def _get_rroq158_maxsim():
    global _rroq158_fn
    if _rroq158_fn is not None:
        return _rroq158_fn
    try:
        from voyager_index._internal.kernels.triton_roq_rroq158 import (
            roq_maxsim_rroq158,
        )

        _rroq158_fn = roq_maxsim_rroq158
        logger.info("Using voyager-index RROQ158 (1.58-bit) MaxSim kernel")
    except ImportError:
        _rroq158_fn = None
        logger.warning("RROQ158 MaxSim kernel unavailable")
    return _rroq158_fn


_rroq158_cpu_fn = None


def _get_rroq158_cpu_kernel():
    """Lazy-load the Rust SIMD CPU kernel from `latence_shard_engine`.

    Returns ``None`` if the wheel is unavailable, in which case
    ``score_rroq158_topk`` on CPU falls back to an empty result and the
    caller should reroute to a CPU FP16 path.
    """
    global _rroq158_cpu_fn
    if _rroq158_cpu_fn is not None:
        return _rroq158_cpu_fn
    try:
        import latence_shard_engine as _eng

        _rroq158_cpu_fn = getattr(_eng, "rroq158_score_batch", None)
        if _rroq158_cpu_fn is not None:
            logger.info("Using latence_shard_engine RROQ158 CPU SIMD kernel")
        else:
            logger.warning(
                "latence_shard_engine missing rroq158_score_batch; rebuild the "
                "Rust wheel from src/kernels/shard_engine to enable CPU rroq158"
            )
    except ImportError:
        _rroq158_cpu_fn = None
        logger.warning("latence_shard_engine unavailable; CPU rroq158 disabled")
    return _rroq158_cpu_fn


def _resolve_rroq158_n_threads(default: int | None = None) -> int | None:
    """Return ``n_threads`` argument for the Rust kernel, or None to use the
    rayon global pool.

    Heuristic:

    - ``VOYAGER_RROQ158_N_THREADS`` env var wins if set (``0`` or empty
      string disables — use the global pool).
    - Otherwise, if the python side runs the kernel from N parallel
      ``ThreadPoolExecutor`` workers (set via ``VOYAGER_RROQ158_N_WORKERS``),
      cap rayon to ``cpu_count() // n_workers`` so the total kernel
      thread count never exceeds physical cores. This avoids the
      1024-way over-subscription scenario (8 python workers × 128 default
      rayon threads on a 128-core box).
    - Default: leave rayon's global pool alone (``None``). The default
      pool is one thread per logical core, which is fine when the kernel
      is the sole consumer.
    """
    import os
    raw = os.environ.get("VOYAGER_RROQ158_N_THREADS")
    if raw is not None:
        try:
            n = int(raw)
            return n if n > 0 else None
        except ValueError:
            pass
    workers_raw = os.environ.get("VOYAGER_RROQ158_N_WORKERS")
    if workers_raw is not None:
        try:
            n_workers = max(1, int(workers_raw))
            cpu = os.cpu_count() or 1
            return max(1, cpu // n_workers)
        except ValueError:
            pass
    return default


def _score_rroq158_cpu(
    query_planes: torch.Tensor,
    query_meta: torch.Tensor,
    qc_table: torch.Tensor,
    doc_centroid_id: torch.Tensor,
    doc_cos_norm: torch.Tensor,
    doc_sin_norm: torch.Tensor,
    doc_sign: torch.Tensor,
    doc_nz: torch.Tensor,
    doc_scales: torch.Tensor,
    doc_ids: List[int],
    k: int,
    documents_mask: torch.Tensor = None,
    queries_mask: torch.Tensor = None,
    n_threads: int | None = None,
) -> Tuple[List[int], List[float]]:
    """CPU dispatch for `score_rroq158_topk` via the Rust SIMD kernel.

    The Rust kernel returns scores for batch dim A as a flat (A*B) vector but
    the search-time top-k machinery here only consumes the first row, so we
    require ``A == 1``. Routing a batched query (A>1) through this path
    would silently drop A-1 result rows. Callers that need batched scoring
    should iterate over the A axis themselves or call the kernel directly.
    """
    fn = _get_rroq158_cpu_kernel()
    if fn is None:
        return [], []

    import numpy as np

    A_dim = query_planes.shape[0]
    if A_dim != 1:
        raise ValueError(
            f"_score_rroq158_cpu only supports A==1 (single batch row); "
            f"got query_planes shape {tuple(query_planes.shape)} (A={A_dim}). "
            "Iterate over the batch axis explicitly when A>1."
        )

    def _to_np(t: torch.Tensor, dtype: np.dtype) -> np.ndarray:
        # Already-CPU tensors `.detach().cpu()` is a no-op; `.numpy()`
        # returns a view that may be non-contiguous after `.permute` / etc.
        # Skip the redundant `.contiguous()` because `np.ascontiguousarray`
        # below makes any final copy a single allocation rather than two.
        return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=dtype)

    qp_np = _to_np(query_planes, np.int32)
    qm_np = _to_np(query_meta, np.float32)
    qc_np = _to_np(qc_table, np.float32)
    cid_np = _to_np(doc_centroid_id, np.int32)
    cos_np = _to_np(doc_cos_norm, np.float32)
    sin_np = _to_np(doc_sin_norm, np.float32)
    ds_np = _to_np(doc_sign, np.int32)
    dn_np = _to_np(doc_nz, np.int32)
    dsc_np = _to_np(doc_scales, np.float32)

    A, S, query_bits, n_words = qp_np.shape
    B, T = ds_np.shape[:2]
    n_groups = dsc_np.shape[-1]
    K = qc_np.shape[-1]

    q_mask_np = None
    if queries_mask is not None:
        q_mask_np = _to_np(queries_mask, np.float32).ravel()
    d_mask_np = None
    if documents_mask is not None:
        d_mask_np = _to_np(documents_mask, np.float32).ravel()

    n_threads_eff = _resolve_rroq158_n_threads(default=n_threads)

    flat = fn(
        qp_np.ravel(),
        qm_np.ravel(),
        qc_np.ravel(),
        ds_np.ravel(),
        dn_np.ravel(),
        dsc_np.ravel(),
        cid_np.ravel(),
        cos_np.ravel(),
        sin_np.ravel(),
        A, B, S, T, n_words, n_groups, query_bits, K,
        q_mask_np,
        d_mask_np,
        n_threads=n_threads_eff,
    )
    scores = flat.reshape(A, B)[0]
    if B == 0:
        return [], []
    final_k = min(k, B)
    top_idx = np.argpartition(-scores, final_k - 1)[:final_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [doc_ids[i] for i in top_idx.tolist()], scores[top_idx].tolist()


def score_rroq158_topk(
    query_planes: torch.Tensor,         # (1, S, query_bits, n_words) int32
    query_meta: torch.Tensor,           # (1, S, 2) float32
    qc_table: torch.Tensor,             # (1, S, K) float32
    doc_centroid_id: torch.Tensor,      # (B, T) int32
    doc_cos_norm: torch.Tensor,         # (B, T) float32 (or fp16 → cast)
    doc_sin_norm: torch.Tensor,         # (B, T) float32 (or fp16 → cast)
    doc_sign: torch.Tensor,             # (B, T, n_words) int32
    doc_nz: torch.Tensor,               # (B, T, n_words) int32
    doc_scales: torch.Tensor,           # (B, T, n_groups) float32
    doc_ids: List[int],
    k: int = 10,
    documents_mask: torch.Tensor = None,
    queries_mask: torch.Tensor = None,
    device: torch.device = None,
) -> Tuple[List[int], List[float]]:
    """Score documents using the RROQ158 fused Triton kernel.

    All tensors are dtype-converted to what the kernel expects (int32 / fp32)
    on the target device. ``doc_cos_norm`` / ``doc_sin_norm`` are stored as
    fp16 on disk to halve their footprint; the kernel reads them as fp32.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_docs = doc_sign.shape[0]
    if n_docs == 0:
        return [], []

    if device.type == "cpu":
        # CPU lane uses the Rust SIMD kernel (latence_shard_engine.
        # rroq158_score_batch). Same math as the Triton GPU kernel; parity
        # validated to rtol=1e-4 in tests/test_rroq158_kernel.py.
        return _score_rroq158_cpu(
            query_planes, query_meta, qc_table,
            doc_centroid_id, doc_cos_norm, doc_sin_norm,
            doc_sign, doc_nz, doc_scales,
            doc_ids, k, documents_mask, queries_mask,
        )

    rroq = _get_rroq158_maxsim()
    if rroq is None:
        return [], []

    qp = query_planes.to(device=device, dtype=torch.int32)
    qm = query_meta.to(device=device, dtype=torch.float32)
    qct = qc_table.to(device=device, dtype=torch.float32)
    cid = doc_centroid_id.to(device=device, dtype=torch.int32)
    cos_n = doc_cos_norm.to(device=device, dtype=torch.float32)
    sin_n = doc_sin_norm.to(device=device, dtype=torch.float32)
    ds = doc_sign.to(device=device, dtype=torch.int32)
    dn = doc_nz.to(device=device, dtype=torch.int32)
    dsc = doc_scales.to(device=device, dtype=torch.float32)

    kwargs = {}
    if documents_mask is not None:
        kwargs["documents_mask"] = documents_mask.to(device)
    if queries_mask is not None:
        kwargs["queries_mask"] = queries_mask.to(device)

    scores = rroq(
        queries_planes=qp,
        queries_meta=qm,
        qc_table=qct,
        docs_centroid_id=cid,
        docs_cos_norm=cos_n,
        docs_sin_norm=sin_n,
        docs_sign=ds,
        docs_nz=dn,
        docs_scales=dsc,
        **kwargs,
    ).squeeze(0)
    final_k = min(k, n_docs)
    top_sc, top_idx = scores.topk(final_k)
    idx_list = top_idx.cpu().tolist()
    return [doc_ids[i] for i in idx_list], top_sc.cpu().tolist()


# ------------------------------------------------------------------
# RROQ4_RIEM (Riemannian 4-bit asymmetric) scoring
# ------------------------------------------------------------------

_rroq4_riem_fn = None


def _get_rroq4_riem_maxsim():
    global _rroq4_riem_fn
    if _rroq4_riem_fn is not None:
        return _rroq4_riem_fn
    try:
        from voyager_index._internal.kernels.triton_roq_rroq4_riem import (
            roq_maxsim_rroq4_riem,
        )

        _rroq4_riem_fn = roq_maxsim_rroq4_riem
        logger.info(
            "Using voyager-index RROQ4_RIEM (4-bit asymmetric) MaxSim kernel"
        )
    except ImportError:
        _rroq4_riem_fn = None
        logger.warning("RROQ4_RIEM MaxSim kernel unavailable")
    return _rroq4_riem_fn


_rroq4_riem_cpu_fn = None


def _get_rroq4_riem_cpu_kernel():
    """Lazy-load the Rust SIMD CPU kernel from `latence_shard_engine`.

    Returns ``None`` if the wheel is unavailable / older than the
    rroq4_riem release.
    """
    global _rroq4_riem_cpu_fn
    if _rroq4_riem_cpu_fn is not None:
        return _rroq4_riem_cpu_fn
    try:
        import latence_shard_engine as _eng

        _rroq4_riem_cpu_fn = getattr(_eng, "rroq4_riem_score_batch", None)
        if _rroq4_riem_cpu_fn is not None:
            logger.info("Using latence_shard_engine RROQ4_RIEM CPU SIMD kernel")
        else:
            logger.warning(
                "latence_shard_engine missing rroq4_riem_score_batch; rebuild the "
                "Rust wheel from src/kernels/shard_engine to enable CPU rroq4_riem"
            )
    except ImportError:
        _rroq4_riem_cpu_fn = None
        logger.warning("latence_shard_engine unavailable; CPU rroq4_riem disabled")
    return _rroq4_riem_cpu_fn


def _resolve_rroq4_riem_n_threads(default: int | None = None) -> int | None:
    """Threading heuristic for the Rust rroq4_riem kernel.

    Mirrors :func:`_resolve_rroq158_n_threads` but with a separate set of
    env vars so operators can tune the two codecs independently. Same
    rationale: cap rayon when the python side already runs the kernel
    from N parallel workers.
    """
    import os
    raw = os.environ.get("VOYAGER_RROQ4_RIEM_N_THREADS")
    if raw is not None:
        try:
            n = int(raw)
            return n if n > 0 else None
        except ValueError:
            pass
    workers_raw = os.environ.get("VOYAGER_RROQ4_RIEM_N_WORKERS")
    if workers_raw is not None:
        try:
            n_workers = max(1, int(workers_raw))
            cpu = os.cpu_count() or 1
            return max(1, cpu // n_workers)
        except ValueError:
            pass
    return default


def _score_rroq4_riem_cpu(
    q_rot: torch.Tensor,
    q_group_sums: torch.Tensor,
    qc_table: torch.Tensor,
    doc_centroid_id: torch.Tensor,
    doc_cos_norm: torch.Tensor,
    doc_sin_norm: torch.Tensor,
    doc_codes: torch.Tensor,
    doc_mins: torch.Tensor,
    doc_deltas: torch.Tensor,
    doc_ids: List[int],
    k: int,
    group_size: int,
    documents_mask: torch.Tensor = None,
    queries_mask: torch.Tensor = None,
    n_threads: int | None = None,
) -> Tuple[List[int], List[float]]:
    """CPU dispatch for `score_rroq4_riem_topk` via the Rust SIMD kernel.

    Same A==1 constraint as :func:`_score_rroq158_cpu` — top-k routing
    only consumes the first row downstream.
    """
    fn = _get_rroq4_riem_cpu_kernel()
    if fn is None:
        return [], []

    import numpy as np

    A_dim = q_rot.shape[0]
    if A_dim != 1:
        raise ValueError(
            f"_score_rroq4_riem_cpu only supports A==1 (single batch row); "
            f"got q_rot shape {tuple(q_rot.shape)} (A={A_dim}). "
            "Iterate over the batch axis explicitly when A>1."
        )

    def _to_np(t: torch.Tensor, dtype) -> np.ndarray:
        return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=dtype)

    qrot_np = _to_np(q_rot, np.float32)
    qgs_np = _to_np(q_group_sums, np.float32)
    qc_np = _to_np(qc_table, np.float32)
    cid_np = _to_np(doc_centroid_id, np.int32)
    cos_np = _to_np(doc_cos_norm, np.float32)
    sin_np = _to_np(doc_sin_norm, np.float32)
    codes_np = _to_np(doc_codes, np.uint8)
    mins_np = _to_np(doc_mins, np.float32)
    dlts_np = _to_np(doc_deltas, np.float32)

    A, S, dim = qrot_np.shape
    B, T = codes_np.shape[:2]
    n_groups = qgs_np.shape[-1]
    K = qc_np.shape[-1]

    q_mask_np = None
    if queries_mask is not None:
        q_mask_np = _to_np(queries_mask, np.float32).ravel()
    d_mask_np = None
    if documents_mask is not None:
        d_mask_np = _to_np(documents_mask, np.float32).ravel()

    n_threads_eff = _resolve_rroq4_riem_n_threads(default=n_threads)

    flat = fn(
        qrot_np.ravel(),
        qgs_np.ravel(),
        qc_np.ravel(),
        codes_np.ravel(),
        mins_np.ravel(),
        dlts_np.ravel(),
        cid_np.ravel(),
        cos_np.ravel(),
        sin_np.ravel(),
        A, B, S, T, dim, n_groups, group_size, K,
        q_mask_np,
        d_mask_np,
        n_threads=n_threads_eff,
    )
    scores = flat.reshape(A, B)[0]
    if B == 0:
        return [], []
    final_k = min(k, B)
    top_idx = np.argpartition(-scores, final_k - 1)[:final_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [doc_ids[i] for i in top_idx.tolist()], scores[top_idx].tolist()


def score_rroq4_riem_topk(
    q_rot: torch.Tensor,                # (1, S, dim) float32
    q_group_sums: torch.Tensor,         # (1, S, n_groups) float32
    qc_table: torch.Tensor,             # (1, S, K) float32
    doc_centroid_id: torch.Tensor,      # (B, T) int32
    doc_cos_norm: torch.Tensor,         # (B, T) float32 (or fp16 → cast)
    doc_sin_norm: torch.Tensor,         # (B, T) float32 (or fp16 → cast)
    doc_codes: torch.Tensor,            # (B, T, dim/2) uint8
    doc_mins: torch.Tensor,             # (B, T, n_groups) float32 (or fp16)
    doc_deltas: torch.Tensor,           # (B, T, n_groups) float32 (or fp16)
    doc_ids: List[int],
    k: int = 10,
    group_size: int = 32,
    documents_mask: torch.Tensor = None,
    queries_mask: torch.Tensor = None,
    device: torch.device = None,
) -> Tuple[List[int], List[float]]:
    """Score documents using the RROQ4_RIEM fused MaxSim kernel.

    Routes to the Triton GPU kernel on CUDA and the Rust SIMD CPU kernel
    on host. Both return identical scores up to fp32 rounding (parity
    validated to rtol=1e-4 in tests/test_rroq4_riem_kernel.py).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_docs = doc_codes.shape[0]
    if n_docs == 0:
        return [], []

    if device.type == "cpu":
        return _score_rroq4_riem_cpu(
            q_rot, q_group_sums, qc_table,
            doc_centroid_id, doc_cos_norm, doc_sin_norm,
            doc_codes, doc_mins, doc_deltas,
            doc_ids, k, group_size, documents_mask, queries_mask,
        )

    rroq = _get_rroq4_riem_maxsim()
    if rroq is None:
        return [], []

    qrot = q_rot.to(device=device, dtype=torch.float32)
    qgs = q_group_sums.to(device=device, dtype=torch.float32)
    qct = qc_table.to(device=device, dtype=torch.float32)
    cid = doc_centroid_id.to(device=device, dtype=torch.int32)
    cos_n = doc_cos_norm.to(device=device, dtype=torch.float32)
    sin_n = doc_sin_norm.to(device=device, dtype=torch.float32)
    if doc_codes.dtype != torch.uint8:
        doc_codes = doc_codes.to(torch.uint8)
    codes = doc_codes.to(device=device)
    mins = doc_mins.to(device=device, dtype=torch.float32)
    dlts = doc_deltas.to(device=device, dtype=torch.float32)

    kwargs = {"group_size": group_size}
    if documents_mask is not None:
        kwargs["documents_mask"] = documents_mask.to(device)
    if queries_mask is not None:
        kwargs["queries_mask"] = queries_mask.to(device)

    scores = rroq(
        queries_rot=qrot,
        queries_group_sums=qgs,
        qc_table=qct,
        docs_centroid_id=cid,
        docs_cos_norm=cos_n,
        docs_sin_norm=sin_n,
        docs_codes_packed=codes,
        docs_mins=mins,
        docs_deltas=dlts,
        **kwargs,
    ).squeeze(0)
    final_k = min(k, n_docs)
    top_sc, top_idx = scores.topk(final_k)
    idx_list = top_idx.cpu().tolist()
    return [doc_ids[i] for i in idx_list], top_sc.cpu().tolist()


# ------------------------------------------------------------------
# Per-shard scoring (legacy, kept for backward compat)
# ------------------------------------------------------------------


def _pad_flat_embeddings(
    flat_emb: torch.Tensor,
    offsets: List[Tuple[int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a flat doc list already resident on *device*."""
    if not offsets:
        dim = flat_emb.shape[1]
        return (
            torch.empty(0, 1, dim, dtype=flat_emb.dtype, device=device),
            torch.empty(0, 1, dtype=torch.float32, device=device),
        )

    lengths = [e - s for s, e in offsets]
    n_docs = len(offsets)
    max_tok = max(lengths)
    min_tok = min(lengths)
    dim = flat_emb.shape[1]

    if min_tok == max_tok:
        padded = flat_emb[: n_docs * max_tok].view(n_docs, max_tok, dim)
        mask = torch.ones(n_docs, max_tok, dtype=torch.float32, device=device)
        return padded, mask

    # Vectorized scatter (Fix 3)
    total_tokens = flat_emb.shape[0]
    lengths_t = torch.tensor(lengths, dtype=torch.int64, device=device)
    doc_indices = torch.repeat_interleave(torch.arange(n_docs, device=device), lengths_t)
    offsets_t = torch.zeros(n_docs + 1, dtype=torch.int64, device=device)
    offsets_t[1:] = lengths_t.cumsum(0)
    token_positions = torch.arange(total_tokens, device=device) - offsets_t[doc_indices]

    padded = torch.zeros(n_docs, max_tok, dim, dtype=flat_emb.dtype, device=device)
    mask = torch.zeros(n_docs, max_tok, dtype=torch.float32, device=device)
    padded[doc_indices, token_positions] = flat_emb
    mask[doc_indices, token_positions] = 1.0

    return padded, mask


def _pad_shard_on_device(
    flat_emb: torch.Tensor,
    offsets: List[Tuple[int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a single shard's docs on GPU. Uses vectorized scatter for variable lengths."""
    if flat_emb.device != device:
        flat_emb = flat_emb.to(device, non_blocking=True)
    return _pad_flat_embeddings(flat_emb, offsets, device)


def score_shards_and_topk(
    query: torch.Tensor,
    shard_chunks: List[ShardChunk],
    k: int = 10,
    device: torch.device = None,
) -> Tuple[List[int], List[float]]:
    """Score per-shard and merge top-k on GPU. Legacy — prefer score_all_docs_topk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maxsim = _get_maxsim()

    q = query.to(device, dtype=torch.float16)
    if q.dim() == 2:
        q = q.unsqueeze(0)

    best_scores: List[torch.Tensor] = []
    best_ids: List[List[int]] = []

    for flat_emb, offsets, doc_ids in shard_chunks:
        if not doc_ids:
            continue

        doc_emb, doc_mask = _pad_shard_on_device(flat_emb, offsets, device)

        scores = maxsim(
            queries_embeddings=q,
            documents_embeddings=doc_emb,
            documents_mask=doc_mask,
        ).squeeze(0)

        shard_k = min(k, len(doc_ids))
        top_sc, top_idx = scores.topk(shard_k)
        best_scores.append(top_sc)
        best_ids.append([doc_ids[i] for i in top_idx.cpu().tolist()])

    if not best_scores:
        return [], []

    all_scores = torch.cat(best_scores)
    all_ids_flat: List[int] = []
    for id_list in best_ids:
        all_ids_flat.extend(id_list)

    final_k = min(k, len(all_ids_flat))
    top_sc, top_idx = all_scores.topk(final_k)

    result_ids = [all_ids_flat[i] for i in top_idx.cpu().tolist()]
    result_scores = top_sc.cpu().tolist()
    return result_ids, result_scores


def score_and_topk(
    query: torch.Tensor,
    doc_embeddings: torch.Tensor,
    doc_mask: torch.Tensor,
    doc_ids: List[int],
    k: int = 10,
    use_quantization: bool = False,
    quantization_mode: str = "int8",
) -> Tuple[List[int], List[float]]:
    """Score documents against query, return top-k IDs and scores. Legacy interface."""
    maxsim = _get_maxsim()

    if query.dim() == 2:
        query = query.unsqueeze(0)

    n_docs = doc_embeddings.shape[0]
    if n_docs == 0:
        return [], []

    scores = maxsim(
        queries_embeddings=query,
        documents_embeddings=doc_embeddings,
        documents_mask=doc_mask,
        use_quantization=use_quantization,
        quantization_mode=quantization_mode,
    ).squeeze(0)

    actual_k = min(k, n_docs)
    top_scores, top_indices = scores.topk(actual_k)

    top_ids = [doc_ids[i] for i in top_indices.cpu().tolist()]
    top_sc = top_scores.cpu().tolist()
    return top_ids, top_sc


def brute_force_maxsim(
    query: torch.Tensor,
    all_doc_vecs: list,
    doc_ids: List[int],
    dim: int,
    k: int = 100,
    device: str = "cuda",
    batch_size: int = 2000,
) -> Tuple[List[int], List[float]]:
    """Brute-force MaxSim over an entire corpus for ground-truth computation."""
    import numpy as np

    if not doc_ids:
        return [], []

    maxsim = _get_maxsim()

    if isinstance(query, np.ndarray):
        query = torch.from_numpy(query)
    q = query.float().unsqueeze(0).to(device)

    all_scores = []
    for start in range(0, len(doc_ids), batch_size):
        end = min(start + batch_size, len(doc_ids))
        batch = all_doc_vecs[start:end]
        max_tok = max(v.shape[0] for v in batch)

        D = np.zeros((end - start, max_tok, dim), dtype=np.float32)
        M = np.zeros((end - start, max_tok), dtype=np.float32)
        for i, v in enumerate(batch):
            fv = v.astype(np.float32) if v.dtype != np.float32 else v
            D[i, : fv.shape[0]] = fv
            M[i, : fv.shape[0]] = 1.0

        scores = maxsim(
            q,
            torch.from_numpy(D).to(device),
            documents_mask=torch.from_numpy(M).to(device),
        ).squeeze(0)
        all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores)
    topk = all_scores.topk(min(k, len(doc_ids)))
    top_ids = [doc_ids[j] for j in topk.indices.tolist()]
    top_sc = topk.values.tolist()
    return top_ids, top_sc
