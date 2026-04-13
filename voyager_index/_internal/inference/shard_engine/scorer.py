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
from typing import Dict, List, Tuple

import numpy as np
import torch

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
) -> Tuple[List[int], List[float]]:
    """Score all fetched docs in one kernel call.

    Concatenates per-shard tensors (not per-doc slices) to avoid O(n_docs)
    torch.cat overhead, then reshapes for the Triton MaxSim kernel.

    When *quantization_mode* is set (e.g. ``"int8"``), the flag is forwarded
    to the Triton kernel so it can use reduced-precision accumulation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return [], []

    n_docs = len(all_doc_ids)
    dim = shard_embs[0].shape[1]

    quant_kwargs: dict = {}
    if quantization_mode:
        quant_kwargs["use_quantization"] = True
        quant_kwargs["quantization_mode"] = quantization_mode

    if is_uniform and tok_per_doc > 0:
        flat = torch.cat(shard_embs, dim=0)
        D_gpu = flat.view(n_docs, tok_per_doc, dim).to(device, dtype=torch.float16)
        scores = maxsim(
            queries_embeddings=q,
            documents_embeddings=D_gpu,
            documents_mask=None,
            **quant_kwargs,
        ).squeeze(0)
    else:
        lengths: List[int] = []
        all_slices: List[torch.Tensor] = []
        for flat_emb, offsets, doc_ids in shard_chunks:
            if not doc_ids:
                continue
            for s, e in offsets:
                all_slices.append(flat_emb[s:e])
                lengths.append(e - s)
        max_tok = max(lengths)
        D = np.zeros((n_docs, max_tok, dim), dtype=np.float16)
        M = np.zeros((n_docs, max_tok), dtype=np.float32)
        for i, sl in enumerate(all_slices):
            tok = sl.shape[0]
            D[i, :tok] = sl.numpy()
            M[i, :tok] = 1.0
        D_gpu = torch.from_numpy(D).to(device)
        M_gpu = torch.from_numpy(M).to(device)
        scores = maxsim(
            queries_embeddings=q,
            documents_embeddings=D_gpu,
            documents_mask=M_gpu,
            **quant_kwargs,
        ).squeeze(0)

    final_k = min(k, n_docs)
    top_sc, top_idx = scores.topk(final_k)

    idx_list = top_idx.cpu().tolist()
    result_ids = [all_doc_ids[i] for i in idx_list]
    result_scores = top_sc.cpu().tolist()
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
            n_docs, max_tok, dim,
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
    ) -> Tuple[List[int], List[float]]:
        maxsim = _get_maxsim()
        q = query.to(self._device, dtype=torch.float16)
        if q.dim() == 2:
            q = q.unsqueeze(0)

        valid_ids = [did for did in candidate_doc_ids if did in self.doc_id_to_idx]
        if not valid_ids:
            return [], []

        indices = torch.tensor(
            [self.doc_id_to_idx[did] for did in valid_ids],
            dtype=torch.long,
            device=self._device,
        )

        D_slice = self.D[indices]
        M_slice = self.M[indices]

        has_padding = (M_slice[:, -1] == 0).any()
        if has_padding:
            actual_max = int(M_slice.sum(dim=1).max().item())
            D_slice = D_slice[:, :actual_max].contiguous()
            M_slice = M_slice[:, :actual_max].contiguous()

        scores = maxsim(
            queries_embeddings=q,
            documents_embeddings=D_slice,
            documents_mask=M_slice if has_padding else None,
        ).squeeze(0)

        n = len(valid_ids)
        final_k = min(k, n)
        top_sc, top_idx = scores.topk(final_k)

        idx_list = top_idx.cpu().tolist()
        return [valid_ids[i] for i in idx_list], top_sc.cpu().tolist()

    @staticmethod
    def fits_on_gpu(n_docs: int, max_tok: int, dim: int, dtype=torch.float16) -> bool:
        bytes_needed = PreloadedGpuCorpus.estimate_gpu_bytes(n_docs, max_tok, dim, dtype=dtype)
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
# Per-shard scoring (legacy, kept for backward compat)
# ------------------------------------------------------------------

def _pad_shard_on_device(
    flat_emb: torch.Tensor,
    offsets: List[Tuple[int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a single shard's docs on GPU. Uses vectorized scatter for variable lengths."""
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

    if flat_emb.device != device:
        flat_emb = flat_emb.to(device, non_blocking=True)

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
