"""Query normalization and routing helpers for LEMUR."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .state import CandidatePlan

class LemurRouterQueryMixin:
    @staticmethod
    def _as_cpu_float32_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            q = value.detach()
            if q.device.type != "cpu":
                q = q.cpu()
            if q.dtype != torch.float32:
                q = q.to(torch.float32)
            if not q.is_contiguous():
                q = q.contiguous()
            return q

        arr = np.asarray(value)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return torch.from_numpy(arr)

    @staticmethod
    def _as_numpy_float32_contiguous(value: torch.Tensor) -> np.ndarray:
        t = value.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        if not t.is_contiguous():
            t = t.contiguous()
        return t.numpy()

    @staticmethod
    def _normalize_query_tokens(query_vectors: torch.Tensor | np.ndarray) -> torch.Tensor:
        q = LemurRouterQueryMixin._as_cpu_float32_tensor(query_vectors)
        if q.dim() == 3 and q.shape[0] == 1:
            q = q.squeeze(0)
        return q.contiguous()

    def _compute_search_k(self, k_candidates: int, search_k_cap: Optional[int]) -> int:
        tombstone_headroom = max(64, int(len(self._tombstones) * 1.5))
        index_size = self._weights.shape[0] if self._weights is not None else k_candidates
        search_k = min(k_candidates + tombstone_headroom, index_size)
        if search_k_cap is not None:
            search_k = min(search_k, int(search_k_cap))
        if self._gpu_index_active:
            search_k = min(search_k, 2048)
        return max(search_k, 1)

    def _candidate_plan_from_rows(
        self,
        row_ids: Sequence[int],
        prefetch_doc_cap: int,
    ) -> CandidatePlan:
        doc_ids: List[int] = []
        for row in row_ids:
            row = int(row)
            if row < 0 or row >= len(self._row_to_doc_id):
                continue
            doc_id = self._row_to_doc_id[row]
            if doc_id in self._tombstones:
                continue
            doc_ids.append(doc_id)
            if len(doc_ids) >= int(prefetch_doc_cap):
                break
        by_shard: Dict[int, List[int]] = {}
        for doc_id in doc_ids:
            shard_id = int(self._doc_id_to_shard[doc_id])
            by_shard.setdefault(shard_id, []).append(doc_id)
        return CandidatePlan(
            doc_ids=doc_ids,
            shard_ids=sorted(by_shard.keys()),
            by_shard=by_shard,
            generation=self._state.generation,
            post_tombstone_count=len(doc_ids),
        )

    def route(
        self,
        query_vectors: torch.Tensor,
        k_candidates: int = 2000,
        prefetch_doc_cap: int = 10000,
        nprobe_override: Optional[int] = None,
        search_k_cap: Optional[int] = 2048,
    ) -> CandidatePlan:
        if self._lemur is None or self._index is None:
            raise RuntimeError("router is not loaded")
        q = self._normalize_query_tokens(query_vectors)
        q_counts = torch.tensor([q.shape[0]], dtype=torch.int32)
        feats = self._as_cpu_float32_tensor(self._lemur.compute_features((q, q_counts)))
        search_k = self._compute_search_k(k_candidates, search_k_cap)
        saved_nprobe = self._apply_nprobe_override(nprobe_override)
        try:
            _, row_ids = self._search(
                feats,
                search_k,
                use_lock=self._gpu_index_active or saved_nprobe is not None,
            )
        finally:
            self._restore_nprobe(saved_nprobe)
        rows = row_ids[0].tolist() if row_ids.size else []
        return self._candidate_plan_from_rows(rows, prefetch_doc_cap)

    def route_batch(
        self,
        query_vectors: Sequence[torch.Tensor | np.ndarray],
        k_candidates: int = 2000,
        prefetch_doc_cap: int = 10000,
        nprobe_override: Optional[int] = None,
        search_k_cap: Optional[int] = 2048,
    ) -> List[CandidatePlan]:
        if self._lemur is None or self._index is None:
            raise RuntimeError("router is not loaded")
        if not query_vectors:
            return []

        normalized = [self._normalize_query_tokens(q) for q in query_vectors]
        q_counts = torch.tensor([q.shape[0] for q in normalized], dtype=torch.int32)
        flat_queries = torch.cat(normalized, dim=0)
        feats = self._as_cpu_float32_tensor(self._lemur.compute_features((flat_queries, q_counts)))
        search_k = self._compute_search_k(k_candidates, search_k_cap)
        saved_nprobe = self._apply_nprobe_override(nprobe_override)
        try:
            _, row_ids = self._search(
                feats,
                search_k,
                use_lock=self._gpu_index_active or saved_nprobe is not None,
            )
        finally:
            self._restore_nprobe(saved_nprobe)

        return [
            self._candidate_plan_from_rows(rows.tolist(), prefetch_doc_cap)
            for rows in row_ids
        ]

