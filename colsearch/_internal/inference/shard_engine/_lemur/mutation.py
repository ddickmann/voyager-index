"""Document update and lifecycle operations for LEMUR routing."""
from __future__ import annotations

from .common import *  # noqa: F401,F403

class LemurRouterMutationMixin:
    def fit_initial(
        self,
        pooled_doc_vectors: torch.Tensor,
        pooled_doc_counts: torch.Tensor,
        doc_ids: Sequence[int],
        doc_id_to_shard: Dict[int, int],
        epochs: int = 10,
    ) -> None:
        vectors = pooled_doc_vectors.detach().cpu().contiguous()
        counts = pooled_doc_counts.detach().cpu().to(torch.int32).contiguous()
        if int(counts.sum(dtype=torch.int64).item()) != int(vectors.shape[0]):
            raise ValueError("pooled_doc_counts do not sum to pooled_doc_vectors rows")

        self._lemur = self._new_lemur_backend()
        self._lemur.fit(train=vectors, train_counts=counts, epochs=int(epochs), verbose=False)
        weights = self._lemur.compute_weights(vectors, counts).detach().cpu().to(torch.float32).contiguous()
        self._weights = weights
        self._doc_ids = [int(x) for x in doc_ids]
        self._row_to_doc_id = list(self._doc_ids)
        self._doc_id_to_row = {doc_id: idx for idx, doc_id in enumerate(self._row_to_doc_id)}
        self._doc_id_to_shard = {int(k): int(v) for k, v in doc_id_to_shard.items()}
        self._tombstones.clear()
        self._state.generation += 1
        self._state.feature_dim = int(weights.shape[1]) if weights.ndim == 2 else 0
        self._state.backend_name = "official_lemur" if OFFICIAL_LEMUR_AVAILABLE else "fallback_proxy"
        self._state.live_docs = len(self._doc_ids)
        self._state.total_docs = len(self._doc_ids)
        self._state.dirty_ops_count = 0
        self._state.dirty_doc_ratio = 0.0
        self._state.dirty_shard_ratio = 0.0
        self._rebuild_ann()
        self.save()

    def add_or_update_docs(
        self,
        pooled_doc_vectors: torch.Tensor,
        pooled_doc_counts: torch.Tensor,
        doc_ids: Sequence[int],
        doc_id_to_shard: Dict[int, int],
    ) -> None:
        if self._lemur is None:
            raise RuntimeError("router is not initialized; call fit_initial first")
        vectors = pooled_doc_vectors.detach().cpu().to(torch.float32).contiguous()
        counts = pooled_doc_counts.detach().cpu().to(torch.int32).contiguous()
        new_weights = self._lemur.compute_weights(vectors, counts).detach().cpu().to(torch.float32).contiguous()

        added_rows = []
        added_ids = []
        for i, doc_id in enumerate(doc_ids):
            doc_id = int(doc_id)
            shard_id = int(doc_id_to_shard[doc_id])
            if doc_id in self._doc_id_to_row:
                row = self._doc_id_to_row[doc_id]
                self._weights[row] = new_weights[i]
                self._doc_id_to_shard[doc_id] = shard_id
                self._tombstones.discard(doc_id)
            else:
                row = len(self._row_to_doc_id)
                self._row_to_doc_id.append(doc_id)
                self._doc_id_to_row[doc_id] = row
                self._doc_id_to_shard[doc_id] = shard_id
                self._doc_ids.append(doc_id)
                added_rows.append(new_weights[i].unsqueeze(0))
                added_ids.append(doc_id)

        if added_rows:
            self._weights = torch.cat([self._weights, torch.cat(added_rows, dim=0)], dim=0)
            self._state.total_docs += len(added_rows)
            self._state.live_docs += len(added_rows)

        self._rebuild_ann()
        self._mark_dirty(len(doc_ids), set(doc_id_to_shard.values()))
        self.save()

    def delete_docs(self, doc_ids: Sequence[int]) -> None:
        deleted_shards = set()
        for doc_id in doc_ids:
            doc_id = int(doc_id)
            if doc_id in self._doc_id_to_shard:
                deleted_shards.add(self._doc_id_to_shard[doc_id])
                self._tombstones.add(doc_id)
        self._state.live_docs = max(0, self._state.total_docs - len(self._tombstones))
        self._mark_dirty(len(doc_ids), deleted_shards)
        self.save()

    def should_full_retrain(
        self,
        retrain_every_ops: int = 50_000,
        retrain_dirty_doc_ratio: float = 0.05,
        retrain_dirty_shard_ratio: float = 0.10,
    ) -> bool:
        return (
            self._state.dirty_ops_count >= retrain_every_ops
            or self._state.dirty_doc_ratio >= retrain_dirty_doc_ratio
            or self._state.dirty_shard_ratio >= retrain_dirty_shard_ratio
        )

    def _mark_dirty(self, changed_ops: int, changed_shards: Iterable[int]) -> None:
        self._state.dirty_ops_count += int(changed_ops)
        if self._state.total_docs > 0:
            self._state.dirty_doc_ratio = min(1.0, self._state.dirty_ops_count / float(self._state.total_docs))
        total_shards = max(1, len(set(self._doc_id_to_shard.values())))
        changed = len(set(int(x) for x in changed_shards))
        if changed > 0:
            self._state.dirty_shard_ratio = min(1.0, max(self._state.dirty_shard_ratio, changed / float(total_shards)))

