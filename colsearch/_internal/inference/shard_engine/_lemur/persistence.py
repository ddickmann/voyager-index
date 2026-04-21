"""Persistence helpers for the LEMUR router."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from dataclasses import asdict

from .state import RouterState

class LemurRouterPersistenceMixin:
    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._weights, self.index_dir / "weights.pt")
        with open(self.index_dir / "doc_maps.json", "w") as f:
            json.dump(
                {
                    "row_to_doc_id": self._row_to_doc_id,
                    "doc_id_to_shard": self._doc_id_to_shard,
                    "tombstones": sorted(self._tombstones),
                },
                f,
            )
        with open(self.index_dir / "router_state.json", "w") as f:
            json.dump(asdict(self._state), f, indent=2)
        if self._use_faiss and self._index is not None:
            idx_to_save = self._index
            if hasattr(faiss, "index_gpu_to_cpu"):
                try:
                    idx_to_save = faiss.index_gpu_to_cpu(self._index)
                except Exception:
                    pass
            faiss.write_index(idx_to_save, str(self.index_dir / "ann.index"))

    def load(self) -> None:
        self._load_if_present(required=True)

    def _load_if_present(self, required: bool = False) -> None:
        state_path = self.index_dir / "router_state.json"
        if not state_path.exists():
            if required:
                raise FileNotFoundError(f"router state missing at {state_path}")
            return
        with open(state_path) as f:
            self._state = RouterState(**json.load(f))
        self.ann_backend = self._state.ann_backend.replace("hnsw", "flat")
        self._weights = torch.load(self.index_dir / "weights.pt", weights_only=True).detach().cpu().to(torch.float32)
        with open(self.index_dir / "doc_maps.json") as f:
            maps = json.load(f)
        self._row_to_doc_id = [int(x) for x in maps["row_to_doc_id"]]
        self._doc_ids = list(self._row_to_doc_id)
        self._doc_id_to_row = {doc_id: idx for idx, doc_id in enumerate(self._row_to_doc_id)}
        self._doc_id_to_shard = {int(k): int(v) for k, v in maps["doc_id_to_shard"].items()}
        self._tombstones = {int(x) for x in maps.get("tombstones", [])}
        self._lemur = self._new_lemur_backend(load_saved=True)
        self._gpu_index_active = False
        if self._use_faiss and (self.index_dir / "ann.index").exists():
            cpu_index = faiss.read_index(str(self.index_dir / "ann.index"))
            self._set_nprobe_if_ivf(cpu_index)
            use_gpu = (
                self.device != "cpu"
                and torch.cuda.is_available()
                and hasattr(faiss, "StandardGpuResources")
            )
            if use_gpu:
                try:
                    res = self._get_gpu_resources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    self._gpu_index_active = True
                    logger.info("ANN index loaded to GPU (faiss-gpu)")
                except Exception:
                    self._index = cpu_index
            else:
                self._index = cpu_index
        else:
            self._rebuild_ann()

