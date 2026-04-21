"""Public LEMUR router composed from focused internal mixins."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .ann import LemurRouterAnnMixin
from .mutation import LemurRouterMutationMixin
from .persistence import LemurRouterPersistenceMixin
from .query import LemurRouterQueryMixin
from .state import RouterState

class LemurRouter(
    LemurRouterMutationMixin,
    LemurRouterQueryMixin,
    LemurRouterPersistenceMixin,
    LemurRouterAnnMixin,
):
    """Production wrapper around LEMUR-style proxy routing.

    The class keeps the retrieval-facing contract stable even if the underlying
    ANN backend or the official LEMUR package changes.
    """

    def __init__(
        self,
        index_dir: Path,
        ann_backend: str = "faiss_flat_ip",
        device: str = "cpu",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        ann_backend = ann_backend.replace("hnsw", "flat")
        self.ann_backend = ann_backend
        self.device = device
        self._lemur = None
        self._index = None
        self._state = RouterState(ann_backend=ann_backend)
        self._doc_ids: List[int] = []
        self._row_to_doc_id: List[int] = []
        self._doc_id_to_row: Dict[int, int] = {}
        self._doc_id_to_shard: Dict[int, int] = {}
        self._tombstones: set[int] = set()
        self._weights = torch.empty((0, 0), dtype=torch.float32)
        self._use_faiss = FAISS_AVAILABLE and ann_backend.startswith("faiss")
        self._gpu_index_active = False
        self._gpu_res = None
        self._gpu_res_lock = __import__("threading").Lock()
        self._search_lock = __import__("threading").Lock()
        self._load_if_present()

    def capability_snapshot(self) -> Dict[str, object]:
        return {
            "backend_name": self._state.backend_name,
            "ann_backend": self._state.ann_backend,
            "official_lemur_available": OFFICIAL_LEMUR_AVAILABLE,
            "faiss_available": FAISS_AVAILABLE,
            "gpu_index_active": self._gpu_index_active,
        }

