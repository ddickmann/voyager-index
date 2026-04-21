"""Public shard store composed from focused internal mixins."""
from __future__ import annotations

from .build import ShardStoreBuildMixin
from .cache import _ShardLRU
from .common import *  # noqa: F401,F403
from .fetch import ShardStoreFetchMixin
from .manifest import DocMeta, ShardMeta, StoreManifest

class ShardStore(ShardStoreBuildMixin, ShardStoreFetchMixin):
    def __init__(self, root_path: Path, lru_max_shards: int = 512):
        self.root = Path(root_path)
        self.shard_dir = self.root / "shards"
        self.manifest_path = self.root / "manifest.json"
        self.doc_index_path = self.root / "doc_index.npz"
        self.manifest: Optional[StoreManifest] = None
        self._meta_cache: Dict[int, ShardMeta] = {}
        self._doc_index: Dict[int, DocMeta] = {}
        self._shard_cache = _ShardLRU(max_shards=lru_max_shards)

        if self.manifest_path.exists():
            self.manifest = StoreManifest.load(self.manifest_path)
            self._meta_cache = {m.shard_id: m for m in self.manifest.shards}
        if self.doc_index_path.exists():
            self._load_doc_index()

    def all_doc_ids(self) -> List[int]:
        """Return all document IDs stored in the index."""
        return sorted(self._doc_index.keys())

    def shard_ids(self) -> List[int]:
        return [s.shard_id for s in self.manifest.shards] if self.manifest else []

    def shard_doc_count(self, shard_id: int) -> int:
        return self._meta_by_id(shard_id).num_docs

    def doc_shard_id(self, doc_id: int) -> int:
        return self._doc_index[int(doc_id)].shard_id

    def _meta_by_id(self, shard_id: int) -> ShardMeta:
        shard_id = int(shard_id)
        if shard_id not in self._meta_cache:
            raise KeyError(f"unknown shard_id={shard_id}")
        return self._meta_cache[shard_id]

    def _decode_embeddings(self, meta: ShardMeta, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if meta.compression == "int8":
            if "embeddings" not in data:
                raise KeyError(f"'embeddings' missing in INT8 shard {meta.shard_id}")
            if "scales" not in data:
                raise KeyError(f"'scales' missing in INT8 shard {meta.shard_id}")
            emb = data["embeddings"].float()
            scales = data["scales"].unsqueeze(-1)
            return (emb * scales).to(torch.float16)
        if meta.compression == "roq4" and "roq_codes" in data:
            if "embeddings" in data:
                return data["embeddings"].to(torch.float16)
            logger.warning("ROQ4 shard %d has no FP16 embeddings alongside codes; "
                           "returning zeros", meta.shard_id)
            dim = self.manifest.dim if self.manifest else 128
            n_tokens = data["roq_codes"].shape[0]
            return torch.zeros(n_tokens, dim, dtype=torch.float16)
        if "embeddings" not in data:
            raise KeyError(f"'embeddings' missing in shard {meta.shard_id} "
                           f"(compression={meta.compression})")
        return data["embeddings"].to(torch.float16)

    def _load_doc_index(self) -> None:
        with np.load(self.doc_index_path) as data:
            self._doc_index = {}
            for doc_id, shard_id, start, end, row_index in zip(
                data["doc_ids"], data["shard_ids"], data["local_starts"], data["local_ends"], data["row_indices"],
            ):
                self._doc_index[int(doc_id)] = DocMeta(
                    doc_id=int(doc_id),
                    shard_id=int(shard_id),
                    local_offset_start=int(start),
                    local_offset_end=int(end),
                    row_index=int(row_index),
                )

    def drop_page_cache(self, shard_id: int) -> None:
        meta = self._meta_by_id(shard_id)
        path = self.shard_dir / meta.file_name
        try:
            fd = os.open(str(path), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (AttributeError, OSError):
            pass

    def drop_all_page_cache(self) -> None:
        if not self.manifest:
            return
        for meta in self.manifest.shards:
            self.drop_page_cache(meta.shard_id)

    def page_cache_residency(self) -> Optional[Dict]:
        """Check page cache residency for shard files via ``mincore(2)``.

        Returns ``{"total_pages": N, "resident_pages": M, "hit_rate": M/N}``
        on Linux, or ``None`` on unsupported platforms / errors.
        """
        import ctypes
        import ctypes.util
        import sys

        if sys.platform != "linux":
            return None
        if not self.manifest:
            return None

        try:
            libc_name = ctypes.util.find_library("c")
            if not libc_name:
                return None
            libc = ctypes.CDLL(libc_name, use_errno=True)
            libc.mincore.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_ubyte),
            ]
            libc.mincore.restype = ctypes.c_int
            _MAP_SHARED = 0x01
            _PROT_READ = 0x1
            libc.mmap.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_long,
            ]
            libc.mmap.restype = ctypes.c_void_p
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.munmap.restype = ctypes.c_int
            _MAP_FAILED = ctypes.c_void_p(-1).value
        except (OSError, AttributeError):
            return None

        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            page_size = 4096
        total_pages = 0
        resident_pages = 0

        for meta in self.manifest.shards:
            path = self.shard_dir / meta.file_name
            if not path.exists():
                continue
            try:
                fd = os.open(str(path), os.O_RDONLY)
                try:
                    size = os.fstat(fd).st_size
                    if size == 0:
                        continue
                    n_pages = (size + page_size - 1) // page_size
                    addr = libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
                    if addr == _MAP_FAILED or addr is None:
                        continue
                    try:
                        vec = (ctypes.c_ubyte * n_pages)()
                        rc = libc.mincore(ctypes.c_void_p(addr), size, vec)
                        if rc == 0:
                            total_pages += n_pages
                            resident_pages += sum(1 for b in vec if b & 1)
                    finally:
                        libc.munmap(ctypes.c_void_p(addr), size)
                finally:
                    os.close(fd)
            except (OSError, ValueError):
                continue

        if total_pages == 0:
            return {"total_pages": 0, "resident_pages": 0, "hit_rate": 0.0}
        return {
            "total_pages": total_pages,
            "resident_pages": resident_pages,
            "hit_rate": resident_pages / total_pages,
        }

