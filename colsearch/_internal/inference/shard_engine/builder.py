"""Public compatibility facade for the offline shard build pipeline."""
from __future__ import annotations

from ._builder import DEFAULT_NPZ, assign_storage_shards, build, load_corpus, main
from ._builder.layout import _index_dir

__all__ = ["DEFAULT_NPZ", "_index_dir", "assign_storage_shards", "build", "load_corpus", "main"]
