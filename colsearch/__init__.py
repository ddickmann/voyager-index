"""
Public OSS API for colsearch.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

try:
    __version__: str = _pkg_version("colsearch")
except PackageNotFoundError:
    __version__ = "0+local"

_EXPORTS = {
    "Index": ("colsearch.index", "Index"),
    "IndexBuilder": ("colsearch.index", "IndexBuilder"),
    "SearchResult": ("colsearch.index", "SearchResult"),
    "ScrollPage": ("colsearch.index", "ScrollPage"),
    "IndexStats": ("colsearch.index", "IndexStats"),
    "BM25Config": ("colsearch.config", "BM25Config"),
    "FusionConfig": ("colsearch.config", "FusionConfig"),
    "IndexConfig": ("colsearch.config", "IndexConfig"),
    "Neo4jConfig": ("colsearch.config", "Neo4jConfig"),
    # Re-export the shard-engine `Compression` enum so the documented form
    # `from colsearch import Compression; ...compression=Compression.RROQ158`
    # works without users reaching into the `_internal` namespace.
    "Compression": (
        "colsearch._internal.inference.shard_engine.serving_config",
        "Compression",
    ),
    "TRITON_AVAILABLE": ("colsearch.kernels", "TRITON_AVAILABLE"),
    "fast_colbert_scores": ("colsearch.kernels", "fast_colbert_scores"),
    "roq_maxsim_1bit": ("colsearch.kernels", "roq_maxsim_1bit"),
    "roq_maxsim_2bit": ("colsearch.kernels", "roq_maxsim_2bit"),
    "roq_maxsim_4bit": ("colsearch.kernels", "roq_maxsim_4bit"),
    "roq_maxsim_8bit": ("colsearch.kernels", "roq_maxsim_8bit"),
    "DEFAULT_MULTIMODAL_MODEL": ("colsearch.multimodal", "DEFAULT_MULTIMODAL_MODEL"),
    "DEFAULT_MULTIMODAL_MODEL_SPEC": ("colsearch.multimodal", "DEFAULT_MULTIMODAL_MODEL_SPEC"),
    "MultimodalModelSpec": ("colsearch.multimodal", "MultimodalModelSpec"),
    "SUPPORTED_MULTIMODAL_MODELS": ("colsearch.multimodal", "SUPPORTED_MULTIMODAL_MODELS"),
    "VllmPoolingProvider": ("colsearch.multimodal", "VllmPoolingProvider"),
    "ColPaliConfig": ("colsearch.search", "ColPaliConfig"),
    "ColPaliEngine": ("colsearch.search", "ColPaliEngine"),
    "MultiModalEngine": ("colsearch.search", "MultiModalEngine"),
    "ColbertIndex": ("colsearch.search", "ColbertIndex"),
    "RENDERABLE_SOURCE_SUFFIXES": ("colsearch.preprocessing", "RENDERABLE_SOURCE_SUFFIXES"),
    "SearchPipeline": ("colsearch.search", "SearchPipeline"),
    "enumerate_renderable_documents": ("colsearch.preprocessing", "enumerate_renderable_documents"),
    "render_documents": ("colsearch.preprocessing", "render_documents"),
    "VectorPayload": ("colsearch.transport", "VectorPayload"),
    "decode_payload": ("colsearch.transport", "decode_payload"),
    "encode_roq_payload": ("colsearch.transport", "encode_roq_payload"),
    "encode_vector_payload": ("colsearch.transport", "encode_vector_payload"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'colsearch' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__) + ["__version__"])


__all__ = [
    "Index",
    "IndexBuilder",
    "SearchResult",
    "ScrollPage",
    "IndexStats",
    "BM25Config",
    "Compression",
    "FusionConfig",
    "IndexConfig",
    "Neo4jConfig",
    "TRITON_AVAILABLE",
    "fast_colbert_scores",
    "roq_maxsim_1bit",
    "roq_maxsim_2bit",
    "roq_maxsim_4bit",
    "roq_maxsim_8bit",
    "DEFAULT_MULTIMODAL_MODEL",
    "DEFAULT_MULTIMODAL_MODEL_SPEC",
    "MultimodalModelSpec",
    "SUPPORTED_MULTIMODAL_MODELS",
    "VllmPoolingProvider",
    "ColPaliConfig",
    "ColPaliEngine",
    "MultiModalEngine",
    "ColbertIndex",
    "RENDERABLE_SOURCE_SUFFIXES",
    "SearchPipeline",
    "enumerate_renderable_documents",
    "render_documents",
    "VectorPayload",
    "decode_payload",
    "encode_roq_payload",
    "encode_vector_payload",
]
