"""
Public OSS API for voyager-index.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

try:
    __version__: str = _pkg_version("voyager-index")
except PackageNotFoundError:
    __version__ = "0+local"

_EXPORTS = {
    "Index": ("voyager_index.index", "Index"),
    "IndexBuilder": ("voyager_index.index", "IndexBuilder"),
    "SearchResult": ("voyager_index.index", "SearchResult"),
    "ScrollPage": ("voyager_index.index", "ScrollPage"),
    "IndexStats": ("voyager_index.index", "IndexStats"),
    "BM25Config": ("voyager_index.config", "BM25Config"),
    "FusionConfig": ("voyager_index.config", "FusionConfig"),
    "IndexConfig": ("voyager_index.config", "IndexConfig"),
    "Neo4jConfig": ("voyager_index.config", "Neo4jConfig"),
    "TRITON_AVAILABLE": ("voyager_index.kernels", "TRITON_AVAILABLE"),
    "fast_colbert_scores": ("voyager_index.kernels", "fast_colbert_scores"),
    "roq_maxsim_1bit": ("voyager_index.kernels", "roq_maxsim_1bit"),
    "roq_maxsim_2bit": ("voyager_index.kernels", "roq_maxsim_2bit"),
    "roq_maxsim_4bit": ("voyager_index.kernels", "roq_maxsim_4bit"),
    "roq_maxsim_8bit": ("voyager_index.kernels", "roq_maxsim_8bit"),
    "DEFAULT_MULTIMODAL_MODEL": ("voyager_index.multimodal", "DEFAULT_MULTIMODAL_MODEL"),
    "DEFAULT_MULTIMODAL_MODEL_SPEC": ("voyager_index.multimodal", "DEFAULT_MULTIMODAL_MODEL_SPEC"),
    "MultimodalModelSpec": ("voyager_index.multimodal", "MultimodalModelSpec"),
    "SUPPORTED_MULTIMODAL_MODELS": ("voyager_index.multimodal", "SUPPORTED_MULTIMODAL_MODELS"),
    "VllmPoolingProvider": ("voyager_index.multimodal", "VllmPoolingProvider"),
    "ColPaliConfig": ("voyager_index.search", "ColPaliConfig"),
    "ColPaliEngine": ("voyager_index.search", "ColPaliEngine"),
    "MultiModalEngine": ("voyager_index.search", "MultiModalEngine"),
    "ColbertIndex": ("voyager_index.search", "ColbertIndex"),
    "RENDERABLE_SOURCE_SUFFIXES": ("voyager_index.preprocessing", "RENDERABLE_SOURCE_SUFFIXES"),
    "SearchPipeline": ("voyager_index.search", "SearchPipeline"),
    "enumerate_renderable_documents": ("voyager_index.preprocessing", "enumerate_renderable_documents"),
    "render_documents": ("voyager_index.preprocessing", "render_documents"),
    "VectorPayload": ("voyager_index.transport", "VectorPayload"),
    "decode_payload": ("voyager_index.transport", "decode_payload"),
    "encode_roq_payload": ("voyager_index.transport", "encode_roq_payload"),
    "encode_vector_payload": ("voyager_index.transport", "encode_vector_payload"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'voyager_index' has no attribute {name!r}")
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
