"""
Pydantic models for the voyager-index reference API.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


class CollectionKind(str, Enum):
    DENSE = "dense"
    LATE_INTERACTION = "late_interaction"
    MULTIMODAL = "multimodal"
    SHARD = "shard"


class SearchStrategy(str, Enum):
    SIMPLE = "simple"
    OPTIMIZED = "optimized"
    SHARD_ROUTED = "shard_routed"


class DenseHybridMode(str, Enum):
    RRF = "rrf"
    TABU = "tabu"


class MultimodalOptimizeMode(str, Enum):
    AUTO = "auto"
    MAXSIM_ONLY = "maxsim_only"
    SOLVER_PREFILTER_MAXSIM = "solver_prefilter_maxsim"
    MAXSIM_THEN_SOLVER = "maxsim_then_solver"


class ScreeningMode(str, Enum):
    GEM = "gem"
    NONE = "none"


class TransportVectorPayload(BaseModel):
    """Base64 vector transport payload shared with the optimizer contract."""

    model_config = ConfigDict(extra="forbid")

    encoding: str
    shape: List[int]
    data_b64: str
    dtype: str = "float32"
    num_bits: Optional[int] = None
    block_size: Optional[int] = None
    num_rounds: Optional[int] = None
    seed: int = 42
    scales_b64: Optional[str] = None
    offsets_b64: Optional[str] = None
    norms_sq_b64: Optional[str] = None
    code_sums_b64: Optional[str] = None
    code_shape: Optional[List[int]] = None


class PointVector(BaseModel):
    """Vector or multivector payload for a point."""

    id: Union[str, int] = Field(..., description="External point ID")
    vector: Optional[Union[List[float], TransportVectorPayload]] = Field(
        default=None,
        description="Single embedding vector as raw floats or a base64 transport payload",
    )
    vectors: Optional[Union[List[List[float]], TransportVectorPayload]] = Field(
        default=None,
        description="Multi-vector embedding matrix as raw floats or a base64 transport payload",
    )
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Metadata payload")

    @model_validator(mode="after")
    def validate_vectors(self) -> "PointVector":
        if self.vector is None and self.vectors is None:
            raise ValueError("Either 'vector' or 'vectors' must be provided")
        if self.vector is not None and self.vectors is not None:
            raise ValueError("Provide only one of 'vector' or 'vectors'")
        return self


class ScoredPoint(BaseModel):
    """Search result point with score."""

    id: Union[str, int]
    score: float
    rank: int
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    vectors: Optional[List[List[float]]] = None


class CreateCollectionRequest(BaseModel):
    """Create a new collection."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dimension": 128,
                "kind": "dense",
                "distance": "cosine",
            }
        }
    )

    name: Optional[str] = Field(default=None, description="Optional collection name mirror")
    dimension: int = Field(..., gt=0, description="Embedding dimension")
    kind: CollectionKind = Field(default=CollectionKind.DENSE, description="Collection runtime type")
    distance: DistanceMetric = Field(default=DistanceMetric.COSINE, description="Distance metric")
    m: int = Field(default=16, description="HNSW M parameter")
    ef_construction: int = Field(default=200, description="HNSW ef_construction")
    storage_mode: str = Field(default="sync", description="Index storage mode for late-interaction collections")
    n_shards: Optional[int] = Field(default=None, ge=1, le=65536, description="Number of shards (shard collections)")
    k_candidates: Optional[int] = Field(
        default=None, ge=1, le=100000, description="LEMUR candidate count (shard collections)"
    )
    use_colbandit: bool = Field(default=False, description="Enable Col-Bandit reranking (shard collections)")
    compression: Optional[str] = Field(default=None, description="Shard storage compression: fp16, int8, or roq4")
    quantization_mode: Optional[str] = Field(
        default=None, description="Shard scoring mode: int8, fp8, roq4, or empty for exact"
    )
    transfer_mode: Optional[str] = Field(
        default=None, description="Shard CPU->GPU transfer mode: pageable, pinned, or double_buffered"
    )
    router_device: Optional[str] = Field(
        default=None, description="Device used by the LEMUR router, typically cpu or cuda"
    )
    lemur_search_k_cap: Optional[int] = Field(
        default=None, ge=1, le=8192, description="Optional LEMUR search cap for shard collections"
    )
    max_docs_exact: Optional[int] = Field(
        default=None, ge=1, le=1000000, description="Optional exact-stage document cap for shard collections"
    )
    n_full_scores: Optional[int] = Field(
        default=None, ge=1, le=1000000, description="Optional shard proxy shortlist size before exact full scoring"
    )
    pinned_pool_buffers: Optional[int] = Field(
        default=None, ge=1, le=64, description="Optional pinned-memory buffer pool size for shard transfers"
    )
    pinned_buffer_max_tokens: Optional[int] = Field(
        default=None, ge=1, le=5000000, description="Optional max tokens per pinned transfer buffer for shard fetch"
    )
    gpu_corpus_rerank_topn: Optional[int] = Field(
        default=None, ge=1, le=4096, description="Optional shard GPU-corpus rerank frontier size"
    )
    n_centroid_approx: Optional[int] = Field(
        default=None, ge=0, le=1000000, description="Optional centroid-approx candidate count for shard collections"
    )
    variable_length_strategy: Optional[str] = Field(
        default=None, description="Shard variable-length exact strategy, e.g. bucketed"
    )
    max_documents: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum documents in this collection; oldest are evicted on overflow",
    )

    @model_validator(mode="after")
    def validate_kind_specific_options(self) -> "CreateCollectionRequest":
        valid_compressions = {"fp16", "int8", "roq4"}
        valid_quant_modes = {"", "none", "int8", "fp8", "roq4"}
        valid_transfer_modes = {"pageable", "pinned", "double_buffered"}
        if self.kind not in (CollectionKind.DENSE, CollectionKind.SHARD):
            if self.distance != DistanceMetric.COSINE:
                raise ValueError("Only dense and shard collections support configurable distance metrics")
            if self.m != 16 or self.ef_construction != 200:
                raise ValueError("Only dense collections support configurable HNSW parameters")
        if self.kind != CollectionKind.LATE_INTERACTION and self.storage_mode != "sync":
            raise ValueError("storage_mode is only supported for late-interaction collections")
        if self.kind != CollectionKind.SHARD:
            shard_only_values = (
                self.n_shards,
                self.k_candidates,
                self.compression,
                self.quantization_mode,
                self.transfer_mode,
                self.router_device,
                self.lemur_search_k_cap,
                self.max_docs_exact,
                self.n_full_scores,
                self.pinned_pool_buffers,
                self.pinned_buffer_max_tokens,
                self.gpu_corpus_rerank_topn,
                self.n_centroid_approx,
                self.variable_length_strategy,
            )
            if any(value is not None for value in shard_only_values):
                raise ValueError("Shard tuning fields are only supported for shard collections")
            if self.use_colbandit:
                raise ValueError("use_colbandit is only for shard collections")
        else:
            if self.compression is not None and self.compression not in valid_compressions:
                raise ValueError("compression must be one of: fp16, int8, roq4")
            if self.quantization_mode is not None and self.quantization_mode not in valid_quant_modes:
                raise ValueError("quantization_mode must be one of: '', none, int8, fp8, roq4")
            if self.transfer_mode is not None and self.transfer_mode not in valid_transfer_modes:
                raise ValueError("transfer_mode must be one of: pageable, pinned, double_buffered")
        return self


class AddPointsRequest(BaseModel):
    """Add or upsert points in a collection."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "points": [
                    {
                        "id": "doc_1",
                        "vectors": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                        "payload": {"title": "Document 1", "text": "example"},
                    }
                ]
            }
        }
    )

    points: List[PointVector] = Field(..., description="Points to add")


class DeletePointsRequest(BaseModel):
    """Delete points from a collection."""

    ids: List[Union[str, int]] = Field(..., description="Point IDs to delete")


class SearchRequest(BaseModel):
    """Search request for dense or multivector collections."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vectors": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                "top_k": 10,
                "with_payload": True,
            }
        }
    )

    vector: Optional[Union[List[float], TransportVectorPayload]] = Field(
        default=None,
        description=(
            "Single query vector as raw floats or a base64 transport payload. "
            "Also accepted as a single-token query for late-interaction and multimodal search."
        ),
    )
    vectors: Optional[Union[List[List[float]], TransportVectorPayload]] = Field(
        default=None,
        description="Multi-vector query as raw floats or a base64 transport payload",
    )
    query_text: Optional[str] = Field(default=None, description="Optional sparse text query")
    query_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional query-side metadata forwarded to solver refinement (e.g. ontology hints or refine options).",
    )
    top_k: int = Field(default=10, ge=1, le=1000, description="Number of results")
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Flat payload equality filter applied consistently across supported collection kinds",
    )
    with_payload: bool = Field(default=True, description="Include payload in results")
    with_vector: bool = Field(default=False, description="Include stored vectors in results")
    strategy: SearchStrategy = Field(default=SearchStrategy.SIMPLE, description="Search strategy")
    dense_hybrid_mode: Optional[DenseHybridMode] = Field(
        default=None,
        description="Dense collections only: use plain BM25+dense RRF fusion or Tabu solver refinement over the fused pool.",
    )
    quantization_mode: Optional[str] = Field(
        default=None,
        description="Shard collections only: override the scoring precision mode (int8, fp8, roq4, or empty for exact).",
    )
    use_colbandit: Optional[bool] = Field(
        default=None,
        description="Shard collections only: override Col-Bandit pruning for this request.",
    )
    transfer_mode: Optional[str] = Field(
        default=None,
        description="Shard collections only: override CPU->GPU transfer mode for this request.",
    )
    max_docs_exact: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000000,
        description="Shard collections only: override the exact-stage document cap for this request.",
    )
    n_full_scores: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000000,
        description="Shard collections only: override the proxy shortlist size before exact full scoring.",
    )
    lemur_search_k_cap: Optional[int] = Field(
        default=None,
        ge=1,
        le=8192,
        description="Shard collections only: override the router search cap for this request.",
    )
    gpu_corpus_rerank_topn: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Shard collections only: override the GPU corpus rerank frontier.",
    )
    n_centroid_approx: Optional[int] = Field(
        default=None,
        ge=0,
        le=1000000,
        description="Shard collections only: override centroid-approx candidate scoring count.",
    )
    variable_length_strategy: Optional[str] = Field(
        default=None,
        description="Shard collections only: override variable-length exact scheduling, e.g. bucketed.",
    )
    pinned_pool_buffers: Optional[int] = Field(
        default=None,
        ge=1,
        le=64,
        description="Shard collections only: override pinned-memory transfer buffer pool size.",
    )
    pinned_buffer_max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000000,
        description="Shard collections only: override the max tokens per pinned transfer buffer.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
        description="Optional token budget for optimized dense refinement",
    )
    max_chunks: Optional[int] = Field(
        default=None,
        ge=1,
        le=256,
        description="Optional selection cap for optimized dense refinement",
    )
    max_per_cluster: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Optional per-cluster cap for optimized solver stages.",
    )
    solver_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional solver config overrides for optimized solver-backed search.",
    )
    optimizer_policy: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Optional optimizer policy preset name or override dict (e.g. 'post_rerank_v1').",
    )
    refine_use_cross_encoder: bool = Field(
        default=False,
        description="For optimized dense search, rerank the fused pool with a cross-encoder before Tabu packing.",
    )
    refine_cross_encoder_model: Optional[str] = Field(
        default=None,
        description="Optional sentence-transformers CrossEncoder model id for dense refinement.",
    )
    refine_cross_encoder_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=512,
        description="Optional cap on candidates reranked by the cross-encoder before Tabu packing.",
    )
    refine_cross_encoder_batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=256,
        description="Optional batch size for cross-encoder reranking during optimized dense refinement.",
    )
    refine_use_nli: bool = Field(
        default=False,
        description="For optimized dense search, score the fused pool with an NLI/utility head before Tabu packing.",
    )
    refine_nli_model: Optional[str] = Field(
        default=None,
        description="Optional transformers sequence-classification model id for NLI refinement.",
    )
    refine_nli_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=512,
        description="Optional cap on candidates scored by the NLI/utility head before Tabu packing.",
    )
    refine_nli_batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=256,
        description="Optional batch size for NLI/utility scoring during optimized dense refinement.",
    )
    refine_nli_promote_base_relevance: bool = Field(
        default=False,
        description="Blend NLI utility back into base relevance before optimization (disabled by default for low-risk rollout).",
    )
    refine_confidence_gating: bool = Field(
        default=False,
        description="Skip expensive CE/NLI passes when the fused top of list is already unambiguous.",
    )
    refine_confidence_gap_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum top-score gap required to skip expensive CE/NLI refinement.",
    )
    refine_confidence_min_candidates: Optional[int] = Field(
        default=None,
        ge=2,
        le=512,
        description="Minimum candidate count before confidence gating is considered.",
    )
    multimodal_optimize_mode: Optional[MultimodalOptimizeMode] = Field(
        default=None,
        description=(
            "Optional ordering for optimized multimodal search: exact MaxSim only, "
            "solver prefilter before exact MaxSim, or exact MaxSim frontier followed by solver packing."
        ),
    )
    multimodal_candidate_budget: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Optional multimodal candidate budget for optimized search before the chosen refinement stage.",
    )
    multimodal_prefilter_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="Optional candidate count kept by the solver before final exact MaxSim reranking.",
    )
    multimodal_maxsim_frontier_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="Optional exact MaxSim frontier size passed into the solver-after packing stage.",
    )
    screening_mode: ScreeningMode = Field(
        default=ScreeningMode.GEM,
        description=(
            'Screening strategy for multimodal search. "gem" (default) runs a '
            "two-stage pipeline: GEM router screening followed by full MaxSim "
            'reranking. "none" skips screening and queries the GEM graph index '
            "directly."
        ),
    )
    ef: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="HNSW ef search parameter (dense collections only; ignored by shard engine)",
    )
    n_probes: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="IVF nprobe override (shard collections with IVF-PQ routing; ignored otherwise)",
    )

    @model_validator(mode="after")
    def validate_query(self) -> "SearchRequest":
        valid_quant_modes = {"", "none", "int8", "fp8", "roq4"}
        valid_transfer_modes = {"pageable", "pinned", "double_buffered"}
        if self.vector is None and self.vectors is None and not self.query_text:
            raise ValueError("Provide 'vector', 'vectors', or 'query_text'")
        if self.vector is not None and self.vectors is not None:
            raise ValueError("Provide only one of 'vector' or 'vectors'")
        if self.quantization_mode is not None and self.quantization_mode not in valid_quant_modes:
            raise ValueError("quantization_mode must be one of: '', none, int8, fp8, roq4")
        if self.transfer_mode is not None and self.transfer_mode not in valid_transfer_modes:
            raise ValueError("transfer_mode must be one of: pageable, pinned, double_buffered")
        return self


class OptimizeRequest(BaseModel):
    """
    Compatibility model for the deprecated collection-scoped optimize endpoint.

    The canonical OSS solver API is `/reference/optimize`, which uses the
    stateless optimizer contract rather than this legacy shape.
    """

    vector: List[float] = Field(..., description="Query vector")
    num_candidates: int = Field(default=500, description="Number of candidates to retrieve")
    max_tokens: int = Field(default=8192, description="Maximum token budget")
    alpha: float = Field(default=1.0, description="Relevance weight")
    beta: float = Field(default=0.3, description="Information density weight")
    gamma: float = Field(default=0.2, description="Document centrality weight")
    lambda_: float = Field(default=0.5, alias="lambda", description="Redundancy penalty")
    must_include_roles: Optional[List[str]] = Field(default=None, description="Required rhetorical roles")
    max_per_cluster: int = Field(default=2, description="Max chunks per semantic cluster")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "vector": [0.1, 0.2, 0.3],
                "max_tokens": 8192,
                "alpha": 1.0,
                "beta": 0.3,
            }
        },
    )


class OptimizerCandidateRequest(BaseModel):
    """Canonical candidate shape for the stateless optimizer API."""

    chunk_id: str = Field(..., description="Candidate chunk identifier")
    text: str = Field(..., description="Candidate text content")
    token_count: int = Field(..., ge=0, description="Token count contributed by this chunk")
    vectors: TransportVectorPayload = Field(..., description="Base64 vector payload for this candidate")
    fact_density: float = Field(default=0.5, description="Optional fact density score")
    centrality_score: float = Field(default=0.5, description="Optional centrality score")
    recency_score: float = Field(default=0.5, description="Optional recency score")
    auxiliary_score: float = Field(default=0.0, description="Optional auxiliary score")
    rhetorical_role: str = Field(default="unknown", description="Optional rhetorical role label")
    cluster_id: Optional[int] = Field(default=None, description="Optional cluster assignment")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata such as dense/sparse/RRF scores"
    )


class ReferenceOptimizeRequest(BaseModel):
    """Typed OpenAPI contract for the canonical `/reference/optimize` endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "invoice total due",
                "query_vectors": {
                    "encoding": "float32",
                    "shape": [1, 4],
                    "dtype": "float32",
                    "data_b64": "...",
                },
                "candidates": [
                    {
                        "chunk_id": "invoice",
                        "text": "invoice total due",
                        "token_count": 64,
                        "vectors": {
                            "encoding": "float32",
                            "shape": [1, 4],
                            "dtype": "float32",
                            "data_b64": "...",
                        },
                        "metadata": {"dense_score": 1.0, "rrf_score": 0.03},
                    }
                ],
                "constraints": {"max_tokens": 96, "max_chunks": 1},
                "solver_config": {"iterations": 16},
            }
        }
    )

    query_text: str = Field(..., description="Original user query text")
    query_vectors: TransportVectorPayload = Field(..., description="Base64 vector payload for the query embedding(s)")
    candidates: List[OptimizerCandidateRequest] = Field(..., description="Candidate chunks considered by the solver")
    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="Optimization constraints such as max_tokens or max_chunks"
    )
    solver_config: Dict[str, Any] = Field(default_factory=dict, description="Optional solver config overrides")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional request-level metadata")


class RenderDocumentsRequest(BaseModel):
    """Render local source documents into PageBundle-like image assets."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_paths": ["/data/source/invoice.pdf"],
                "output_dir": "/data/rendered-pages",
            }
        }
    )

    source_paths: List[str] = Field(
        default_factory=list,
        description="Local file or directory paths to preprocess",
    )
    source_dir: Optional[str] = Field(
        default=None,
        description="Optional root directory to scan for renderable documents",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional destination directory for rendered page images",
    )
    recursive: bool = Field(
        default=True,
        description="Recursively scan directories when source_dir or directory paths are provided",
    )

    @model_validator(mode="after")
    def validate_sources(self) -> "RenderDocumentsRequest":
        if not self.source_paths and not self.source_dir:
            raise ValueError("Provide 'source_paths' or 'source_dir'")
        return self


class RenderedPage(BaseModel):
    """Rendered page asset produced by document preprocessing."""

    page_id: str
    page_number: int
    image_path: str
    source_uri: Optional[str] = None
    text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PageBundleResponse(BaseModel):
    """PageBundle-like response emitted by the preprocessing endpoint."""

    bundle_version: str
    doc_id: str
    source_uri: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pages: List[RenderedPage]


class RenderDocumentsResponse(BaseModel):
    """Rendered document corpus summary."""

    status: str
    output_dir: str
    bundles: List[PageBundleResponse]
    skipped: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search results."""

    results: List[ScoredPoint] = Field(..., description="Scored results")
    total: int = Field(..., description="Total results returned")
    time_ms: float = Field(..., description="Search time in milliseconds")
    objective_score: Optional[float] = Field(default=None, description="Solver objective score")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens in result")


class CollectionInfo(BaseModel):
    """Collection information."""

    name: str
    kind: CollectionKind
    dimension: int
    distance: DistanceMetric
    num_points: int
    indexed: bool
    storage_mb: Optional[float] = None
    m: Optional[int] = None
    ef_construction: Optional[int] = None
    storage_mode: Optional[str] = None
    storage_path: Optional[str] = None
    n_shards: Optional[int] = None
    k_candidates: Optional[int] = None
    total_tokens: Optional[int] = None
    compression: Optional[str] = None
    quantization_mode: Optional[str] = None
    transfer_mode: Optional[str] = None
    router_device: Optional[str] = None
    use_colbandit: Optional[bool] = None
    max_docs_exact: Optional[int] = None
    n_full_scores: Optional[int] = None
    pinned_pool_buffers: Optional[int] = None
    pinned_buffer_max_tokens: Optional[int] = None
    lemur_search_k_cap: Optional[int] = None
    gpu_corpus_rerank_topn: Optional[int] = None
    n_centroid_approx: Optional[int] = None
    variable_length_strategy: Optional[str] = None
    hybrid_search: Optional[bool] = None


class CollectionListResponse(BaseModel):
    """List of collections."""

    collections: List[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    collections: int
    gpu_available: bool


class MetricsResponse(BaseModel):
    """Metric names mirrored by the Prometheus text endpoint."""

    voyager_search_requests_total: int
    voyager_search_latency_seconds_sum: float
    voyager_search_latency_seconds_count: int
    voyager_collections_total: int
    voyager_points_total: int


class ShardInfo(BaseModel):
    """Metadata for a single shard."""

    shard_id: int = Field(ge=0)
    num_docs: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    avg_tokens: float = Field(ge=0)
    p95_tokens: float = Field(ge=0)


class ShardListResponse(BaseModel):
    """List of shards in a shard collection."""

    collection: str
    num_shards: int
    shards: List[ShardInfo]


class WalStatusResponse(BaseModel):
    """WAL status for a shard collection."""

    collection: str
    wal_entries: int = Field(ge=0)
    memtable_docs: int = Field(ge=0)
    memtable_tombstones: int = Field(ge=0)


class ScrollRequest(BaseModel):
    """Scroll (paginate) through documents in a collection."""

    limit: int = Field(default=100, ge=1, le=10000, description="Page size")
    offset: int = Field(default=0, ge=0, description="Offset to start from")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional payload filters")


class ScrollResponse(BaseModel):
    """Scroll results."""

    ids: List[int]
    next_offset: Optional[int] = None


class RetrieveRequest(BaseModel):
    """Retrieve specific documents by ID."""

    ids: List[int] = Field(..., description="Document IDs to retrieve")
    with_vector: bool = Field(default=False, description="Include stored vectors")
    with_payload: bool = Field(default=True, description="Include payload")


class RetrieveResponse(BaseModel):
    """Retrieve results."""

    points: List[Dict[str, Any]]


class SearchBatchRequest(BaseModel):
    """Batch search request."""

    searches: List[SearchRequest] = Field(..., description="List of search requests")


class SearchBatchResponse(BaseModel):
    """Batch search results."""

    results: List[SearchResponse]


class ErrorResponse(BaseModel):
    """Structured error response (JSON {code, message, details})."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error context")
    request_id: Optional[str] = Field(default=None, description="Request trace ID")

    error: Optional[str] = Field(default=None, description="Legacy error field (deprecated)")
    detail: Optional[str] = Field(default=None, description="Legacy detail field (deprecated)")
    status_code: Optional[int] = Field(default=None, description="Legacy status code (deprecated)")


# ------------------------------------------------------------------
# Async mutation task
# ------------------------------------------------------------------


class MutationTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MutationTaskResponse(BaseModel):
    """Returned on 202 Accepted for async mutations."""

    task_id: str = Field(..., description="Unique task identifier")
    status: MutationTaskStatus = Field(default=MutationTaskStatus.PENDING)
    message: str = Field(default="Accepted for processing")


class TaskStatusResponse(BaseModel):
    """Status of an async mutation task."""

    task_id: str
    status: MutationTaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


# ------------------------------------------------------------------
# Payload / metadata CRUD
# ------------------------------------------------------------------


class SetPayloadRequest(BaseModel):
    """Set or merge payload for specific points."""

    payload: Dict[str, Any] = Field(..., description="Payload fields to set/merge")
    points: List[Union[str, int]] = Field(..., description="Point IDs to update")


class DeletePayloadKeysRequest(BaseModel):
    """Delete specific payload keys from points."""

    keys: List[str] = Field(..., description="Payload keys to remove")
    points: List[Union[str, int]] = Field(..., description="Point IDs to update")


class ClearPayloadRequest(BaseModel):
    """Clear all payload fields from points."""

    points: List[Union[str, int]] = Field(..., description="Point IDs to clear payload from")


# ------------------------------------------------------------------
# Encode / Rerank
# ------------------------------------------------------------------


class EncodeRequest(BaseModel):
    """Encode text or images into embeddings."""

    texts: Optional[List[str]] = Field(default=None, description="Texts to encode")
    images: Optional[List[str]] = Field(default=None, description="Image paths or URLs to encode")
    model: Optional[str] = Field(default=None, description="Model name override")
    truncate: bool = Field(default=True, description="Truncate inputs to model max length")

    @model_validator(mode="after")
    def validate_input(self) -> "EncodeRequest":
        if not self.texts and not self.images:
            raise ValueError("Provide 'texts' or 'images'")
        return self


class EncodeResponse(BaseModel):
    """Encoding results."""

    embeddings: List[List[List[float]]] = Field(..., description="Embeddings per input (multi-vector)")
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class RerankRequest(BaseModel):
    """Rerank documents against a query."""

    query: str = Field(..., description="Query text")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_k: int = Field(default=10, ge=1, le=1000, description="Number of results")
    model: Optional[str] = Field(default=None, description="Model name override")


class RerankResult(BaseModel):
    """Single rerank result."""

    index: int
    score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    """Rerank results."""

    results: List[RerankResult]
    model: Optional[str] = None
