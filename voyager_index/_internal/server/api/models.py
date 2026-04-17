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


class GraphMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


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


class _DictLikeModel(BaseModel):
    """Pydantic model with lightweight dict-style access for compatibility."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.model_dump(mode="python").keys()

    def items(self):
        return self.model_dump(mode="python").items()

    def values(self):
        return self.model_dump(mode="python").values()

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)


class GraphPolicyMetadata(_DictLikeModel):
    """Typed graph policy metadata surfaced in search responses."""

    applied: bool = Field(default=False, description="Whether the optional graph lane was applied.")
    reason: Optional[str] = Field(default=None, description="Why the graph lane was applied or skipped.")
    mode: Optional[str] = Field(default=None, description="Resolved graph mode after policy evaluation.")
    entity_count: Optional[int] = Field(default=None, description="Number of entity cues detected in the request.")
    relation_cue_count: Optional[int] = Field(
        default=None, description="Number of relation cues detected in the request."
    )
    low_agreement: Optional[bool] = Field(
        default=None, description="Whether first-stage dense and sparse routes disagreed."
    )
    low_confidence: Optional[bool] = Field(default=None, description="Whether first-stage confidence was too low.")
    graph_available: Optional[bool] = Field(
        default=None, description="Whether the graph sidecar was healthy and available."
    )
    query_class: Optional[str] = Field(
        default=None, description="Policy classification such as ordinary, relation, or graph_native."
    )
    trigger_reasons: List[str] = Field(default_factory=list, description="Triggers that caused graph activation.")
    mandatory_reason: Optional[str] = Field(default=None, description="Mandatory workflow override, when present.")


class GraphProvenanceRecord(_DictLikeModel):
    """Typed graph provenance record for a rescued or graph-scored target."""

    reason: Optional[str] = Field(default=None, description="Primary graph reason for this target.")
    lanes: List[str] = Field(
        default_factory=list, description="Graph lanes contributing to this target, e.g. graph_local."
    )
    graph_score: Optional[float] = Field(default=None, description="Merged graph score for this target.")
    community_ids: List[str] = Field(
        default_factory=list, description="Community identifiers contributing to this target."
    )
    support_count: Optional[float] = Field(default=None, description="Support count or neighborhood support strength.")
    path: Optional[List[str]] = Field(
        default=None, description="Optional explain path labels when graph_explain is enabled."
    )


class GraphSearchSummary(_DictLikeModel):
    """Structured summary of post-retrieval graph augmentation."""

    model_config = ConfigDict(extra="allow")

    reason: Optional[str] = Field(default=None, description="Summary reason emitted by the graph sidecar or policy.")
    graph_available: Optional[bool] = Field(
        default=None, description="Whether the graph sidecar was healthy for this request."
    )
    graph_applied: Optional[bool] = Field(default=None, description="Whether graph augmentation actually ran.")
    graph_result_count: Optional[int] = Field(
        default=None, description="Total graph-scored candidates returned by the sidecar."
    )
    added_candidate_count: Optional[int] = Field(
        default=None, description="How many candidates were added additively beyond the base order."
    )
    merge_mode: Optional[str] = Field(
        default=None, description="Merge behavior for graph candidates. The shipped mode is additive."
    )
    invoked_after_first_stage: Optional[bool] = Field(
        default=None, description="Whether graph augmentation ran only after first-stage retrieval."
    )
    base_order_preserved: Optional[bool] = Field(
        default=None, description="Whether the base retrieval order was preserved ahead of graph rescues."
    )
    local_budget: Optional[int] = Field(default=None, description="Configured local rescue budget for this request.")
    community_budget: Optional[int] = Field(
        default=None, description="Configured community rescue budget for this request."
    )
    evidence_budget: Optional[int] = Field(
        default=None, description="Configured linked-evidence rescue budget for this request."
    )
    max_hops: Optional[int] = Field(default=None, description="Configured graph hop limit for this request.")
    rescued_ids: List[int] = Field(
        default_factory=list, description="Document ids appended by graph rescue when available."
    )


class GraphSearchMetadata(_DictLikeModel):
    """Typed graph metadata surfaced under search response metadata."""

    model_config = ConfigDict(extra="allow")

    graph_applied: Optional[bool] = Field(default=None, description="Whether the optional graph lane was applied.")
    reason: Optional[str] = Field(default=None, description="Top-level graph reason for this request.")
    policy: Optional[GraphPolicyMetadata] = Field(default=None, description="Resolved graph policy decision.")
    summary: Optional[GraphSearchSummary] = Field(
        default=None, description="Post-retrieval graph augmentation summary."
    )
    provenance: Dict[str, GraphProvenanceRecord] = Field(
        default_factory=dict,
        description="Optional graph provenance keyed by external or logical target id.",
    )


class SearchResponseMetadata(_DictLikeModel):
    """Structured metadata returned alongside search results."""

    model_config = ConfigDict(extra="allow")

    graph: Optional[GraphSearchMetadata] = Field(
        default=None,
        description="Optional Latence graph metadata including policy, additive merge summary, and provenance.",
    )
    solver: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional solver feature summary for optimized dense or multimodal flows.",
    )


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
                "vector": [0.1, 0.2, 0.3],
                "query_text": "service c lineage policy",
                "query_payload": {
                    "ontology_terms": ["Service C", "Export Control"],
                    "workflow_type": "compliance",
                },
                "graph_mode": "auto",
                "graph_local_budget": 4,
                "graph_community_budget": 4,
                "graph_evidence_budget": 8,
                "graph_explain": True,
                "dense_hybrid_mode": "rrf",
                "top_k": 5,
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
    query_text: Optional[str] = Field(
        default=None,
        description="Optional sparse text query. Dense collections accept it over HTTP; shard, late-interaction, and multimodal collections remain vector-only over HTTP.",
    )
    query_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional query-side metadata forwarded to solver refinement and the optional Latence graph lane "
            "(for example ontology hints, graph_required flags, tenant_graph_policy, workflow_type, or refine options). "
            "This is the preferred way to steer graph policy on vector-only routes such as shard HTTP search."
        ),
    )
    graph_mode: GraphMode = Field(
        default=GraphMode.OFF,
        description=(
            "Optional LatenceAI premium graph routing mode. "
            "'off' keeps the base OSS path only, 'auto' applies post-retrieval graph augmentation when the query looks graph-shaped, "
            "and 'force' requires the premium graph sidecar when available. Requires the optional voyager-index[latence-graph] extra for the full premium lane."
        ),
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
    graph_max_hops: Optional[int] = Field(
        default=None,
        ge=1,
        le=4,
        description="Optional maximum hop depth for the post-retrieval Latence graph sidecar expansion.",
    )
    graph_local_budget: Optional[int] = Field(
        default=None,
        ge=0,
        le=128,
        description="Optional additive budget for local entity or neighborhood graph candidate rescue.",
    )
    graph_community_budget: Optional[int] = Field(
        default=None,
        ge=0,
        le=128,
        description="Optional additive budget for high-level community or thematic graph candidate rescue.",
    )
    graph_evidence_budget: Optional[int] = Field(
        default=None,
        ge=0,
        le=256,
        description="Optional additive budget for graph-linked evidence candidates fetched after graph expansion.",
    )
    graph_explain: bool = Field(
        default=False,
        description="Include graph provenance details in response metadata when the optional Latence graph lane is used.",
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "id": "doc-1",
                        "score": 1.0,
                        "rank": 1,
                        "payload": {"text": "Team Alpha owns Service B."},
                    }
                ],
                "total": 2,
                "time_ms": 4.2,
                "metadata": {
                    "graph": {
                        "graph_applied": True,
                        "reason": "entity_heavy",
                        "policy": {"applied": True, "mode": "auto", "reason": "entity_heavy"},
                        "summary": {
                            "merge_mode": "additive",
                            "invoked_after_first_stage": True,
                            "added_candidate_count": 1,
                            "base_order_preserved": True,
                        },
                        "provenance": {
                            "doc-2": {
                                "reason": "graph_local",
                                "lanes": ["graph_local"],
                                "graph_score": 0.93,
                            }
                        },
                    }
                },
            }
        }
    )

    results: List[ScoredPoint] = Field(..., description="Scored results")
    total: int = Field(..., description="Total results returned")
    time_ms: float = Field(..., description="Search time in milliseconds")
    objective_score: Optional[float] = Field(default=None, description="Solver objective score")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens in result")
    metadata: SearchResponseMetadata = Field(
        default_factory=SearchResponseMetadata,
        description="Optional structured response metadata. Graph details are returned under metadata.graph with policy, summary, and provenance.",
    )


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
    runtime_capabilities: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additive shard/runtime capability metadata including active fallbacks.",
    )
    graph_health: Optional[str] = Field(
        default=None, description="Optional Latence graph sidecar health for this collection."
    )
    graph_dataset_id: Optional[str] = Field(
        default=None, description="Latence dataset identifier backing the optional graph lane."
    )
    graph_contract_version: Optional[str] = Field(
        default=None, description="Version of the stored graph contract bundle."
    )
    graph_sync_status: Optional[str] = Field(
        default=None, description="Most recent graph Dataset Intelligence sync status."
    )
    graph_sync_reason: Optional[str] = Field(
        default=None, description="Reason associated with the latest graph sync status."
    )
    graph_last_sync_at: Optional[str] = Field(default=None, description="Timestamp of the latest graph sync attempt.")
    graph_last_successful_sync_at: Optional[str] = Field(
        default=None,
        description="Timestamp of the latest successful graph sync.",
    )
    graph_sync_job_id: Optional[str] = Field(
        default=None, description="Dataset Intelligence job id when the latest sync created one."
    )


class CollectionListResponse(BaseModel):
    """List of collections."""

    collections: List[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    collections: int
    gpu_available: bool
    runtime_capabilities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global runtime capability probe results for optional shard backends.",
    )


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


class GroundednessSegmentationMode(str, Enum):
    SENTENCE = "sentence"
    SENTENCE_PACKED = "sentence_packed"
    PARAGRAPH = "paragraph"


class GroundednessPrimaryMetric(str, Enum):
    REVERSE_CONTEXT = "reverse_context"
    TRIANGULAR = "triangular"


class GroundednessRequest(BaseModel):
    """Beta post-generation groundedness scoring request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_ids": ["doc-1", "doc-7"],
                "query_text": "When was Teardrops released in the United States?",
                "response_text": "Teardrops was released in the United States on 20 July 1981.",
                "evidence_limit": 5,
                "primary_metric": "reverse_context",
                "raw_context_chunk_tokens": 256,
            }
        }
    )

    response_text: str = Field(
        ...,
        min_length=1,
        description="Generated response text to score for groundedness / hallucination detection.",
    )
    query_text: Optional[str] = Field(
        default=None,
        description="Optional query text used for query-conditioned diagnostic channels and grounded coverage.",
    )
    chunk_ids: Optional[List[Union[str, int]]] = Field(
        default=None,
        description="Preferred production fast path: external chunk ids whose stored support vectors should be reused.",
    )
    raw_context: Optional[str] = Field(
        default=None,
        description="Compatibility fallback: raw context text to segment and re-encode on demand.",
    )
    segmentation_mode: GroundednessSegmentationMode = Field(
        default=GroundednessSegmentationMode.SENTENCE_PACKED,
        description=(
            "How raw_context should be segmented into support units before scoring. "
            "The default sentence_packed mode packs adjacent sentences into token-budgeted windows."
        ),
    )
    raw_context_chunk_tokens: int = Field(
        default=256,
        ge=1,
        le=8192,
        description=(
            "Approximate token budget for packed raw_context support windows. "
            "Applies to raw_context only and is used by sentence_packed mode. "
            "Budgets above the active encoder limit may trigger warnings and truncation."
        ),
    )
    primary_metric: GroundednessPrimaryMetric = Field(
        default=GroundednessPrimaryMetric.REVERSE_CONTEXT,
        description="Primary scalar score exposed as the headline groundedness metric. The shipped Beta default is reverse_context.",
    )
    evidence_limit: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Maximum top evidence links to return in the bounded sparse response.",
    )
    debug_dense_matrices: bool = Field(
        default=False,
        description="Include dense token similarity matrices when payload size permits. Intended for debugging, not default UI payloads.",
    )
    include_triangular_diagnostics: bool = Field(
        default=True,
        description="When query_text is provided, include optional query-conditioned diagnostics such as triangular groundedness, echo, and grounded coverage.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional groundedness encoder override for response/query/raw-context encoding.",
    )
    query_prompt_name: Optional[str] = Field(
        default=None,
        description="Optional asymmetric prompt name for query encoding, e.g. query.",
    )
    document_prompt_name: Optional[str] = Field(
        default=None,
        description="Optional asymmetric prompt name for response/raw_context encoding, e.g. document.",
    )
    verification_samples: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional caller-supplied alternate responses drawn from the same generator "
            "at sampling temperature > 0. When present and semantic-entropy fusion is "
            "configured (VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY > 0 plus an NLI "
            "model), the service clusters the samples by bidirectional entailment and "
            "surfaces a semantic-entropy channel."
        ),
    )
    content_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional structured-source hint used by the structured-source adapter "
            "(Phase I). Known values: application/json, text/markdown, application/json+schema. "
            "When omitted the adapter auto-detects JSON and markdown table sources."
        ),
    )
    risk_band_stratum: Optional[str] = Field(
        default=None,
        description=(
            "Optional failure-mode hint consumed by the calibrated risk-band "
            "classifier (Phase H). Allowed values match the calibrated strata, "
            "typically one of: 'entity_swap', 'date_swap', 'number_swap', "
            "'unit_swap', 'negation', 'role_swap', 'partial', or 'default' "
            "(the conservative maximum). When omitted the classifier uses the "
            "hardest calibrated threshold so the green band stays honest."
        ),
    )

    @model_validator(mode="after")
    def validate_input_modes(self) -> "GroundednessRequest":
        has_chunk_ids = bool(self.chunk_ids)
        has_raw_context = bool((self.raw_context or "").strip())
        if has_chunk_ids == has_raw_context:
            raise ValueError("Provide exactly one of 'chunk_ids' or 'raw_context'")
        if self.primary_metric == GroundednessPrimaryMetric.TRIANGULAR and not (self.query_text or "").strip():
            raise ValueError("triangular primary_metric requires query_text")
        return self


class GroundednessScores(BaseModel):
    """Aggregate groundedness scores returned by the Beta endpoint."""

    primary_name: str
    primary_score: float
    reverse_context: float
    reverse_context_calibrated: Optional[float] = None
    literal_guarded: Optional[float] = None
    literal_mismatch_count: Optional[int] = None
    literal_match_count: Optional[int] = None
    literal_total_count: Optional[int] = None
    nli_aggregate: Optional[float] = None
    nli_claim_count: Optional[int] = None
    nli_skipped_count: Optional[int] = None
    groundedness_v2: Optional[float] = None
    consensus_hardened: Optional[float] = None
    reverse_query_context: Optional[float] = None
    triangular: Optional[float] = None
    echo_mean: Optional[float] = None
    grounded_coverage: Optional[float] = None
    null_bank_size: Optional[int] = None
    semantic_entropy_aggregate: Optional[float] = None
    semantic_entropy_raw: Optional[float] = None
    semantic_entropy_sample_count: Optional[int] = None
    structured_source_guarded: Optional[float] = None
    structured_source_detected: Optional[bool] = None
    risk_band: Optional[str] = None


class GroundednessLiteral(BaseModel):
    """A narrow-scope literal extracted from response or support text."""

    kind: str
    value: str
    normalized: str
    start: int
    end: int


class GroundednessLiteralDiagnostics(BaseModel):
    """Per-request literal extraction and matching diagnostics."""

    response_literals: List[GroundednessLiteral] = Field(default_factory=list)
    matches: List[GroundednessLiteral] = Field(default_factory=list)
    mismatches: List[GroundednessLiteral] = Field(default_factory=list)


class GroundednessNLIAtom(BaseModel):
    """Per-atom entailment record produced by atomic-fact decomposition (Phase F3)."""

    atom_index: int
    text: str
    char_start: int
    char_end: int
    entailment: float
    neutral: float
    contradiction: float
    score: float
    skipped: bool
    skip_reason: Optional[str] = None
    premise_count: int


class GroundednessNLIClaim(BaseModel):
    """Per-claim entailment record returned by the NLI verifier."""

    index: int
    text: str
    char_start: int
    char_end: int
    entailment: float
    neutral: float
    contradiction: float
    score: float
    skipped: bool
    skip_reason: Optional[str] = None
    premise_count: int
    atoms: List[GroundednessNLIAtom] = Field(default_factory=list)


class GroundednessNLIDiagnostics(BaseModel):
    """Per-request NLI verification diagnostics."""

    aggregate_score: Optional[float] = None
    claims: List[GroundednessNLIClaim] = Field(default_factory=list)


class GroundednessSemanticEntropyCluster(BaseModel):
    """One equivalence cluster emitted by the semantic-entropy peer."""

    cluster_id: int
    size: int
    representative: str


class GroundednessSemanticEntropyDiagnostics(BaseModel):
    """Per-request semantic-entropy (Phase G) diagnostics."""

    aggregate_score: Optional[float] = None
    entropy_raw: Optional[float] = None
    sample_count: int = 0
    cluster_count: int = 0
    clusters: List[GroundednessSemanticEntropyCluster] = Field(default_factory=list)


class GroundednessStructuredTripleMatch(BaseModel):
    """A source/response triple match emitted by the structured-source adapter."""

    subject: str
    predicate: str
    object: str
    matched: bool
    mismatch_kind: Optional[str] = None


class GroundednessStructuredDiagnostics(BaseModel):
    """Per-request structured-source (Phase I) diagnostics."""

    source_format: Optional[str] = None
    source_triple_count: int = 0
    response_triple_count: int = 0
    matches: List[GroundednessStructuredTripleMatch] = Field(default_factory=list)
    mismatches: List[GroundednessStructuredTripleMatch] = Field(default_factory=list)


class GroundednessQueryToken(BaseModel):
    """Per-query-token grounded coverage diagnostics."""

    index: int
    token: str
    coverage: float


class GroundednessResponseToken(BaseModel):
    """Per-response-token groundedness diagnostics."""

    index: int
    token: str
    weight: float
    reverse_context: float
    reverse_context_calibrated: Optional[float] = None
    reverse_context_z: Optional[float] = None
    null_mean: Optional[float] = None
    null_std: Optional[float] = None
    nli_score: Optional[float] = None
    consensus_hardened: Optional[float] = None
    support_unit_hits_above_threshold: Optional[int] = None
    support_unit_soft_breadth: Optional[float] = None
    effective_support_units: Optional[float] = None
    reverse_query_context: Optional[float] = None
    triangular: Optional[float] = None
    echo: Optional[float] = None
    support_unit_index: Optional[int] = None
    support_token_index: Optional[int] = None
    support_token: Optional[str] = None
    chunk_id: Optional[Union[str, int]] = None
    heatmap_score: float


class GroundednessSupportUnit(BaseModel):
    """Support unit returned for chunk- or raw-context-mode groundedness."""

    index: int
    support_id: str
    chunk_id: Optional[Union[str, int]] = None
    source_mode: str
    text: str
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    token_count: int
    tokens: List[str]
    token_scores: List[float]
    score: float
    matched_response_tokens: int


class GroundednessEvidence(BaseModel):
    """Top evidence alignment between a response token and a support token."""

    response_token_index: int
    response_token: str
    support_unit_index: int
    support_token_index: int
    support_token: str
    chunk_id: Optional[Union[str, int]] = None
    metric: str
    score: float


class GroundednessEligibility(BaseModel):
    """Eligibility and fidelity metadata for groundedness trust boundaries."""

    collection_kind: CollectionKind
    vector_source: str
    storage_compression: Optional[str] = None
    quantization_mode: Optional[str] = None
    dequantized: bool
    user_facing_supported: bool
    warnings: List[str] = Field(default_factory=list)


class GroundednessDebugPayload(BaseModel):
    """Optional dense matrices for debugging or custom heatmaps."""

    response_to_support: Optional[List[List[float]]] = None
    response_to_query: Optional[List[List[float]]] = None
    triangular_gated: Optional[List[List[float]]] = None


class GroundednessResponse(BaseModel):
    """Beta groundedness scoring response with heatmap-ready sparse data."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection": "tutorial-li",
                "mode": "chunk_ids",
                "model": "lightonai/GTE-ModernColBERT-v1",
                "scores": {
                    "primary_name": "reverse_context",
                    "primary_score": 0.97,
                    "reverse_context": 0.97,
                    "consensus_hardened": 0.96,
                    "reverse_query_context": 0.98,
                    "triangular": 0.82,
                },
                "response_tokens": [
                    {
                        "index": 0,
                        "token": "Teardrops",
                        "weight": 1.0,
                        "reverse_context": 0.99,
                        "consensus_hardened": 0.98,
                        "support_unit_hits_above_threshold": 2,
                        "support_unit_soft_breadth": 1.7,
                        "effective_support_units": 1.8,
                        "heatmap_score": 0.99,
                        "support_unit_index": 0,
                        "support_token_index": 3,
                        "support_token": "Teardrops",
                        "chunk_id": "doc-7",
                    }
                ],
                "support_units": [
                    {
                        "index": 0,
                        "support_id": "doc-7",
                        "chunk_id": "doc-7",
                        "source_mode": "chunk_ids",
                        "text": "Teardrops is a single by George Harrison, released on 20 July 1981 in the United States.",
                        "token_count": 16,
                        "tokens": ["Teardrops", "is", "a", "single"],
                        "token_scores": [0.99, 0.35, 0.0, 0.71],
                        "score": 0.93,
                        "matched_response_tokens": 4,
                    }
                ],
                "top_evidence": [
                    {
                        "response_token_index": 0,
                        "response_token": "Teardrops",
                        "support_unit_index": 0,
                        "support_token_index": 3,
                        "support_token": "Teardrops",
                        "chunk_id": "doc-7",
                        "metric": "reverse_context",
                        "score": 0.99,
                    }
                ],
                "eligibility": {
                    "collection_kind": "late_interaction",
                    "vector_source": "stored_vectors",
                    "dequantized": True,
                    "user_facing_supported": True,
                    "warnings": [],
                },
                "time_ms": 4.2,
            }
        }
    )

    collection: str
    mode: str
    model: Optional[str] = None
    scores: GroundednessScores
    response_tokens: List[GroundednessResponseToken]
    support_units: List[GroundednessSupportUnit]
    top_evidence: List[GroundednessEvidence]
    eligibility: GroundednessEligibility
    query_tokens: Optional[List[GroundednessQueryToken]] = None
    debug: Optional[GroundednessDebugPayload] = None
    warnings: List[str] = Field(default_factory=list)
    literal_diagnostics: Optional[GroundednessLiteralDiagnostics] = None
    nli_diagnostics: Optional[GroundednessNLIDiagnostics] = None
    semantic_entropy_diagnostics: Optional[GroundednessSemanticEntropyDiagnostics] = None
    structured_diagnostics: Optional[GroundednessStructuredDiagnostics] = None
    time_ms: float


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
