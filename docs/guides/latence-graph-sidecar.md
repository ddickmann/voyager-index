# Latence Graph Sidecar

`voyager-index` keeps the serving hot path in the OSS retrieval engine and adds
graph awareness through an optional LatenceAI sidecar. This preserves the
single-node retrieval story while making the graph lane a premium, failure-
isolated feature instead of a tax on every query.

## Product Split

| Layer | What lives here |
| --- | --- |
| OSS retrieval core | multimodal preprocessing, embeddings/model-serving seams, late-interaction index, BM25S, quantization, fusion, and optional solver packing |
| Optional Latence graph plane | `LatenceGraphSidecar`, graph-aware candidate rescue, graph-aware solver features, provenance, and Dataset Intelligence sync metadata |

The important boundary is that the OSS engine stays usable with `graph_mode="off"`
or without the commercial dependency installed. The graph lane is additive and
post-retrieval.

## What Graph Data Comes From

The optional graph lane is grounded in **Latence graph data derived from the
customer corpus**, not in a disconnected hand-authored graph.

Publicly, the model is:

- indexed targets remain the source of truth for retrieval and serving
- Latence Dataset Intelligence or equivalent sync paths derive graph fragments
  from those targets
- the sidecar stores target-linked graph contracts such as entities, relations,
  community assignments, and evidence links
- search-time graph rescue always resolves back to real target IDs and real
  retrieval candidates

What stays intentionally abstract in public docs:

- exact extraction thresholds and scoring weights
- proprietary policy lexicons and trigger heuristics
- non-public SDK and persistence internals

## Public Algorithm Summary

The graph lane follows the same architectural cut described in the README and
production guide:

1. base retrieval runs first through LEMUR-routed dense search, with BM25 when the route supports it
2. the graph policy checks whether graph augmentation is worth paying for
3. the sidecar resolves entities from the query plus first-stage documents
4. local neighborhoods, community-level themes, and linked evidence are fetched
5. graph-derived candidates and feature payloads are appended before final solver packing

That is the practical LightRAG-inspired shape: low-level entity/relation rescue,
high-level thematic/community rescue, and additive evidence stitching without
turning graph traversal into the default first-stage router.

## Retrieval Flow

```text
query
  -> normalize / detect entities
  -> dense retrieval (LEMUR + exact late interaction)
  -> BM25S in parallel when query_text is available
  -> merge / dedupe first-stage candidates
  -> graph policy check
       off  -> solver / result packing
       on   -> Latence graph augmentation
                 - resolve entities from query + first-stage docs
                 - low-level local expansion
                 - high-level community retrieval
                 - linked evidence rescue
                 - graph feature injection
  -> final solver / context packing
  -> answer context or ranked results
```

This is intentionally aligned with the useful part of LightRAG:

- low-level entity and relation retrieval
- high-level community or thematic retrieval
- incremental graph freshness instead of full rebuild dependency

It does **not** replace the base retrieval engine with graph traversal.

## Query Policy

The optional graph lane is controlled by `graph_mode` plus a query policy:

- `off`: disable graph augmentation completely
- `auto`: enable graph only for graph-shaped queries
- `force`: always invoke the graph lane when the sidecar is healthy

The shipped policy considers:

- entity-heavy queries
- relation cues such as `related`, `depends`, `impact`, `why`, `caused`
- low agreement between dense and BM25 first-stage routes
- low first-stage confidence
- mandatory workflow hints such as compliance or lineage

Known-item, FAQ, and short navigational queries stay graph-off in `auto`.

## Sidecar Responsibilities

The sidecar boundary is intentionally narrow:

- `resolve_entities(query, docs) -> node_ids`
- `expand_local(node_ids, budget) -> related nodes, edges, target IDs`
- `retrieve_community(node_ids, budget) -> community summaries, hubs, target IDs`
- `linked_evidence(node_ids, budget) -> additive evidence targets`
- `graph_features(query, candidate_ids) -> solver-facing feature payloads`

`voyager-index` consumes the sidecar as a retrieval helper. The graph contributes
candidate rescue, context assembly features, and provenance. It does not replace
BM25, LEMUR, or exact MaxSim.

## HTTP Controls

The reference API exposes the optional lane through the standard search contract:

- `graph_mode`
- `graph_max_hops`
- `graph_local_budget`
- `graph_community_budget`
- `graph_evidence_budget`
- `graph_explain`

These budgets are independent. For example, evidence rescue can be enabled while
local expansion stays at zero.

Important route note:

- dense HTTP search can use `query_text` plus `query_payload`
- shard, late-interaction, and multimodal HTTP search remain vector-only, so
  `query_payload` is the preferred way to provide ontology hints, workflow hints,
  and other graph-policy signals on those routes

## Provenance And Health

When `graph_explain=true`, search responses include graph metadata under
`metadata.graph`:

- `policy`: why the lane was applied or skipped
- `summary`: additive merge details, rescue counts, and hop budgets
- `provenance`: lane tags such as `graph_local` or `graph_community`

Collection inspection exposes sidecar lifecycle state through
`GET /collections/{name}/info`:

- `graph_health`
- `graph_dataset_id`
- `graph_contract_version`
- `graph_sync_status`
- `graph_sync_reason`
- `graph_last_sync_at`
- `graph_last_successful_sync_at`
- `graph_sync_job_id`

Readiness checks surface degraded or failed sync states so operators can see the
premium lane status without breaking the OSS retrieval path.

## Dataset Intelligence Lifecycle

Graph freshness follows the collection lifecycle:

- collection upserts normalize graph fragments and append them to the sidecar
- deletes remove target-to-node mappings locally and keep the search path safe
- dataset delta ingestion applies targeted graph updates without a full rebuild
- restart recovery reloads the persisted sidecar state before search resumes

The sidecar stores sync history and last-successful timestamps so graph freshness
is observable instead of implicit.

## Evidence And Validation

The repo includes graph-specific validation for:

- LightRAG-style local plus community rescue behavior
- additive merge semantics
- dense, shard, late-interaction, and multimodal parity
- persistence and degraded restart handling
- route-conformance and latency evidence in `tools/benchmarks/benchmark_latence_graph_quality.py`

Use the graph benchmark directly with:

```bash
python tools/benchmarks/benchmark_latence_graph_quality.py --mode benchmark
python tools/benchmarks/benchmark_latence_graph_quality.py --mode ablation
```

The representative public metrics live in [Benchmarks And Methodology](../benchmarks.md).
This guide stays focused on the graph contract, architecture boundary, data
origin, and operational behavior rather than duplicating the benchmark report.
