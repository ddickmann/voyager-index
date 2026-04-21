# Enterprise Control Plane Boundary

`colsearch` now has a clear data-plane story. The remaining enterprise layer
is the control plane: the services that decide how models, indexes, ingestion
jobs, and evaluation gates are operated over time.

This document makes that boundary explicit so the OSS runtime is not mistaken
for a full orchestration platform.

## Data Plane Vs Control Plane

| Layer | Responsibilities |
| --- | --- |
| Data plane | indexing, CRUD, dense or late-interaction search, BM25 fusion, optional graph augmentation, solver packing, persistence, readiness, and metrics |
| Control plane | deployment, lifecycle management, scheduling, promotion gates, tenancy and policy, observability, rollback, and fleet operations |

The OSS package ships the data plane. The control plane is the adjacent
enterprise layer that turns the engine into an operable platform.

## What The Data Plane Already Provides

The runtime is already opinionated and production-facing:

- single-host shard, dense, late-interaction, and multimodal collections
- CRUD, WAL, checkpoint, and restart-safe recovery
- BM25 plus dense fusion
- optional Tabu or knapsack-style solver refinement
- optional Latence graph sidecar with additive rescue and provenance
- health, readiness, and metrics endpoints

That means the hard retrieval path is already present. The control plane should
not reimplement search; it should manage and validate it.

## Control Plane Responsibilities

The enterprise control plane should own the following concerns:

### 1. Deployment and placement

- model deployment and version rollout
- index build or restore orchestration
- CPU or GPU placement decisions
- promotion between staging and production lanes

### 2. Collection lifecycle

- create, update, snapshot, restore, reindex, and rollback collections
- schedule rebuilds or compaction windows
- coordinate schema or payload migrations

### 3. Ingestion scheduling

- queue document-processing and embedding jobs
- coordinate graph delta ingestion and freshness SLAs
- track failures, retries, and partial replays

### 4. Evaluation and regression gates

- run benchmark suites before promoting a new model, shard layout, or graph sync
- enforce route-conformance checks for the optional graph lane
- compare latency, recall, and cost regressions across versions

### 5. Observability

- shard hot spots, latency spikes, stale corpora, and sync failures
- per-collection health, graph freshness, and solver availability
- operator-facing event streams and audit history

### 6. Tenancy, auth, and policy

- tenant isolation and quotas
- policy-driven `graph_mode` defaults
- access control for premium graph workflows and admin actions

## Recommended Service Shape

For an enterprise deployment, a pragmatic control-plane split is:

- `deployment-service`: model and index rollout plus placement decisions
- `ingestion-service`: schedules document, embedding, and graph update jobs
- `evaluation-service`: benchmark, route-conformance, and promotion gates
- `operations-api`: snapshots, restores, rollbacks, and fleet status
- `tenant-policy-service`: auth, quotas, workflow policy, and premium entitlements

The control plane can remain thin if it treats `colsearch` as the retrieval
data plane rather than as a library that each service reimplements differently.

## Graph-Specific Control Plane Needs

The optional Latence graph lane adds a few control-plane requirements that the
base OSS runtime does not solve by itself:

- Dataset Intelligence job monitoring
- graph freshness SLAs and drift alerts
- premium entitlement and tenant policy enforcement
- graph-route release gates before production promotion
- provenance or compliance audit exports for graph-native workflows

## Promotion Flow

A recommended promotion path is:

1. build or update the collection and graph sidecar in staging
2. run graph-off baseline checks plus graph-on conformance checks
3. compare latency, candidate growth, and graph-shaped uplift
4. promote only if retrieval and operability gates pass
5. keep rollback artifacts for both the collection and graph state

This keeps the optional graph lane valuable without letting it silently erode
the stable OSS retrieval path.

## What This Repo Deliberately Does Not Claim

The current OSS repo is not claiming to ship:

- a multi-cluster scheduler
- a tenant-management UI
- hosted embedding or graph SaaS
- automatic fleet-wide promotion orchestration

Those are control-plane concerns. The repo instead exposes the runtime seams and
health signals a real control plane would need.
