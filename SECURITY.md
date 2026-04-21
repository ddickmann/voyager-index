# Security Policy

## Supported Surface

Security fixes are prioritized for the `colsearch` OSS foundation surface:

- public package entrypoints under `colsearch`
- the reference API under `colsearch.server` and `deploy/reference-api/`
- local storage, persistence, and collection metadata handling
- exposed kernel wrappers and documented CPU/GPU fallback behavior

## Reporting

If you discover a security issue, do not open a public issue with exploit
details.

Use one of these private reporting paths:

- the repository's GitHub Security Advisory / private vulnerability reporting flow
- the maintainer contact method documented in the repository security settings, if advisory reporting is not available

When reporting, include:

- affected file or API surface
- impact and expected severity
- reproduction steps
- whether the issue affects only local deployments or also shipped artifacts

## Response Goals

- acknowledge receipt promptly
- reproduce and triage the issue
- fix and document the affected release surface
- update release notes when the fix ships

## Current Deployment Posture

The reference server now supports single-host multi-worker deployments for QPS
scaling. Collection mutations coordinate through OS-backed locks, collection
metadata revisions, and shared async task state so multiple local workers can
observe durable changes without manual restarts.

The server is still a reference deployment, not a turnkey multi-tenant control
plane. If you expose it on the public internet, you are expected to place it
behind your own authentication, TLS termination, request filtering, and network
policy.

Current guarantees and limits:

- collection names are restricted to a single validated path segment
- collection metadata writes are atomic and collection mutations use snapshot rollback
- single-host multi-worker mutation visibility is supported for local worker processes sharing the same storage root
- legacy unsafe local index formats such as pickle-backed HNSW fallback state and monolithic ColPali `.pt` snapshots are rejected
- `/health` is a liveness check and `/ready` reports degraded startup state such as sparse-index load failures
- built-in middleware provides request IDs, rate limiting, and concurrency limiting, but you should still run a reverse proxy / ingress layer for internet-facing deployments
- distributed multi-host coordination is out of scope for the reference server and requires an external control plane if you need it
