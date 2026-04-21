# Groundedness Sidecar

`colsearch` is the OSS retrieval engine. Once retrieval has returned the
right chunks and an LLM has produced an answer, you usually still need an
auditable answer to the question *is this answer grounded in the context we
gave it?* That post-generation lane is provided by the **Latence Trace**
groundedness sidecar — a separate, commercially licensed service from
[latence.ai](https://latence.ai) that you run next to `colsearch`.

The OSS retrieval core stays usable on its own. The groundedness sidecar is
additive: it sits on the answer path, never on the retrieval path.

## Product Split

| Layer | What lives here |
| --- | --- |
| OSS retrieval core (this repo) | multimodal preprocessing, embeddings/model-serving seams, late-interaction index, BM25S, quantization, fusion, and optional solver packing |
| Optional Latence graph plane | `LatenceGraphSidecar`, graph-aware candidate rescue, graph-aware solver features, provenance, and Dataset Intelligence sync metadata |
| Optional Latence Trace groundedness sidecar | post-generation calibrated reverse MaxSim, literal guardrails, NLI / claim-level verifier with cross-encoder premise reranking, atomic-claim decomposition, semantic-entropy peer, structured-source verification, retrieval-coverage observability, response chunking, calibrated risk band, multilingual EN+DE, three Pareto-optimal profiles |

This is the same architectural cut as the graph plane: a premium,
failure-isolated lane that you can adopt incrementally.

## Current Capabilities (commercial sidecar)

Headline numbers (n=120 RAGTruth + internal pairs, n=60 paired
HaluEval, n=30 FActScore biographies = ~750 atomic claims;
A5000 batch=1; English NLI = `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
where applicable, multilingual NLI = `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`):

| Lane | Stratum | n | Metric | Value |
|---|---|--:|:------:|------:|
| Internal min-pairs | lexical (entity / date / number / unit swap) | 120 | paired_acc | 0.95 |
| Internal min-pairs | semantic (negation / role swap) | 60 | paired_acc | 0.98 |
| Internal min-pairs | hard structured (JSON / md table) | 30 | paired_acc | 0.93 |
| Internal min-pairs | German | 26 | paired_acc | 1.00 |
| RAGTruth | qa | 120 | F1@median | 0.73 (precision 0.98) |
| RAGTruth | summarization | 120 | F1@median | 0.65 (precision 0.80) |
| HaluEval QA (English NLI) | paired ranking | 60 | paired_acc | 0.78 |
| HaluEval Summ (English NLI) | paired ranking | 60 | paired_acc | 0.75 |
| FActScore biographies (per-claim atomic, Wikipedia-grounded) | claim_precision / recall / F1 @ best-F1 threshold | 748 | precision / recall / F1 | 0.61 / 0.62 / 0.62 |
| Latency | end-to-end (NLI on, reranker on, atomic on) | — | p95 | 118 ms |

In-scope workloads: **RAG QA + summarization grounding, structured-source
verification, bilingual EN+DE retrieval pipelines, retrieval-efficiency
observability**.

Known boundary: open-domain dialogue continuations that introduce new
real-world facts not in the dialogue context (HaluEval Dialogue
stratum). The sidecar exposes the workload-boundary contract through
its `/agent-help` and `/.well-known/ai-plugin.json` so callers can
self-discover it.

Three Pareto-optimal profiles:

- `fast` — sub-200 ms p95, dense + literal-only, no NLI
- `balanced` (default) — ~190 ms p95, NLI peer on, no reranker
- `quality` — ~195 ms p95, NLI + cross-encoder premise reranker +
  atomic-claim splitter, full stack

Each profile bundles calibrated thresholds and fusion-weight artefacts
out of the box; an environment variable swap selects between them.

## What The Sidecar Does

The groundedness sidecar exposes a single endpoint:

```text
POST /groundedness
```

It accepts the same dual-mode contract you would otherwise have called
on `colsearch`:

- **`chunk_ids[]` fast path** — your generation layer passes the exact
  chunk IDs that were stitched into the LLM context. The sidecar
  retrieves the corresponding multi-vector embeddings (via a
  caller-supplied resolver, so it can read directly from
  `colsearch`, an in-memory cache, or any other vector store) and
  scores the response against them.
- **`raw_context` fallback** — when chunk IDs are unavailable, the
  sidecar segments the raw context into 256-token packed sentence
  windows, encodes them with the configured ColBERT-style multi-vector
  encoder, and scores the response against those windows.

The response carries:

- `scores.reverse_context` — raw response-token-against-context MaxSim
- `scores.reverse_context_calibrated` — z-scored against a held-out null bank and squashed into `(0, 1)` for a wide, readable dynamic range
- `scores.literal_guarded` — calibrated headline discounted by unsupported response literals (dates, numbers, units, currency, URLs, emails, identifiers)
- `scores.nli_aggregate` and `nli_diagnostics.claims` — per-claim entailment scores from the NLI verifier, with optional cross-encoder premise reranking and atomic-fact decomposition
- `scores.semantic_entropy_*` — bidirectional-entailment clustering of multiple LLM samples for consistency
- `scores.structured_source_*` — triple-level matching on JSON / Markdown-table sources
- `scores.groundedness_v2` — convex fusion of the channels above with calibrated weights
- `scores.risk_band` — calibrated `green` / `amber` / `red` classification by stratum
- `scores.context_coverage_ratio` and per-unit `support_units[i].coverage_score` / `support_units[i].used` — retrieval-efficiency observability: which retrieved chunks the response actually used, surfacing dead-weight chunks and over-fetch
- per-token `response_tokens[]`, per-unit `support_units[]`, and `top_evidence[]` for heatmap and evidence-trace UIs

Both `raw_context` and `response_text` are sentence-packed into windows
that fit the encoder's max sequence length, then scored chunk-by-chunk
with a parity-preserving stitch — so neither side is silently truncated
when input exceeds the encoder limit.

## Integration Shape

A typical colsearch plus groundedness sidecar deployment runs them as two
processes:

```text
                  +-----------------------+
client query  --> | colsearch server  | -> chunk_ids[], context
                  +-----------------------+
                                |
                                v
                  +-----------------------+
LLM response  --> | Latence Trace sidecar | -> groundedness scores,
                  +-----------------------+    risk band, evidence
                                ^
                                |
                  colsearch acts as the chunk-vector resolver
```

The sidecar is configured with a `ChunkResolver` callback that turns
`chunk_ids` into multi-vector embeddings. In production this typically
calls colsearch's reference HTTP API (`GET /collections/{name}/points/{id}`)
or an in-process accessor; both keep the sidecar storage-agnostic.

A minimal request, with the sidecar mounted at `:8090`:

```bash
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_ids": ["doc-1", "doc-7"],
    "query_text": "When was Teardrops released in the United States?",
    "response_text": "Teardrops was released in the United States on 20 July 1981.",
    "evidence_limit": 5
  }'
```

The same shape is available for the `raw_context` fallback by replacing
`chunk_ids` with the raw passage text.

## Operational Boundary

The two processes are deliberately decoupled:

- **Latency budget.** Retrieval is sub-10 ms p95 in colsearch. The
  groundedness lane runs an NLI verifier and an optional semantic-entropy
  peer that legitimately consume ~100–150 ms p95 even on a single
  workstation GPU; running them out of process keeps the retrieval hot
  path tight.
- **Release cadence.** The sidecar evolves with NLI model updates,
  calibration sweeps, and dataset-specific risk bands without forcing
  colsearch releases.
- **Failure isolation.** A degraded NLI lane never breaks first-stage
  retrieval, and a retrieval restart never invalidates calibrated
  thresholds.
- **Licensing.** Calibrated hallucination scoring is shipped under a
  commercial license so customers can rely on calibration, peer fusion
  weights, and drift monitoring as part of the product, not as a
  best-effort OSS extra.

## Where To Go Next

- **Product page and access:** [latence.ai](https://latence.ai) hosts the
  Latence Trace product surface, including pricing and onboarding.
- **Retrieval boundary:** `colsearch` itself remains the right place
  to look for indexing, hybrid search, quantization, and the optional
  Latence graph lane.
