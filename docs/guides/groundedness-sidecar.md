# Groundedness Sidecar

`voyager-index` keeps the OSS retrieval engine focused on getting the right
chunks back fast. Post-generation hallucination scoring is shipped as a
separate, commercial sidecar called
[`latence-trace`](https://github.com/ddickmann/latence-trace) so the two
products can scale, be licensed, and evolve independently.

## Product Split

| Layer | What lives here |
| --- | --- |
| OSS retrieval core (this repo) | multimodal preprocessing, embeddings/model-serving seams, late-interaction index, BM25S, quantization, fusion, and optional solver packing |
| Optional Latence graph plane | `LatenceGraphSidecar`, graph-aware candidate rescue, graph-aware solver features, provenance, and Dataset Intelligence sync metadata |
| Optional `latence-trace` groundedness sidecar | post-generation calibrated reverse MaxSim, literal guardrails, NLI / claim-level verifier, semantic-entropy peer, structured-source verification, and a calibrated risk band |

The OSS engine stays fully usable without the groundedness sidecar
installed - it is additive and post-generation, sitting on the answer
path, not the retrieval path.

## Why It Lives In A Separate Repo

- **Different latency budget.** Retrieval is sub-10 ms p95 in voyager-index. The groundedness lane runs an NLI verifier and an optional semantic-entropy peer that legitimately consume 100-150 ms p95 even on an A5000.
- **Different licensing posture.** Retrieval should remain commodity OSS. Calibrated hallucination scoring is the kind of feature customers expect a commercial guarantee on (calibration, peer fusion weights, drift monitoring), so it is shipped under a proprietary license.
- **Different release cadence.** The sidecar evolves with NLI model updates, calibration sweeps, and dataset-specific risk bands without forcing voyager-index releases.
- **Failure isolation.** A degraded NLI lane never breaks first-stage retrieval, and a retrieval restart never invalidates calibrated thresholds.

## What The Sidecar Does

`latence-trace` exposes a single endpoint:

```text
POST /groundedness
```

It accepts the same dual-mode contract you would otherwise have called
on voyager-index:

- **`chunk_ids[]` fast path** - your generation layer passes the exact
  chunk IDs that were stitched into the LLM context. The sidecar
  retrieves the corresponding multi-vector embeddings (via a
  caller-supplied resolver, so it can read directly from voyager-index,
  Qdrant, an in-memory cache, or any other vector store) and scores the
  response against them.
- **`raw_context` fallback** - when chunk IDs are unavailable, the
  sidecar segments the raw context into 256-token packed sentence
  windows, encodes them with the configured ColBERT-style multi-vector
  encoder, and scores the response against those windows.

The response carries:

- `scores.reverse_context` - raw response-token-against-context MaxSim
- `scores.reverse_context_calibrated` - z-scored against a held-out null bank and squashed into `(0, 1)` for a wide, readable dynamic range
- `scores.literal_guarded` - calibrated headline discounted by unsupported response literals (dates, numbers, units, currency, URLs, emails, identifiers)
- `scores.nli_aggregate` and `nli_diagnostics.claims` - per-claim entailment scores from the NLI verifier (default `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, with optional cross-encoder premise reranking and atomic-fact decomposition)
- `scores.semantic_entropy_*` - bidirectional-entailment clustering of multiple LLM samples for consistency
- `scores.structured_source_*` - triple-level matching on JSON / Markdown-table sources
- `scores.groundedness_v2` - convex fusion of the channels above with calibrated weights
- `scores.risk_band` - calibrated `green` / `amber` / `red` classification by stratum
- per-token `response_tokens[]`, per-unit `support_units[]`, and `top_evidence[]` for heatmap and evidence-trace UIs

## Integration Shape

A typical voyager-index plus latence-trace deployment runs them as two
processes:

```text
                  +-----------------------+
client query  --> | voyager-index server  | -> chunk_ids[], context
                  +-----------------------+
                                |
                                v
                  +-----------------------+
LLM response  --> | latence-trace sidecar | -> groundedness scores,
                  +-----------------------+    risk band, evidence
                                ^
                                |
                  voyager-index acts as the chunk-vector resolver
```

The sidecar is configured with a `ChunkResolver` callback that turns
`chunk_ids` into multi-vector embeddings. In production this typically
calls voyager-index's reference HTTP API (`GET /collections/{name}/points/{id}`)
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

## Where To Go Next

- **Sidecar README, Beta overview, and full algorithm audit:**
  the private [`latence-trace`](https://github.com/ddickmann/latence-trace)
  repository.
- **Product page:** the commercial groundedness tracker is part of the
  [latence.ai](https://latence.ai) product surface.
- **Retrieval boundary:** `voyager-index` itself remains the right place
  to look for indexing, hybrid search, quantization, and the optional
  Latence graph lane.
