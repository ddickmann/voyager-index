# Groundedness Beta

`voyager-index` exposes a **Beta** groundedness / hallucination detection
endpoint for post-generation answers:

```text
POST /collections/{name}/groundedness
```

Use it when you already have a final answer and want to measure how well that
answer is supported by the context that was actually provided to the model.

This feature is intentionally scoped:

- useful for groundedness scoring, evidence tracing, and response-token heatmaps
- useful for QA and user-facing support views
- **not** a final factuality oracle

Dense similarity can still be too forgiving on negation, entity swaps, dates,
numbers, units, and other semantically close factual errors. Treat the result as
a support signal, not a truth guarantee.

Very long mixed-support contexts are another real Beta boundary. In the
repo-level long-context stress run near the 8k-token range, short anchor-style
separation did not carry over cleanly. Keep wording conservative when the answer
draws from long, noisy context blocks and the support boundary is ambiguous even
to a human reader.

## Production Scoring Policy

The production headline score is **naive reverse-context MaxSim** over the final
answer and the supplied support context.

Optional query-conditioned signals can also be returned, but they are diagnostic
only. They help explain prompt echo and coverage behavior; they are not the
recommended product score.

## Two Input Modes

### 1. `chunk_ids[]` fast path

Use this when your generation layer already knows which chunks were shown to the
LLM.

- the endpoint fetches stored support vectors by external chunk ID
- it does **not** re-encode the support context
- this is the preferred production path

### 2. `raw_context` fallback

Use this when chunk IDs are unavailable.

- the endpoint splits `raw_context` into sentence or paragraph support units
- it encodes those support units on demand
- it returns the same groundedness response schema as `chunk_ids[]`

## Encoder Requirement

For text collections, the service needs an encoder available at runtime so it
can encode the response and optional query:

```bash
VOYAGER_GROUNDEDNESS_MODEL=lightonai/GTE-ModernColBERT-v1 voyager-index-server
```

`VOYAGER_ENCODE_MODEL` can also act as the fallback encoder source.

## Example: `chunk_ids[]`

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-li/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_ids": ["doc-1", "doc-7"],
    "query_text": "When was Teardrops released in the United States?",
    "response_text": "Teardrops was released in the United States on 20 July 1981.",
    "evidence_limit": 5
  }'
```

## Example: `raw_context`

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-li/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "raw_context": "Teardrops is a single by George Harrison, released on 20 July 1981 in the United States. It was the second single from Somewhere in England.",
    "query_text": "When was Teardrops released in the United States?",
    "response_text": "Teardrops was released in the United States on 20 July 1981.",
    "segmentation_mode": "sentence"
  }'
```

## What The Response Returns

The response is designed to be heatmap-ready without a second round-trip:

- `scores`: scalar groundedness values
- `response_tokens`: per-token support scores for the generated answer
- `support_units`: the support chunks or segmented raw-context units, each with
  token scores
- `top_evidence`: top response-token to support-token alignments
- `eligibility`: storage/dequantization metadata for user-facing trust
- `debug`: optional dense matrices when explicitly requested

## Eligibility And Dequantization

The endpoint is only trustworthy when support vectors can be scored in float
precision:

- `late_interaction`: supported with stored float vectors
- `multimodal`: supported when stored vectors can be materialized back to float
- `shard` with `fp16`: supported
- `shard` with `int8`: supported after dequantization/materialization
- `shard` with `roq4`: only supported for user-facing groundedness when an FP16
  sidecar embedding is available
- `fp8`: treated as a scoring-mode detail, not as a persisted storage fetch path

## Beta Boundaries

The current Beta is a good fit for:

- final-answer debugging
- operator QA
- user-facing evidence traces with careful product wording

It is not yet the right contract for:

- a hard truth badge
- automated policy action without human review
- claim verification where exact lexical fidelity is required
- very long mixed-support context blocks near the model limit

If you need stronger protection on high-risk tokens, combine the groundedness
score with simple lexical checks for entities, dates, numbers, and units.
