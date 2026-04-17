# Groundedness Tracker (Beta)

`voyager-index` exposes a **Beta** groundedness tracker endpoint for
post-generation answers:

```text
POST /collections/{name}/groundedness
```

Use it when you already have a final answer and want to measure how well that
answer is supported by the context that was actually provided to the model.

This feature is intentionally scoped:

- useful for groundedness scoring, evidence tracing, and response-token heatmaps
- useful for QA and user-facing support views
- a solid starting point for groundedness tracking today
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

Starting with the Real-Hardening rollout the response also carries a calibrated
reverse-context score (`reverse_context_calibrated`) and per-token z-scores
(`reverse_context_z`) standardized against an in-process **null bank** of
unrelated short documents. The calibrated value lives in `(0, 1)` and is the
recommended UI-facing aggregate when the encoder produces tightly clustered
raw similarities. Use the raw `reverse_context` for backwards compatibility,
audits, and exact-merge sanity checks.

Optional query-conditioned signals can also be returned, but they are diagnostic
only. They help explain prompt echo and coverage behavior; they are not the
recommended product score.

### Calibration null bank

The service maintains a small, deterministic bank of unrelated short documents
and encodes it once per provider (and document prompt name). For each request
it computes, per response token, a max-similarity distribution against the bank
and reports:

- `null_mean`, `null_std`: per-token null statistics
- `reverse_context_z`: per-token standardized score `(g_t - mu_t) / sigma_t`
- `reverse_context_calibrated`: aggregate sigmoid-of-z, weighted by the same
  content weights as the raw headline

If the null bank cannot be encoded (provider failure or
`VOYAGER_GROUNDEDNESS_DISABLE_CALIBRATION=1`), the calibrated score falls back
to the raw `reverse_context`, the response includes a `calibration_disabled`
warning, and `null_bank_size` is `0`.

### NLI / claim-level verifier (Phase D, opt-in)

Dense similarity also struggles with negation, role swaps, and other
contradictions that share most surface vocabulary with the support. Phase D
adds an optional **claim-level NLI verifier** that runs as a primary peer of
`reverse_context_calibrated`:

- enable with `VOYAGER_GROUNDEDNESS_NLI_ENABLED=1`
- model: defaults to `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; override
  with `VOYAGER_GROUNDEDNESS_NLI_MODEL`
- claim splitter: sentence-level with optional split on `;`/``but``/``however``
  for very long sentences; capped at `VOYAGER_GROUNDEDNESS_NLI_MAX_CLAIMS`
  (default `16`)
- premise selection: top-`k` lexical overlap support units per claim, with
  `k = VOYAGER_GROUNDEDNESS_NLI_TOP_K` (default `3`); falls back to a
  concatenated premise only if no overlap is found
- batched entailment: `VOYAGER_GROUNDEDNESS_NLI_BATCH` (default `16`)
- latency budget: `VOYAGER_GROUNDEDNESS_NLI_LATENCY_MS` (default `2000`); on
  budget exhaustion the verifier emits an `nli_budget_exceeded` warning and
  falls back to embedding-only scoring for the remaining claims

Output fields:

- `scores.nli_aggregate`: per-claim signed margin
  `entailment - contradiction`, averaged across non-skipped claims and
  squashed into `(0, 1)`.
- `scores.nli_claim_count`, `scores.nli_skipped_count`: bookkeeping for the
  number of claims actually verified vs skipped.
- `scores.groundedness_v2`: convex combination of
  `reverse_context_calibrated`, `literal_guarded`, and `nli_aggregate`.
  Weights default to `(0.5, 0.2, 0.3)` and can be overridden with
  `VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED`,
  `VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL`, and
  `VOYAGER_GROUNDEDNESS_FUSION_W_NLI`. When some channels are unavailable the
  weights are renormalized so the fused value stays in `[0, 1]`.
- `nli_diagnostics.claims`: per-claim record with text, character offsets,
  entailment/neutral/contradiction probabilities, signed score, premise count,
  and any `skip_reason` (`no_premises`, `latency_budget`,
  `nli_provider_error`).
- `response_tokens[*].nli_score`: per-token projection of the claim score onto
  the response tokens that fall inside that claim's span. Tokens outside any
  claim span (or when NLI is disabled) report `null`.

When the verifier is disabled (default) every NLI field is `null` and the
fused `groundedness_v2` falls back to a renormalized convex combination of
the remaining peers (`reverse_context_calibrated` and `literal_guarded`).

### Literal guardrails

Dense similarity is forgiving on dates, numbers, units, currency, and explicit
identifiers. The service ships a narrow-scope rule-based literal extractor and
a literal-guarded secondary score:

- `literal_diagnostics.response_literals`: every literal found in the response,
  with `kind` (one of `date`, `year`, `currency`, `percent`, `measurement`,
  `number`, `url`, `email`, `identifier`), `value`, `normalized` form, and
  character offsets.
- `literal_diagnostics.matches` / `literal_diagnostics.mismatches`: the same
  literals split by whether the kind+normalized value also appears anywhere in
  the support text union (years can match against support dates that contain
  the year; bare numbers can match against support measurements that begin
  with the same numeric literal).
- `scores.literal_mismatch_count`: the number of unsupported response literals.
- `scores.literal_guarded`: `reverse_context_calibrated` (or the raw headline
  if calibration is disabled) multiplied by `(1 - rate)^k` where `k` is the
  mismatch count. Use this when you need a literal-aware aggregate that is
  much more conservative than the embedding-only headline.

The extractor is deliberately lexical and conservative. It only fires on
narrow, easy-to-verify patterns; broader factual checks (negation, role swap,
entailment, exact-vs-paraphrased claim contradiction) are reserved for the
NLI/claim verifier in the Real-Hardening Phase D rollout.

## Two Input Modes

### 1. `chunk_ids[]` fast path

Use this when your generation layer already knows which chunks were shown to the
LLM.

- the endpoint fetches stored support vectors by external chunk ID
- it does **not** re-encode the support context
- this is the preferred production path

### 2. `raw_context` fallback

Use this when chunk IDs are unavailable.

- the endpoint defaults to `segmentation_mode="sentence_packed"`
- it packs adjacent sentences into support windows with a default
  `raw_context_chunk_tokens` budget of `256`
- there is no explicit overlap field; overflow sentences move into the next
  support unit intact
- it encodes those support units on demand
- it returns the same groundedness response schema as `chunk_ids[]`

## Encoder Requirement

For text collections, the service needs an encoder available at runtime so it
can encode the response and optional query:

```bash
VOYAGER_GROUNDEDNESS_MODEL=lightonai/GTE-ModernColBERT-v1 voyager-index-server
```

`VOYAGER_ENCODE_MODEL` can also act as the fallback encoder source.

For a remote production setup with `vllm-factory` ModernColBERT:

```bash
VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT=http://127.0.0.1:8000 \
VOYAGER_GROUNDEDNESS_VLLM_MODEL=VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
voyager-index-server
```

Tune the remote path with:

- `VOYAGER_GROUNDEDNESS_SCORE_BATCH_UNITS` (default `64`)
- `VOYAGER_GROUNDEDNESS_VLLM_BATCH_SIZE`
- `VOYAGER_GROUNDEDNESS_VLLM_MAX_CONCURRENCY`
- `VOYAGER_GROUNDEDNESS_VLLM_TIMEOUT`

Keep `raw_context_chunk_tokens` at or below the encoder's real document-length
limit. The API warns when the requested packed window is larger than the active
encoder can reliably process, because overly large packed windows may be
truncated during encoding.

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
    "response_text": "Teardrops was released in the United States on 20 July 1981."
  }'
```

The example above relies on the default packed fallback:

- `segmentation_mode="sentence_packed"`
- `raw_context_chunk_tokens=256`
- no explicit overlap

Override either field only when you want a smaller packed budget or a different
segmentation strategy such as explicit `sentence` or `paragraph`.

## What The Response Returns

The response is designed to be heatmap-ready without a second round-trip:

- `scores`: scalar groundedness values, including headline `reverse_context`,
  the calibrated headline `reverse_context_calibrated`, the literal-guarded
  secondary `literal_guarded`, the consensus-hardened secondary
  `consensus_hardened`, the calibration sample size `null_bank_size`, and the
  literal counters `literal_mismatch_count`, `literal_match_count`, and
  `literal_total_count`
- `response_tokens`: per-token support scores for the generated answer, plus
  calibration diagnostics (`reverse_context_calibrated`, `reverse_context_z`,
  `null_mean`, `null_std`) and breadth diagnostics such as
  `support_unit_hits_above_threshold`, `support_unit_soft_breadth`, and
  `effective_support_units`
- `literal_diagnostics`: per-request narrow-scope literal extraction with
  `response_literals`, `matches`, and `mismatches`
- `support_units`: the support chunks or segmented raw-context units, each with
  token scores
- `top_evidence`: top response-token to support-token alignments
- `eligibility`: storage/dequantization metadata for user-facing trust
- `debug`: optional dense matrices when explicitly requested

`consensus_hardened` is intentionally conservative: it slightly discounts narrow
single-unit support and should be treated as a robustness cue, not as a formal
statistical test.

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
- packed raw-context windows that exceed the active encoder's usable token limit

Current hard-suite audit (`lightonai/GTE-ModernColBERT-v1`, `256`-token packed
windows, mean context about `7.8k` tokens):

- reverse-context AUROC: `1.0`
- consensus-hardened AUROC: `1.0`
- score-only latency: about `90.9 ms` p50 / `97.2 ms` p95
- exact merge proof on the hardest rerun: zero diff for reverse-context,
  consensus-hardened, and effective-support-units diagnostics

That means the algorithmic merge path is exact, but long-context latency still
misses the current `25 ms` production gate on this model.

If you need stronger protection on high-risk tokens, combine the groundedness
score with simple lexical checks for entities, dates, numbers, and units.

### Real-Hardening Phase E & Breakthrough Phase F-J: real-world verdict

The repo ships a deterministic minimal-pair harness at
`research/triangular_maxsim/groundedness_external_eval.py`. It now generates
200 pairs across **10 strata** (entity_swap, date_swap, number_swap, unit_swap,
negation, role_swap, partial, `hard_compound_facts`, `hard_structured`,
`hard_dialogue_distributed`), with long-context "HARD" template families per
stratum that use distractors and tightly paraphrased candidates. The harness
measures encode and score latency separately, runs warm-up passes, and uses
`groundedness_v2` as the headline whenever it is available.

Three production lanes are pre-registered (A5000, batch 1, per-stratum
samples):

- **Dense + literal guardrails (no NLI)**, GTE-ModernColBERT-v1
  - lexical strata (entity, date, number, unit): paired accuracy `0.80`
  - semantic strata (negation, role_swap): `0.48`
    (negation `0.85`, role_swap `0.10`)
  - partial: `0.95`; `hard_compound` `0.55`; `hard_structured` `0.65`;
    `hard_dialogue_distributed` `0.25`
  - latency p95 = encode `1` + score `91` = `92 ms`
  - harness verdict: *"feature in Beta, NLI required for negation/role/partial"*
- **Dense + literal + NLI peer** with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`,
  cross-encoder premise reranker `BAAI/bge-reranker-v2-m3`, atomic-fact
  splitter, multi-premise concatenation, and Phase I structured-source
  verification
  - lexical: `0.99`; semantic: `1.00`; partial: `1.00`;
    `hard_compound` `1.00`; `hard_structured` `0.80`;
    `hard_dialogue_distributed` `1.00`
  - latency p95 = encode `1` + score `102` = `102 ms`
    (well under the `250 ms` NLI budget)
  - harness verdict: *"feature in Beta with NLI peer, ready for evidence/QA"*
- **NLI peer + semantic-entropy consistency** (Phase G) with 4 synthetic
  verification samples per case, fused at the `groundedness_v2` layer
  - lexical: `0.98`; semantic: `1.00`; partial: `1.00`;
    `hard_compound` `1.00`; `hard_structured` `0.73`;
    `hard_dialogue_distributed` `0.87`
  - latency p95 = `125 ms` (under the `400 ms` NLI + semantic-entropy
    budget for 4 samples)
  - harness verdict: *"feature in Beta with NLI peer, ready for evidence/QA"*

The same lanes were re-run with the external benchmark loaders pointed at
real RAGTruth (`ParticleMedia/RAGTruth`) and real HaluEval
(`RUCAIBox/HaluEval`) data:

| Real-world criterion                    | Target    | no-NLI | NLI    | NLI+SE | Status (best lane) |
|-----------------------------------------|----------:|-------:|-------:|-------:|:-------------------|
| HaluEval QA paired-proxy F1             | ≥ 0.75   |  0.75  | **0.90** | 0.80 | pass               |
| HaluEval summarization F1               | —        |  0.40  | 0.70   | 0.50   | advisory           |
| HaluEval dialogue F1                    | ≥ 0.60   |  0.40  | 0.40   | 0.40   | miss (hardest split) |
| RAGTruth QA F1                          | —        |  0.53  | **0.69** | 0.71 | advisory           |
| RAGTruth summarization F1               | —        |  0.71  | 0.56   | **0.83** | advisory       |
| RAGTruth data2text F1                   | ≥ 0.60   |  0.25  | 0.24   | 0.25   | miss (structured-source edge) |
| RAGTruth macro span F1                  | ≥ 0.55   |  0.50  | 0.49   | **0.60** | pass (NLI+SE lane) |
| Latency p95 (NLI on)                    | ≤ 250 ms | —      | **102 ms** | 125 ms | pass          |

Minimal-pair internal strata (NLI lane, 20 pairs/stratum, 95% CI lower
bound in parens): entity_swap `0.95 (0.85)`, date_swap `1.00 (1.00)`,
number_swap `1.00 (1.00)`, unit_swap `1.00 (1.00)`, negation `1.00 (1.00)`,
role_swap `1.00 (1.00)`, partial `1.00 (1.00)`, `hard_compound_facts`
`1.00 (1.00)`, `hard_structured` `0.80 (0.65)`, `hard_dialogue_distributed`
`1.00 (1.00)`.

Without the NLI peer, the same datasets degrade as expected on
semantic-heavy strata: HaluEval summarization drops from `0.70` → `0.40`,
HaluEval QA from `0.90` → `0.75`, and `hard_dialogue_distributed` from
`1.00` → `0.25`. The NLI peer + semantic-entropy consistency is what
makes the real-world numbers production-shaped on RAGTruth.

Honest caveats:

- 100% on internal minimal pairs in the NLI lanes reflects the in-house
  adversarial templates designed to be hard for dense MaxSim. Read it as
  "the NLI peer fixes the patterns dense MaxSim is known to miss", not
  "the system is always right."
- **HaluEval dialogue** (F1 `0.40`) and **RAGTruth data2text** (F1
  `0.25`) are still the weakest splits. Conversational dialogue and
  structured data→text both put response tokens that do not appear
  verbatim in the source, so textual late-interaction has limited
  signal even with Phase I structured-source verification helping.
  Treat groundedness as advisory in those two lanes and rely on the
  calibrated risk band (see
  [risk bands](#risk-bands-and-calibrated-thresholds)).
- FActScore was not configured locally so its lane stays `skipped`.

Practical recommendation: enable the NLI peer
(`VOYAGER_GROUNDEDNESS_NLI_ENABLED=1`) with the cross-encoder reranker
(`VOYAGER_GROUNDEDNESS_NLI_RERANKER_MODEL=BAAI/bge-reranker-v2-m3`) and
atomic-claim decomposition
(`VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS=1`) for any product surface that
has to be safe against negation, role-swap, compound-fact, or close-factual
hallucinations. Layer in semantic-entropy peer
(`VOYAGER_GROUNDEDNESS_SEMANTIC_ENTROPY_ENABLED=1`) when the caller can
supply >= 2 verification samples; this is what delivered the RAGTruth
macro ≥ 0.55 headline.

To reproduce these numbers point the harness at the cloned datasets:

```bash
export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=$PWD/research/triangular_maxsim/external_data/RAGTruth/voyager_layout
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=$PWD/research/triangular_maxsim/external_data/HaluEval/data

# No-NLI lane
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 --max-external-per-stratum 20 \
  --out research/triangular_maxsim/reports/phase_j_no_nli.json

# NLI lane (reranker + atomic + concat-premises)
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 --max-external-per-stratum 20 \
  --enable-nli --reranker-model BAAI/bge-reranker-v2-m3 \
  --concat-premises --atomic-claims \
  --out research/triangular_maxsim/reports/phase_j_nli.json

# NLI + semantic-entropy lane (synthetic 4-sample peer)
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 15 --max-external-per-stratum 10 \
  --enable-nli --reranker-model BAAI/bge-reranker-v2-m3 \
  --concat-premises --atomic-claims \
  --enable-semantic-entropy --semantic-entropy-samples 4 \
  --out research/triangular_maxsim/reports/phase_j_nli_sem.json
```

See `research/triangular_maxsim/README.md` for the dataset prep steps and
committed reports under `research/triangular_maxsim/reports/`.

### Risk bands and calibrated thresholds

Phase H replaces the binary headline with a **calibrated, per-stratum risk
band** (`green` / `amber` / `red`). Thresholds are produced offline by
`research/triangular_maxsim/calibrate_thresholds.py`, which sweeps candidate
values on a precision target (default `0.75`) per stratum and writes a JSON
artefact loaded at runtime. Override the artefact location via
`VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH`. Every response carries the active
`risk_band`, `risk_band_stratum`, and a `thresholds` summary so callers can
audit what decision was taken and why. A missing score is always classified
as `red`.

### Structured-source verification (Phase I)

When the request declares `content_type=application/json` or
`content_type=text/markdown` (or the raw context is auto-detected as a JSON
object or pipe-table), the service runs a deterministic triple extractor
over the source, a rule-based extractor over the response, and matches
triples with string normalization, numeric tolerance, and alias
resolution. Mismatches (numeric flips, wrong cell lookups, missing
predicates) drop `structured_source_guarded` toward `0`, which feeds into
`groundedness_v2` via the `structured` fusion channel
(`VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED`). The full diagnostic payload
(matches, mismatches, detected format) is returned in
`structured_diagnostics`.

### Fusion weights (Phase J)

`groundedness_v2` is a convex combination over `calibrated`, `literal`,
`nli`, `semantic_entropy`, and `structured` channels, renormalized at
runtime over whichever channels are available. Default weights were chosen
by `research/triangular_maxsim/sweep_fusion_weights.py`, which grid-searches
over fusion weights to maximise the **minimum per-stratum F1** (a
"weakest-link" objective) on minimal pairs. All weights are overridable via
`VOYAGER_GROUNDEDNESS_FUSION_W_{CALIBRATED,LITERAL,NLI,SEMANTIC_ENTROPY,STRUCTURED}`.
