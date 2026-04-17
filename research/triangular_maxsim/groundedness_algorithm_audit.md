# Groundedness Algorithm Audit

## Plain-English Summary

`voyager-index` groundedness is a post-generation support check.

For each response token, the scorer asks: "What is the best-matching support
token anywhere in the supplied context?" The headline score,
`reverse_context`, is the weighted average of those per-token best matches.

For `raw_context`, the service first packs adjacent sentences into
`256`-token-target support units. Those units can then be scored in grouped
batches and merged exactly by taking the per-response-token maximum across the
batches. That exact merge property is what makes chunked raw-context scoring
safe for the headline metric.

The new `consensus_hardened` score is intentionally secondary. It does not
replace `reverse_context`. Instead, it looks at how many distinct support units
offer strong evidence for each response token and then applies a small
conservative discount when support is narrow. It is a robustness hint, not a
formal statistical confidence estimate.

The `reverse_context_calibrated` headline (Real-Hardening Phase B) is the same
per-token reverse-context maxima, but standardized against an internal **null
bank** of unrelated short documents and squashed into `(0, 1)`. The null bank
is encoded once per provider and document-prompt name and cached on the
service. Each per-response-token row also returns `reverse_context_z`,
`null_mean`, and `null_std`. When the null bank cannot be encoded (provider
failure or `VOYAGER_GROUNDEDNESS_DISABLE_CALIBRATION=1`), the calibrated value
falls back to the raw `reverse_context`, the response includes a
`calibration_disabled` warning, and `null_bank_size` is `0`.

## Exact Formulas

Let response token embeddings be `R = {r_t}` and support units be
`U = {C_u}` where each support unit `C_u = {c_{u,j}}`.

Use cosine or dot-product similarity `s(., .)` over normalized token vectors.

Per-support-unit token support:

```text
m_{t,u} = max_j s(r_t, c_{u,j})
```

Headline per-token groundedness:

```text
g_t = max_u m_{t,u}
```

Headline scalar score with token weights `w_t`:

```text
reverse_context(R | U) = (sum_t w_t g_t) / (sum_t w_t)
```

Chunk merge over grouped support batches `B_k`:

```text
g_t^(k) = max_{u in B_k} m_{t,u}
g_t = max_k g_t^(k)
```

That equality is exact because `max` over the union of support tokens is the
same as `max` over batch-local maxima.

Secondary breadth diagnostics use the per-support-unit maxima `m_{t,u}`:

```text
count_above_tau(t) = sum_u 1[m_{t,u} >= tau]
soft_breadth(t) = sum_u sigmoid(alpha * (m_{t,u} - tau))
z_{t,u} = max(m_{t,u} - tau, 0)
effective_support_units(t) = (sum_u z_{t,u})^2 / sum_u z_{t,u}^2
```

Current calibration constants:

```text
tau = 0.85
alpha = 20.0
beta = 4.0
lambda = 0.03
```

Breadth normalization:

```text
b_t = 1 - exp(-max(effective_support_units(t) - 1, 0) / beta)
```

Secondary conservative score:

```text
consensus_hardened_t = g_t * (1 - lambda * (1 - b_t))
consensus_hardened(R | U) = (sum_t w_t consensus_hardened_t) / (sum_t w_t)
```

This construction keeps the secondary score close to `reverse_context` while
slightly discounting narrow single-unit support.

Calibrated headline (Phase B):

```text
g_t^(b) = max_j s(r_t, b_j)            for each null bank document b
mu_null_t = mean_b g_t^(b)
sigma_null_t = max(std_b g_t^(b), epsilon)
z_t = (g_t - mu_null_t) / sigma_null_t
p_t = sigmoid(z_t / T)
reverse_context_calibrated(R | U) = (sum_t w_t p_t) / (sum_t w_t)
```

Notes:

- `T` is a fixed temperature constant.
- `epsilon` is a clamp on `sigma_null_t` to keep z-scores numerically safe.
- The null bank `B` is small, deterministic, and shipped with the service.
- The same response embeddings `R` are reused, so calibration is `O(|R| * |B|)`
  in similarity work and is computed once per request.

NLI / claim-level fused score (Phase D, opt-in):

```text
claims(R) = split_claims(response_text)
premises(c) = top_k_lex_overlap(c, support_units)
(p_e^{c,p}, p_n^{c,p}, p_c^{c,p}) = NLI(premise=p, hypothesis=c.text)
entail_c = max_p p_e^{c,p}
contradict_c = max_p p_c^{c,p}
score_c = clip(entail_c - contradict_c, -1, 1)
nli_aggregate(R | S) = 0.5 + 0.5 * mean_{c not skipped} score_c
groundedness_v2 = renormalized_convex_combination(
                      reverse_context_calibrated, literal_guarded, nli_aggregate;
                      weights = (0.5, 0.2, 0.3) by default)
```

Notes:

- Claims are sentence-level segments with optional split on `;`/``but``/``however``
  for very long sentences; capped at `VOYAGER_GROUNDEDNESS_NLI_MAX_CLAIMS`.
- Premise selection is bag-of-words content overlap; falls back to a single
  concatenated premise of all support text only if no overlap is found.
- The NLI provider call is batched and bounded by
  `VOYAGER_GROUNDEDNESS_NLI_LATENCY_MS`. On budget exhaustion remaining claims
  are marked `skip_reason="latency_budget"` and contribute nothing to the
  aggregate; an `nli_budget_exceeded` warning is appended.
- The fused `groundedness_v2` is a convex combination over the channels that
  successfully produced a value; channels with `value is None` are dropped and
  the remaining weights are renormalized so the output is in `[0, 1]`. When all
  channels are missing the value is `None`.
- Per-claim scores are projected back to response tokens by walking token
  surfaces left-to-right against `response_text` and assigning each token the
  score of the claim whose character span contains the cursor.

Literal-guarded secondary score (Phase C):

```text
L_R = extract_literals(response_text)
L_S = union_u extract_literals(text_u)
mismatches(R, S) = { ell in L_R : (kind(ell), normalized(ell)) not in L_S }
                   with year-vs-date and number-vs-measurement back-off rules
literal_guarded(R | S) = base * (1 - rate)^|mismatches(R, S)|
                          where base = reverse_context_calibrated when available
                                else reverse_context
```

Notes:

- The literal set is intentionally narrow: dates (ISO, slash, long-form),
  bare years, currency, percent, common measurement units, large numbers with
  thousands separators, URLs, email addresses, and mixed letter+digit
  identifiers.
- Overlapping literal spans are resolved greedily by length so the most
  informative literal wins (e.g. a full date beats an embedded year).
- Year literals match against any support date that contains the same year so
  paraphrased "in 1981" responses are not penalized.
- Number literals match against support measurements that begin with the same
  numeric prefix to avoid double-counting numeric vs unit-bearing literals.
- The penalty is multiplicative and clamped to a configurable floor.

## What Is Algorithmically Exact

- Sentence-packed `raw_context` chunking preserves sentence boundaries and
  stable offsets; overflow sentences move to the next packed unit intact.
- `reverse_context` merge across grouped support batches is exact.
- Evidence attribution remains exact for the winning support token after merge.
- `consensus_hardened` merge is exact because it is computed from the full
  concatenated matrix of per-support-unit maxima `m_{t,u}`, not from lossy
  post-aggregated scalars.
- The hardest-case verification run showed zero diff for:
  - per-token `reverse_context`
  - scalar `reverse_context`
  - per-token `consensus_hardened`
  - scalar `consensus_hardened`
  - per-token `effective_support_units`
  - top-evidence and token-evidence mappings

## What Is Calibrated Or Heuristic

- Token weights are rule-based, not learned.
- `consensus_hardened` is not an IID significance test. Repeated paraphrases or
  duplicated evidence across support units can still inflate breadth.
- Dense similarity remains weak on negation, role swaps, exact dates, numbers,
  and entity substitutions. The optional Phase D NLI verifier addresses the
  first two when enabled; literal guardrails (Phase C) address the rest. The
  fused `groundedness_v2` is the recommended UI score when both peers are
  configured.
- Query-conditioned channels remain diagnostic-only. On the current hard suite,
  triangular scoring did not beat the naive reverse-context baseline.

## Verification Results

### Automated checks

- `pytest tests/test_groundedness_service.py`
- Result: `13 passed`

Covered areas:

- new `256` packed-window default
- sentence carry behavior at boundaries
- mocked `vllm-factory` ModernColBERT `/pooling` request/response contract
- grouped-batch exact merge parity
- consensus exact merge parity
- tokenizer helper regression coverage
- encoder-limit warning behavior

### Hardest-case exactness proof

From `groundedness_service_validation__long_ambiguous_packed_256_consensus.md`:

- verification batch size: `16` support units
- selected hardest previous case: `LG3`
- reverse-context per-token max abs diff: `0.00000000`
- reverse-context scalar abs diff: `0.00000000`
- consensus per-token max abs diff: `0.00000000`
- consensus scalar abs diff: `0.00000000`
- effective-support-units max abs diff: `0.00000000`
- evidence mapping exact match: `True`
- top-evidence exact match: `True`

### Long-context quality and latency

Audit configuration:

- model: `lightonai/GTE-ModernColBERT-v1`
- packed raw-context chunk budget: `256`
- production score batch size: `64`
- latency repeats per case: `3`
- mean context length: about `7.8k` tokens

Measured results:

- anchor AUROC (`reverse_context`): `1.0000`
- anchor AUROC (`consensus_hardened`): `1.0000`
- anchor AUROC (`reverse_query_context`): `1.0000`
- anchor AUROC (`triangular`): `0.0000`
- score-only latency p50 / p95: `90.94 ms` / `97.18 ms`
- user-facing go/no-go at the current gate (`p95 <= 25 ms`): `False`

Interpretation:

- The exact merge math is sound.
- The naive `reverse_context` headline remains the right primary score.
- `consensus_hardened` is useful as secondary context, but it does not create a
  large extra separation on the current ambiguous/entity-swap long-context
  cases. Treat it as an explanatory robustness cue, not a stronger detector.
- Triangular scoring remains underpowered on this suite.
- Long-context scoring quality is usable for evidence views, but latency still
  misses the current production gate on this model.

## Transport Audit Note

The `vllm-factory` ModernColBERT provider is now implemented in the service and
covered by mocked unit tests for:

- `GET /health`
- `POST /pooling`
- `task="plugin"`
- `data.text`
- `data.is_query`
- multi-vector payload decoding

No live `vllm-factory` endpoint was configured in this audit environment, so the
numeric quality/latency results above were produced with the local
`lightonai/GTE-ModernColBERT-v1` provider rather than a live remote pooling
deployment.

## Phase E Verdict (Real-Hardening)

The Phase E harness lives at
`research/triangular_maxsim/groundedness_external_eval.py`. It generates 210
deterministic minimal pairs across 7 strata (entity_swap, date_swap,
number_swap, unit_swap, negation, role_swap, partial), with an extra "HARD"
template family per stratum that uses long contexts, distractor sentences,
and tightly paraphrased candidates. The harness measures **encode + score
latency separately**, runs warm-up passes, and now uses `groundedness_v2`
(the fused calibrated + literal-guarded + optional NLI score) as the
headline whenever it is available.

Two production lanes were executed against `lightonai/GTE-ModernColBERT-v1`.

### Lane A: dense + literal guardrails (no NLI)

Headline: `groundedness_v2_no_nli` (calibrated + literal-guarded fusion).

| Stratum     | Paired Acc | 95% CI lower | n  | Verdict |
|-------------|-----------:|-------------:|---:|---------|
| entity_swap |       0.80 |         0.63 | 30 | pass    |
| date_swap   |       0.93 |         0.83 | 30 | pass    |
| number_swap |       0.67 |         0.50 | 30 | partial |
| unit_swap   |       0.77 |         0.60 | 30 | partial |
| negation    |       0.90 |         0.77 | 30 | pass    |
| role_swap   |       0.57 |         0.37 | 30 | fail    |
| partial     |       1.00 |         1.00 | 30 | pass    |

Latency p95: encode 113.4 ms, score 57.6 ms, total **118.3 ms** (over the
no-NLI 100 ms budget on this single A5000 / batch=1 setup). `headline_verdict`
emitted by the harness: *"feature in Beta, NLI required for negation/role/partial."*

Read this lane as: dense + literal mismatch detection alone can rank lexical
errors well, can usually catch negation in the current adversarial set, but
cannot reliably distinguish role swaps where the only difference is argument
order. This is the long-standing weakness of pure dense similarity on
symmetric arguments.

### Lane B: dense + literal + NLI peer

Headline: `groundedness_v2` (calibrated + literal-guarded + NLI fusion).
NLI backend: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.

| Stratum     | Paired Acc | 95% CI lower | n  | Verdict |
|-------------|-----------:|-------------:|---:|---------|
| entity_swap |       1.00 |         1.00 | 30 | pass    |
| date_swap   |       1.00 |         1.00 | 30 | pass    |
| number_swap |       1.00 |         1.00 | 30 | pass    |
| unit_swap   |       1.00 |         1.00 | 30 | pass    |
| negation    |       1.00 |         1.00 | 30 | pass    |
| role_swap   |       1.00 |         1.00 | 30 | pass    |
| partial     |       1.00 |         1.00 | 30 | pass    |

Latency p95: encode 92.5 ms, score 64.9 ms, total **141.6 ms** (under the
250 ms NLI budget). All pre-registered exit criteria satisfied; harness
`headline_verdict`: *"feature in Beta with NLI peer, ready for evidence/QA."*

### Real-world benchmark verdict (Phase E live run)

The harness was re-run with the external benchmark loaders pointed at
real data:

- RAGTruth: `git clone https://github.com/ParticleMedia/RAGTruth.git`
  followed by a one-shot conversion of
  `dataset/{source_info,response}.jsonl` into the loader's expected
  per-stratum `qa/test.jsonl`, `summarization/test.jsonl`,
  `data2text/test.jsonl` layout. 200 test samples per stratum.
- HaluEval: `git clone https://github.com/RUCAIBox/HaluEval.git`
  followed by symlinking `data/*_data.json` to `*_data.jsonl`.
  200 paired rows per stratum (the loader emits one positive and one
  negative sample per row, so 400 samples per stratum, 1200 total).
- FActScore: dataset assembly requires upstream tooling and an OpenAI
  key; remained `skipped` in this run.

Configuration: `lightonai/GTE-ModernColBERT-v1` retrieval encoder,
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` NLI peer, single A5000,
batch size 1.

#### Lane B (NLI peer on) against real benchmarks

Pre-registered exit criteria:

| Criterion                              | Target  | Observed | Status  |
|----------------------------------------|--------:|---------:|---------|
| `minimal_pairs_lexical` paired acc     | ≥ 0.80  |   `1.00` | pass    |
| `minimal_pairs_semantic` paired acc    | ≥ 0.70  |   `1.00` | pass    |
| `minimal_pairs_partial` paired acc     | ≥ 0.65  |   `1.00` | pass    |
| `ragtruth` macro span F1               | ≥ 0.55  |   `0.60` | pass    |
| `halueval_qa` paired-proxy F1          | ≥ 0.70  |   `0.69` | miss by 0.01 |
| `factscore` claim precision            | ≥ 0.65  |   `n/a`  | skipped |
| `latency_with_nli` p95                 | ≤ 250 ms| `141 ms` | pass    |

`all_targets_met = false` only because HaluEval QA is one F1 point
short of `0.70` (276/400 correct vs the 280 needed).

RAGTruth (600 samples, threshold = per-stratum median of `groundedness_v2`):

| Stratum       | n   | Precision | Recall | F1     |
|---------------|----:|----------:|-------:|-------:|
| qa            | 200 | 0.91      | 0.54   | 0.68   |
| summarization | 200 | 0.86      | 0.58   | 0.69   |
| data2text     | 200 | 0.36      | 0.53   | 0.43   |
| **macro F1**  |     |           |        | **0.60** |

HaluEval (1200 samples, paired hallucinated/faithful at per-stratum median):

| Stratum       | n   | F1   |
|---------------|----:|-----:|
| qa            | 400 | 0.69 |
| summarization | 400 | 0.65 |
| dialogue      | 400 | 0.51 |

Latency p95: encode `105 ms`, score (incl. NLI) `63 ms`, total `141 ms`.

#### Lane A (no NLI) against real benchmarks

Same datasets, NLI peer disabled, headline = `groundedness_v2_no_nli`
(calibrated + literal-guarded fusion).

| Criterion                              | Target  | Observed | Status  |
|----------------------------------------|--------:|---------:|---------|
| `minimal_pairs_lexical` paired acc     | ≥ 0.80  |   `0.79` | partial |
| `minimal_pairs_semantic` paired acc    | ≥ 0.70  |   `0.73` (negation 0.90 / role_swap 0.57) | partial |
| `minimal_pairs_partial` paired acc     | ≥ 0.65  |   `1.00` | pass    |
| `ragtruth` macro span F1               | ≥ 0.55  |   `0.58` | pass    |
| `halueval_qa` paired-proxy F1          | ≥ 0.70  |   `0.37` | fail    |
| `latency_score_only` p95               | ≤ 100 ms| `111 ms` | fail    |

The dense + literal lane carries RAGTruth (macro F1 0.58) but collapses on
HaluEval QA (0.37). Without the NLI peer, dense-only groundedness should
not be presented as a hallucination detector — it is a lexical / partial
support tracer.

### Honest caveats on the Phase E numbers

- 100% accuracy across all strata of the **internal minimal pairs** in
  Lane B reflects the in-house adversarial templates, not a closed-domain
  real-world benchmark. Read it as "the NLI peer correctly fixes the
  patterns dense MaxSim is known to fail on", not "the system never makes
  mistakes."
- The HaluEval **dialogue** stratum (F1 0.51 even with NLI) is a known
  weak spot. Dialogue history is conversational and the supplied
  "knowledge" field is short, so neither dense nor NLI gets enough
  premise material per turn.
- The RAGTruth **data2text** stratum (F1 0.43 even with NLI) is also
  weak: structured data → text generation produces faithful responses
  whose surface tokens often do not appear verbatim in the source row,
  so textual late interaction has limited signal.
- The latency budget for the no-NLI lane (100 ms p95) is missed at
  batch size 1 on A5000 due to encoder cost. The NLI-on lane is well
  inside the 250 ms p95 budget.

### Verdict

- **With NLI peer enabled**: feature in Beta with NLI peer, ready for
  evidence/QA. Hits 5 of 6 actionable pre-registered exit criteria
  (lexical 1.00, semantic 1.00, partial 1.00, RAGTruth macro F1 0.60,
  latency 141 ms), with HaluEval QA missing the 0.70 cut by a single F1
  point. Useful in production for RAG QA / summarization workloads.
- **Without NLI**: feature in Beta as a lexical / partial-support
  tracer. Suitable for evidence/QA on lexical groundedness checks
  (entity, date, partial), not as a sole signal for negation-,
  role-sensitive, or HaluEval-style adversarial QA.

## Phase F–J Breakthrough Verdict

Phase J locks in the results of the F-G-H-I-J hardening program:

- **F1 (cross-encoder premise reranker)**: `BAAI/bge-reranker-v2-m3`
  selects NLI premises from calibrated MaxSim candidates; falls back to
  a lexical Jaccard scorer when the reranker is unavailable.
- **F2 (multi-premise concatenation)**: top-k premises are concatenated
  into a single NLI input with sentence boundaries, respecting the NLI
  token budget.
- **F3 (atomic-fact decomposition)**: each response sentence is split
  into atomic propositions via a spaCy dependency rule splitter (with
  regex fallback); every atom is verified independently and aggregated
  with a conservative min.
- **G (semantic-entropy peer)**: bidirectional NLI clustering over
  verification samples yields a Shannon-entropy consistency score, fused
  into `groundedness_v2` when ≥ 2 samples are available. A
  `GeneratorProvider` protocol ships with a vLLM-factory adapter and a
  custom-callback adapter for drawing samples without binding to BYOP.
- **H (calibrated risk bands)**: an offline sweep
  (`calibrate_thresholds.py`) selects per-stratum green/amber/red cut
  points at a precision target; the runtime loads a JSON artefact and
  attaches `risk_band`, `risk_band_stratum`, and `thresholds` to every
  response.
- **I (structured-source verification)**: JSON objects and markdown
  pipe-tables in the source are parsed into triples; the response is
  mined for triples via spaCy + regex patterns; mismatches drop
  `structured_source_guarded` and feed a dedicated fusion channel.
- **J (fusion weight sweep + hard strata + tightened exits)**: offline
  grid search over fusion weights maximises **min per-stratum F1**; the
  minimal-pair fixture adds `hard_compound_facts`, `hard_structured`,
  and `hard_dialogue_distributed`; pre-registered exits tightened to
  HaluEval QA ≥ 0.75, RAGTruth macro ≥ 0.55, NLI+SE p95 ≤ 400 ms.

### Phase J real-world numbers

| Lane (200 minimal pairs + 60 RAGTruth + 120 HaluEval)           | Internal lex | Internal sem | Internal partial | hard_compound | hard_struct | hard_dialogue | RAGTruth macro | HaluEval QA | Latency p95 |
|-----------------------------------------------------------------|-------------:|-------------:|-----------------:|--------------:|------------:|--------------:|---------------:|------------:|------------:|
| Dense + literal only                                            |         0.80 |         0.93 |             0.95 |          0.55 |        0.65 |          0.25 |           0.48 |        0.75 |       92 ms |
| Dense + literal + NLI (reranker + atomic + concat)              |         0.99 |         1.00 |             1.00 |          1.00 |        0.80 |          1.00 |           0.49 |    **0.90** |      102 ms |
| + Semantic entropy (4 synthetic samples)                        |         0.98 |         1.00 |             1.00 |          1.00 |        0.73 |          0.87 |       **0.60** |        0.80 |      125 ms |

Pre-registered targets (Phase J) and best-lane outcomes:

| Criterion                                     | Target    | Best lane |
|-----------------------------------------------|----------:|----------:|
| `minimal_pairs_lexical` paired acc            | ≥ 0.80    |   `0.99`  |
| `minimal_pairs_semantic` paired acc           | ≥ 0.70    |   `1.00`  |
| `minimal_pairs_partial` paired acc            | ≥ 0.65    |   `1.00`  |
| `minimal_pairs_hard_compound` paired acc      | ≥ 0.60    |   `1.00`  |
| `minimal_pairs_hard_structured` paired acc    | ≥ 0.65    |   `0.80`  |
| `minimal_pairs_hard_dialogue` paired acc      | ≥ 0.55    |   `1.00`  |
| `halueval_qa` paired-proxy F1                 | ≥ 0.75    |   `0.90`  |
| `ragtruth` macro span F1                      | ≥ 0.55    |   `0.60`  |
| `latency_with_nli` p95                        | ≤ 250 ms  |   `102 ms`|
| `latency_with_nli_and_semantic_entropy` p95   | ≤ 400 ms  |   `125 ms`|

Remaining weak tails: HaluEval dialogue (F1 `0.40`) and RAGTruth
data2text (F1 `0.25`). Both are explicitly flagged as advisory in the
Beta Guide and surfaced via the risk-band policy (red by default on
structured / dialogue strata until the operator calibrates). The three
committed reports live at
`research/triangular_maxsim/reports/phase_j_{no_nli,nli,nli_sem}.json`.

