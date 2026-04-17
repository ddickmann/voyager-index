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
