# rroq158 outlier-rescue: design for breaking the dense-ternary floor

> **Status**: design proposal, not implemented. Companion to the
> Pareto probe in `scripts/rroq158_pareto_probe.py` and the audit in
> `reports/audit_rroq158_cpu_2026q2.md`.

## 1. Why uniform compression hits a wall on ModernColBERT

Endpoint sanity-check on arguana@n_eval=300, CPU-8w
(`reports/rroq158_pareto_endpoints/arguana/`):

| Config | B/tok | bits/coord | NDCG@10 | Recall@100 | p95 ms | QPS |
|---|---:|---:|---:|---:|---:|---:|
| Baseline `K=8192, gs=32` (today) | 46.0 | 2.875 | 0.3713 | 0.9633 | 666 | 13.8 |
| Floor `K=1024, gs=128` (extreme) | 40.0 | 2.500 | 0.3524 | 0.9567 | 520 | 16.4 |
| Δ relative | −13% | −13% | **−5.1%** | −0.7% | **−22%** | **+19%** |

The bytes/tok formula (see
`benchmarks/beir_benchmark.py:_build_rroq158_gpu_payload`):

```
B/tok = 2·(dim/8)              # sign + nz planes (dense ternary)
      + (dim/group_size)·2     # per-group fp16 scales
      + 2                      # int16 centroid id
      + 4                      # cos+sin fp16 norms
```

For `dim=128`:

* Dense ternary planes: **32 B/tok, fixed**, 70% of payload
* Scales (gs=32 → 8 B/tok, gs=128 → 2 B/tok): only knob that shrinks
* `cid` and norms: fixed at 6 B/tok
* `K` shrinks the *codebook* (in-RAM table) but does **not** change the on-disk
  per-token cid (it's int16 nominal regardless of K)

So the achievable uniform compression on dim=128 is bounded by the dense ternary
plane: **40 B/tok ≈ 2.5 bits/coord** — and you pay −5% NDCG for that −13%
storage win.

To break through 2.5 b/c we have to either (a) drop one of the planes, or
(b) keep the planes but pay them only on a small fraction of tokens.

## 2. Two-regime hybrid: rich + sign-only

Same idea as the user's KV-cache outlier-rescue prototype
(p=0.30 → 1.85 b/d with measurable quality lift over pure binary).
Each token is encoded under one of two regimes:

| Regime | Sign | NZ | Scales | cid | Norms | B/tok (dim=128) | bits/coord |
|---|---|---|---|---|---|---:|---:|
| `rich` (today) | u8[dim/8] | u8[dim/8] | fp16[dim/gs_r] (gs_r=32) | int16 | fp16×2 | 46 | 2.875 |
| `cheap` (sign-only) | u8[dim/8] | — | fp16[dim/gs_c] (gs_c=128) | int16 | fp16×2 | 16 + 2 + 2 + 4 = **24** | **1.500** |

Per-token `regime: u8` flag (1 B/tok overhead) + a fraction `p ∈ (0,1)` of
tokens encoded as `rich`:

```
B/tok(p) = 1 + p·46 + (1−p)·24 = 25 + 22·p
```

Predicted blended density vs. p:

| p (rescue fraction) | B/tok | bits/coord | quality story |
|---:|---:|---:|---|
| 0.00 | 25.0 | 1.563 | all sign-only (floor) |
| 0.05 | 26.1 | 1.631 | tiny rescue |
| 0.10 | 27.2 | 1.700 | rescue top 10% |
| 0.20 | 29.4 | 1.838 | matches user's KV-cache p=0.30 design point |
| 0.30 | 31.6 | 1.975 | heavy rescue |
| 1.00 | 47.0 | 2.938 | all rich + 1B regime tax |

Even at p=0.30 we hit **1.975 b/coord**, ~31% smaller than today's 2.875 and
right at the user's KV-cache demonstrated sweet spot. At p=0.10 we hit
**1.7 b/coord** (~41% smaller) — assuming the rescue actually defends quality.

## 3. Outlier selector

The natural outlier signal is already in the encoded payload: `sin_norm =
sinc(||r||)·||d||`. Tokens with high `sin_norm` carry a lot of
off-centroid (residual) energy and are exactly the ones the cheap codec
(which loses the nonzero mask) hurts most. Sort tokens by `sin_norm`
descending; the top-`p` fraction is the rescue set.

Why `sin_norm` and not `||r||` directly: `sin_norm` already incorporates
the document-norm scaling that the score formula multiplies through, so
it is a calibrated proxy for the residual term's contribution to the
final similarity.

Alternative selectors to ablate:

* **Top-p by `sin_norm`** (recommended baseline)
* Top-p by per-token reconstruction MSE under the cheap codec
* Top-p by post-FWHT energy outside the centroid direction
* Mixed signal: `α·sin_norm + (1−α)·MSE`

## 4. Encoder API surface

`Rroq158Encoded` (current shape, see `quantization/rroq158.py:302`):

```python
@dataclass
class Rroq158Encoded:
    centroids: np.ndarray          # (K, dim) float32
    centroid_id: np.ndarray        # (n_tok,) uint16
    sign_plane: np.ndarray         # (n_tok, dim/8) uint8
    nonzero_plane: np.ndarray      # (n_tok, dim/8) uint8
    scales: np.ndarray             # (n_tok, n_groups) float16
    cos_norm: np.ndarray           # (n_tok,) float16
    sin_norm: np.ndarray           # (n_tok,) float16
    fwht_seed: int
    dim: int
    group_size: int
```

Proposed evolution (additive, on-disk shape changes per regime):

```python
@dataclass
class Rroq158EncodedHybrid:
    centroids: np.ndarray          # unchanged
    centroid_id: np.ndarray        # unchanged

    # Streams shared by both regimes
    sign_plane: np.ndarray         # (n_tok, dim/8) uint8  -- always present
    cos_norm: np.ndarray           # (n_tok,) float16
    sin_norm: np.ndarray           # (n_tok,) float16

    # Per-token regime flag (0 = cheap, 1 = rich)
    regime: np.ndarray             # (n_tok,) uint8

    # Two scale streams (different group_size per regime)
    scales_cheap: np.ndarray       # (n_tok_cheap, dim/gs_cheap) float16
    scales_rich: np.ndarray        # (n_tok_rich, dim/gs_rich) float16

    # Rich-only nz plane, gathered for rich tokens only
    nonzero_plane_rich: np.ndarray # (n_tok_rich, dim/8) uint8

    # Compaction indices: for token t with regime r,
    # the rich-only stream entry is at compact_index_rich[t]
    # (UINT32_MAX if t is cheap). Symmetrical for cheap.
    compact_index_rich: np.ndarray # (n_tok,) uint32
    compact_index_cheap: np.ndarray# (n_tok,) uint32

    fwht_seed: int
    dim: int
    group_size_rich: int           # gs_r (typically 32)
    group_size_cheap: int          # gs_c (typically 128)
```

**Encoder** (`encode_rroq158` extension):

1. Run today's full encode: produces `(sign, nz, scales@gs_r, cos, sin, cid)`.
2. Compute outlier mask: `is_rich = sin_norm >= np.quantile(sin_norm, 1−p)`.
3. For cheap tokens, re-encode `scales` at `gs_c` from the *already-FWHT-rotated
   residual* (we can reuse `r_rot` if cached, otherwise recompute — costs
   one extra FWHT per token but only on cheap tokens, run once at index time).
4. Drop `nonzero_plane` for cheap tokens; gather `nonzero_plane_rich` for the
   rich subset only.
5. Pack as above.

Storage accounting matches §2 to within the regime-byte overhead.

## 5. Kernel changes (`shard_engine/src/fused_rroq158.rs`)

Today's `score_pair_body` does, per (q_token, d_token):

```
s_g = popcount(sign_q & sign_d & nz_q & nz_d) - popcount(sign_q ^ sign_d & ...)
```

(Schematic — see kernel for exact bit pattern.)

For the cheap regime we need a sign-only variant that *implicitly* treats
every coordinate as nonzero:

```
s_g_cheap = popcount(sign_q & sign_d) - popcount(sign_q ^ sign_d)
          = group_size - 2·popcount(sign_q ^ sign_d)
```

The cheap path is **strictly cheaper** than the rich path (one popcount per
group instead of two) which gives a small additional latency win on top of
the smaller payload.

Branch structure (per d_token):

```rust
let regime_t = regime[t];                 // 0 = cheap, 1 = rich
let scales_t = if regime_t == 1 {
    &scales_rich[compact_index_rich[t]]
} else {
    &scales_cheap[compact_index_cheap[t]]
};
let nz_t = if regime_t == 1 {
    Some(&nz_rich[compact_index_rich[t]])
} else {
    None
};
// inner loop branches on regime_t once per d_token (amortized like
// the existing s_g amortization fix)
```

Branching once per d_token (not per q_token, not per coord) keeps the
cache pressure flat and aligns with the existing inner-loop reorder
optimization. Rich and cheap both share the same `sign_plane` stride, so
the popcount over `sign_q ^ sign_d` is regime-agnostic and runs uncondi-
tionally.

## 6. Phased implementation

The codec change touches the on-disk format, the Python encoder, the
PyO3 bindings, and the Rust kernel — **non-trivial**. Suggested phases
to de-risk:

### Phase A: Python-only score-merge prototype (no kernel changes)

Run two parallel encodes: `enc_rich` (today) and `enc_cheap` (sign-only,
implementable in pure Python by zeroing out `nonzero_plane` before the
existing kernel call). Per query token, score against *both* encodings,
then merge per (q_token, d_token) by a regime mask:

```python
score = np.where(is_rich[None, :], score_rich, score_cheap)
```

This **proves the ranking quality** of the hybrid before we burn engineering
budget on the kernel. If hybrid quality at p=0.10 already collapses, the
kernel change is moot. If hybrid quality at p=0.10 is within 0.5% NDCG@10
of baseline, the codec change is justified.

Estimated effort: ~1 day. New file: `research/low_bit_roq/hybrid_prototype.py`.
Output: a quality-vs-p curve on arguana matching the predicted-storage
table in §2.

### Phase B: On-disk format + encoder

Implement `Rroq158EncodedHybrid` and `encode_rroq158_hybrid` per §4.
Validate parity vs. Phase A's score-merge prototype.

Estimated effort: ~2 days. Touches: `quantization/rroq158.py`, on-disk
serialization in `shard_engine/serialize.py` (add a v2 format tag,
backward-compatible loader).

### Phase C: Rust kernel branch

Add `score_pair_body_hybrid` per §5. Microbench against the existing
`score_pair_body` to confirm cheap-path is faster, rich-path is parity.
Wire through PyO3.

Estimated effort: ~3 days. Touches: `kernels/shard_engine/src/fused_rroq158.rs`,
`kernels/shard_engine/src/lib.rs` (PyO3 surface), the GPU Triton
kernel `kernels/triton/rroq158.py`.

### Phase D: BEIR validation + default flip

Run the full 6-dataset BEIR sweep at the chosen p. If quality holds
(NDCG@10 within −0.005 of baseline on ≥ 5/6 datasets), open a PR to flip
the production default.

Estimated effort: 1 day (mostly compute time).

## 7. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Sign-only cheap regime tanks quality even with rescue | medium | high | Phase A proves this in Python in 1 day before any kernel work |
| Per-token branch breaks vectorization | low | medium | Branch once per d_token (existing pattern); microbench in Phase C |
| On-disk format break for existing indexes | high | medium | v2 format tag + backward-compatible loader; never auto-migrate |
| Per-query encoder needs a regime flag too | low | medium | Queries are NEVER cheap-encoded — query side stays at gs_r resolution; only docs are hybrid |
| Outlier selector mis-ranks tokens | medium | medium | Phase A ablates 4 selectors (sin_norm, MSE, energy, mixed) |
| `regime[t]` byte adds 1 B/tok overhead (3% of cheap floor) | certain | low | Already accounted for in §2 storage math |

## 8. Connection to the user's KV-cache work

The KV-cache prototype's smoke-test table (from the message thread):

| p (rescue) | b/d/stream | rel_err |
|---:|---:|---:|
| 0.00 | 1.55 | 0.600 |
| 0.10 | 1.65 | 0.581 |
| 0.30 | 1.85 | 0.544 |
| 0.50 | 2.05 | 0.508 |
| 1.00 | 2.55 | 0.423 |

Same Pareto pattern: p=0.30 lands in the sweet spot where reconstruction
error has dropped meaningfully (0.600 → 0.544, ~9% rel_err reduction)
while bit-rate has only grown 19% (1.55 → 1.85). The rroq158 analog
(this design) lands at p=0.30 → 1.975 b/coord vs. all-cheap 1.563 b/coord
— same shape, same trade-off geometry.

Cross-pollination opportunities:

* Reuse the KV-cache outlier selector's calibration (it already
  validated `sin_norm`-style energy ranking on a real workload).
* Share the v2 wire format conventions if both codecs end up in the
  same library so there's a single regime-byte format spec.
* If the KV-cache eval on TinyLlama-1.1B shows that p=0.30 generalizes
  across tasks, that strengthens the prior for rroq158 — both codecs
  are exploiting the same heavy-tailed residual distribution that
  ColBERT-style and decoder-style activations both seem to share.
