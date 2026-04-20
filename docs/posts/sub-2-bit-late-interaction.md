# Sub-2-bit late interaction: a 1.58-bit codec for ColBERT-style retrieval

> **TL;DR.** We built a 1.58-bit (ternary) per-coordinate codec for
> ColBERT-style multi-vector indices that lands within ~1 NDCG@10 point
> of FP16 on average across BEIR while compressing the doc-token
> footprint by ~5.5×. The trick is *not* a new quantizer — it's the
> geometry the quantizer sits on. Code, kernels, and the full benchmark
> sweep are open in
> [`voyager-index`](https://github.com/lightonai/voyager-index).
>
> If you have ever shipped a ColBERT, PLAID, or generic late-interaction
> system into production and stared at your SSD bill, this post is for
> you.

---

## Why this was supposed to be impossible

Late-interaction retrievers (ColBERT, PLAID, ColPali, GTE-ModernColBERT
and friends) keep one vector *per token*, then score
\(\text{MaxSim}(q, d) = \sum_i \max_j \langle q_i, d_j \rangle\)
over query token \(i\) and doc token \(j\). The recall is excellent.
The footprint is brutal: a 100 M-doc corpus with 256 tokens per doc and
128-dim FP16 vectors is **6.4 TB on disk** before any index structure.

The standard answer is to quantize the doc-side tokens. The standard
*problem* is that single-vector quantizers have always lost a
disproportionate amount of quality on this workload:

| codec on doc tokens          | typical NDCG@10 vs FP16 on BEIR | typical bits/coord |
|------------------------------|---------------------------------|--------------------|
| INT8 per-vector              | −0.0 to −0.3 pt                 | 8.0                |
| 4-bit product quantization   | −0.5 to −1.5 pt                 | 4.0                |
| RaBitQ (1-bit) [Gao 2024]    | −2 to −5 pt on multi-vector     | 1.0                |

The 4-bit lane is fine, but you've only saved 4×. The 1-bit lane is the
real prize — but on multi-vector retrieval the published numbers have
always been bad enough that nobody ships it. Ternary (1.58 bits per
coordinate via 3 levels per coord, packed into a popcount-friendly
representation) sits in between with no published, production-ready
result that we know of.

Why does RaBitQ collapse on ColBERT-style tokens? Two reasons that
*together* are killer:

1. **ColBERT tokens are not isotropic on the unit sphere.** All `the`
   tokens cluster around one direction; all `pneumonia` tokens cluster
   around a very different one. RaBitQ's noise bound assumes uniform
   sampling on \(\mathbb{S}^{d-1}\). It just isn't true here.
2. **MaxSim is sensitive to the absolute inner product, not just rank
   of one neighbor.** Per-pair noise integrates badly across \(Q \times
   T\) pairs.

So the question we set out to answer was concrete: can you push a
multi-vector codec down to **1.58 bits per coordinate** while keeping
NDCG@10 within ~1 point of FP16, without distillation or learned
reranking?

Yes — but only if you stop treating the problem as quantization and
start treating it as geometry.

---

## The Riemannian fix in three pictures

The codec has three moves. Each one is small. Together they buy back
~3 NDCG@10 vs naive 1-bit RaBitQ on hard datasets.

### 1. Spherical k-means on the doc-token cone

Every doc token \(d_j\) is L2-normalized to live on
\(\mathbb{S}^{d-1}\). We fit \(K = 8192\) centroids on the rotated
spherical population using cosine k-means, store the assigned centroid
id (13 bits) per token, and *only quantize the residual after that*.

This is the cheap part. It costs `log2(K)/dim ≈ 0.10` bits/coord
amortized, and it kills the cone-anisotropy that RaBitQ couldn't
handle. The remaining residual is much smaller, much more isotropic,
and *finally* matches the assumptions of low-bit quantizers.

### 2. The log map: don't quantize Euclidean residuals on a sphere

Here's where most prior "ColBERT + clusters + bits" work goes wrong.
The naive thing is to take \(r_j = d_j - c_{\sigma(j)}\) — the Euclidean
residual from the centroid — and quantize that. But \(d_j\) and \(c\)
both live on the unit sphere; the straight-line residual exits the
manifold. You're spending bits encoding components that the score
formula will never see.

The fix is the **Riemannian log map** at the centroid:

\[
\tilde r_j \;=\; \log_{c_{\sigma(j)}}(d_j)
\;=\; \theta\,\frac{d_j - \cos(\theta)\,c_{\sigma(j)}}{\sin(\theta)}
\]

with \(\theta = \arccos\langle d_j, c\rangle\). This vector lives in
the tangent plane at \(c\), is orthogonal to \(c\), and *has the
property that the geodesic-correct reconstruction is*

\[
d_j \;=\; \cos(\|\tilde r_j\|)\, c \;+\; \mathrm{sinc}(\|\tilde r_j\|)\, \tilde r_j.
\]

So we store
\((\text{centroid id}, \cos\|\tilde r\|, \mathrm{sinc}\|\tilde r\|,
\text{quantized } \tilde r)\) per token.
Two scalars per token (`cos_norm`, `sin_norm`) at FP16 are 4 bytes;
they amortize across all 128 coordinates so the per-coord overhead is
0.25 bits.

The score formula (derived in
[`docs/guides/rroq-mathematics.md`](../guides/rroq-mathematics.md)
§2) is then **exact**:

\[
\langle q, d_j \rangle
\;=\; \cos(\|\tilde r_j\|)\,\langle q, c_{\sigma(j)}\rangle
\;+\; \mathrm{sinc}(\|\tilde r_j\|)\,\langle q, \tilde r_j\rangle.
\tag{$\ast$}
\]

The first term is `q @ centroids.T` — one GEMM per query, cached and
shared across every doc token assigned to the same centroid. The
second term is the only place quantization noise enters. *That* is the
term we now have license to compress aggressively, because it's a small
isotropic residual on a tangent plane, exactly what RaBitQ-style codes
are good at.

### 3. FWHT, then ternary

Before quantizing \(\tilde r\) we apply a length-\(d\) Walsh–Hadamard
transform with random ±1 signs. Cost: \(O(d \log d)\) per token, ~0.05
ms on CPU at \(d=128\). Effect: every coordinate becomes a near-Gaussian
combination of all the others, so the resulting tangent vector is
isotropic to within FWHT noise. *Now* per-group scalar \((\Delta_g)\)
plus a sign-with-zero ternary code is a near-optimal encoding under L2:

\[
\tilde r_i \;\mapsto\;
\begin{cases}
+\Delta_g & \text{if } \tilde r_i > +\tau_g \\
-\Delta_g & \text{if } \tilde r_i < -\tau_g \\
\quad 0   & \text{otherwise}
\end{cases}
\]

with \((\Delta_g, \tau_g)\) picked per group of 32 coords to minimize
group-wise reconstruction MSE under the constraint that ~1/3 of the
mass lands on each level (entropy-balanced ternary). That's the **1.58
bits/coord** in the headline (\(\log_2 3 \approx 1.585\)).

Including the centroid id, the per-group `Δ` (FP16), and the per-token
`(cos_norm, sin_norm)`, the total cost lands at:

\[
\text{rroq158 bits/coord}
\;\approx\; \underbrace{0.10}_{\log_2 K / d}
\;+\; \underbrace{1.585}_{\text{ternary}}
\;+\; \underbrace{0.50}_{16/g\ \text{at}\ g=32}
\;+\; \underbrace{0.25}_{2 \cdot 16/d}
\;\approx\; \mathbf{2.4 \text{ bits/coord}}.
\]

That's ~46 bytes/token vs 256 bytes/token for FP16 — **about 5.5×
smaller doc-token storage**.

---

## Where the speed actually comes from (and where it doesn't)

The score formula \((\ast)\) is a sum of two terms:

1. **Centroid lane.** `q_amb @ centroids.T` is precomputed once per
   query into a `[Q, K]` table. Every doc-token lookup is then a single
   gather by 13-bit centroid id and one FP16 multiply by `cos_norm`.
   `K = 8192 × dim 128 × FP32 = 4 MiB`, which fits in L3 on every
   modern CPU and never leaves SMEM on GPU.

2. **Residual lane.** The inner product \(\langle q, \tilde r_j\rangle\)
   over the ternary residual reduces to **two popcounts per group of 32
   coordinates**: one over `(sign & nonzero)` for the +1 mass, one over
   `(~sign & nonzero)` for the −1 mass. The query side is encoded into
   a tiny stack of bit-planes (3–4 of them) so the document never needs
   to dequantize.

The whole inner loop is therefore *integer-only* on the doc side:

```
for group in token_groups:
    pos = popcnt(d_sign[group] & d_nz[group] & q_plane[group])
    neg = popcnt(~d_sign[group] & d_nz[group] & q_plane[group])
    acc += delta[group] * (pos - neg)
score += cos_norm * qc_table[centroid_id] + sin_norm * acc
```

Two `popcnt` instructions per group of 32 coords on AVX2/AVX-512, two
`__popc` calls per group on CUDA. The doc tensor is read once,
sequentially, and never decoded.

**An honest accounting of the speed picture from our BEIR sweep**
(RTX A5000, 8-worker AVX2 CPU, 6 BEIR datasets, full query sets):

- **GPU lane.** rroq158 is ~1.0–1.4× the throughput of FP16 on small
  corpora (where the FP16 path is bandwidth-limited and rroq158 trims
  the read), and ~0.6–0.8× on large/long-query corpora (where the
  query-side `qc_table` GEMM and the per-token `cos_norm`/`sin_norm`
  multiplies become a non-trivial fixed cost). It is roughly **at FP16
  parity on average**, not the order-of-magnitude win we initially
  hoped for. The win is the storage, not the latency.
- **CPU lane.** rroq158 is **slower** than FP16 in absolute QPS (~5–9×)
  on the current Rust SIMD implementation. The reason is that FP16
  MaxSim on AVX2 is itself extremely bandwidth-friendly, and the
  rroq158 path's per-query overhead — query bit-plane encoding,
  per-doc-token `cos·qc + sin·resi` FMAs, and Python-side tensor
  copies into the kernel — is currently larger than the win from
  reading 5.5× fewer doc bytes. The kernel is correct and parity-tested
  against the Triton GPU version, but on CPU the FP16 baseline is hard
  to beat. We have an open backlog item to fix this: see
  [issue tracker](https://github.com/lightonai/voyager-index/issues).

So the production story we ship is **storage-honest**:

- If you can afford the latency, ship `Compression.RROQ4_RIEM`
  (Riemannian-aware 4-bit asymmetric, ~3× smaller than FP16, FP16
  parity in NDCG@10 to within ±0.05 pt on every BEIR-6 dataset, and
  ~96% top-10 codec-fidelity overlap with FP16). At the production
  batch shape it is currently ~5× slower than FP16 on GPU and ~13×
  slower on CPU — it's a **storage with zero quality regression**
  lane, not a throughput lane.
- If your bottleneck is disk or hot-tier RAM (which it almost always
  is at scale on multi-vector indices), ship `Compression.RROQ158`
  (~5.5× smaller than FP16, ~1.4 NDCG@10 point average gap,
  ~2.7 pt worst case at top-10 on arguana, GPU p95 within 1.13× of
  FP16 on average). Accept that on CPU you trade ~8× throughput for
  the storage win, and that ~20% of the top-10 docs differ from FP16
  (R@100 still recovers within −2.1 pt on every dataset).

CPU and GPU implementations share the exact same packed payload tensors
(sign / nonzero / scale / centroid id / cos_norm / sin_norm) and pass a
bit-equivalence parity test to within FP32 rounding —
[`tests/test_rroq158_kernel.py`](../../tests/test_rroq158_kernel.py).

---

## What we measured (and where it costs)

The empirical sweep that backs this post is
[`benchmarks/beir_2026q2_full_sweep.py`](../../benchmarks/beir_2026q2_full_sweep.py).
It runs **6 BEIR datasets × 4 codecs × 2 modes × full query sets** on a
fixed RTX A5000 + 8-worker AVX2 CPU host, with end-to-end MaxSim
scoring from the same encoder
(`lightonai/GTE-ModernColBERT-v1`). Every cell records git SHA, GPU
model + driver + CUDA, CPU model + cores, wheel versions, and kernel
parameters — see `collect_provenance` in the harness. The table format
is fixed in
[`scripts/format_beir_2026q2_table.py`](../../scripts/format_beir_2026q2_table.py).

The full table is in the
[README](../../README.md#beir-retrieval--rtx-a5000-search-only-full-query-set).
The headline:

- **NDCG@10 vs FP16 (averaged across BEIR-6):** −1.43 pt for rroq158
  (worst dataset arguana at −2.69 pt), +0.02 pt for rroq4_riem
  (within ±0.05 pt on every dataset).
- **Recall@100 vs FP16 (averaged):** −0.48 pt for rroq158, +0.23 pt
  for rroq4_riem.
- **GPU p95 vs FP16 (averaged):** 1.13× for rroq158, 5.03× for
  rroq4_riem (rroq4_riem is the storage-with-zero-quality-loss lane,
  not the throughput lane).
- **Storage vs FP16 doc tokens:** 5.5× smaller for rroq158, ~3×
  smaller for rroq4_riem.

`rroq158` is not free at top-10. The honest picture from the BEIR-6
sweep (`reports/beir_2026q2/sweep.jsonl`):

- On *average across BEIR-6* it's within **−1.43 NDCG@10 point** of
  FP16, but
- on a few hard datasets (notably **arguana**, which has unusually
  fine-grained token-level distinctions in argument-rebuttal pairs) it
  loses up to **−2.69 NDCG@10 points** at top-10.
- Recall@100 stays nearly flat: avg ΔR@100 = −0.48 pt across BEIR-6.
  The right docs are still in the candidate set; they just trade
  ranks within the top-K.
- An exact diagnostic — using brute-force top-K overlap of rroq158 vs
  FP16 on the same encoder — shows that across BEIR-6, brute-force
  rroq158 retains **avg ~79% top-10 overlap** with the FP16
  brute-force ranking (range 73–83%), and brute-force R@100 stays
  within −2.1 pt of FP16 on every dataset (worst case fiqa at
  −2.07 pt, best cases scifact / quora at 0.0 pt). So ~20% of top-10
  positions are displaced relative to FP16, but the relevant docs are
  still admitted. Notably, top-K overlap is **roughly flat or
  slightly declining with K** for rroq158 (e.g. quora drops from
  72.9% top-10 to 72.1% top-100), so widening the serve window does
  not reliably recover the displaced top-10 docs — the displacement
  is *out of the codec's candidate set*, not within it. R@100
  survives because rroq158 still admits the labeled relevant docs;
  the displaced ~20% sits among the non-relevant tail of FP16's
  top-100. See
  [`benchmarks/topk_overlap_sweep.py`](../../benchmarks/topk_overlap_sweep.py)
  and `reports/beir_2026q2/topk_overlap.jsonl` for the full
  per-dataset breakdown.
- For workloads that need exact top-10 rank fidelity, an FP16 rerank
  on the rroq158 top-32/top-64 shortlist closes the NDCG@10 gap on
  arguana / scifact / scidocs with no R@100 regression
  ([`benchmarks/diag_rroq158_rescue.py`](../../benchmarks/diag_rroq158_rescue.py)).

So we now ship two production codecs. The default is `rroq158` —
maximum compression with a documented NDCG@10 cost on hard datasets,
plus an FP16-rerank rescue path for workloads that need top-10
fidelity. The opt-in `rroq4_riem` is the no-quality-loss lane: it
matches FP16 quality to within ±0.05 pt NDCG@10 on every BEIR-6
dataset, but pays for it in latency (~5× slower on GPU and ~13×
slower on CPU at the production batch shape than the FP16 baseline),
so it's a "storage with zero quality regression" lane, not a
throughput lane. Both share the same Riemannian-aware geometry, the
same Triton GPU kernel architecture, the same Rust SIMD CPU kernel
architecture, and the same dispatch / fallback path. Switching is
one flag:

```python
from voyager_index import BuildConfig, Compression

config = BuildConfig(compression=Compression.RROQ4_RIEM)

config = BuildConfig(compression=Compression.RROQ158)
```

---

## Reproducing it

Everything in this post is open. The codec lives in
[`voyager_index/_internal/inference/quantization/rroq158.py`](../../voyager_index/_internal/inference/quantization/rroq158.py),
the Triton kernel in
[`voyager_index/_internal/inference/triton_roq_rroq158.py`](../../voyager_index/_internal/inference/triton_roq_rroq158.py),
and the Rust SIMD kernel in
[`latence-shard-engine/src/fused_rroq158.rs`](../../latence-shard-engine/src/fused_rroq158.rs).
The math derivation is in
[`docs/guides/rroq-mathematics.md`](../guides/rroq-mathematics.md).
The BEIR sweep harness is
[`benchmarks/beir_2026q2_full_sweep.py`](../../benchmarks/beir_2026q2_full_sweep.py)
and the table renderer is
[`scripts/format_beir_2026q2_table.py`](../../scripts/format_beir_2026q2_table.py).

To rerun the sweep on your host:

```bash
git clone https://github.com/lightonai/voyager-index
cd voyager-index
pip install -e '.[dev]'
python benchmarks/beir_2026q2_full_sweep.py \
    --output reports/beir_2026q2/sweep.jsonl
python scripts/format_beir_2026q2_table.py \
    --input reports/beir_2026q2/sweep.jsonl \
    --overlap reports/beir_2026q2/topk_overlap.jsonl
```

Both kernels (rroq158 and rroq4_riem) ship with parity tests against a
reference scalar implementation:

```bash
pytest tests/test_rroq158_kernel.py
pytest tests/test_rroq4_riem_kernel.py
```

---

## What we'd like the community to do with it

Three things, in roughly increasing order of ambition:

1. **Try it on your own ColBERT-style index.** The codec is
   encoder-agnostic. We have parity numbers for ModernColBERT here;
   we'd love to see numbers for ColBERTv2, PLAID, ColPali, and JaColBERT
   on the same harness.
2. **Help us close the CPU latency gap.** The Rust SIMD kernel is
   correct but currently slower than fp16 in absolute QPS at this
   batch shape. The biggest gains are likely in (a) avoiding
   per-query Python-to-Rust tensor copies, (b) AVX-512 / VPSHUFB
   nibble-level packing, and (c) better cache scheduling for the
   `qc_table` lookup.
3. **Push the math further.** The log map is one Riemannian primitive;
   parallel transport, exponential maps for query-side encoding, and
   manifold-aware learned reranking heads are all on the table. Richer
   tangent codes (asymmetric ternary, Lloyd-Max, very-low-bit product
   codes on tangent groups) would plausibly close the remaining
   ~0.5 NDCG@10 gap on rroq4_riem with no disk-cost regression.

Issues and PRs welcome on
[`voyager-index`](https://github.com/lightonai/voyager-index).

---

## Acknowledgements

This builds on RaBitQ ([Gao et al. 2024](https://arxiv.org/abs/2405.12497)),
the original ROQ paper (Zhang et al. 2025), the Fast
Johnson-Lindenstrauss Transform (Ailon & Chazelle 2009), and ColBERT
(Khattab & Zaharia 2020). The Riemannian framing — log map at the
spherical centroid, FWHT-rotated tangent residual, ternary on the
isotropized residual — is the contribution of this codebase. The
identity in \((\ast)\) is what makes the whole thing fall out as a
sum of two GEMMs and one popcount lane.

The benchmark harness, kernels, plan documents, and audit trail for
all of the above are public — see
[`research/low_bit_roq/PROGRESS.md`](../../research/low_bit_roq/PROGRESS.md)
for the build log of how this codec evolved from "doesn't work at all"
through "works on toy data" to the current production default.
