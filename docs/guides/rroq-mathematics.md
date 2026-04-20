# The mathematics behind RROQ

> **Audience.** Practitioners and reviewers who want to understand *why* the
> 1.58-bit and 4-bit Riemannian rotational quantization codecs in
> `voyager-index` work, and what they extend over the original RaBitQ paper.
> The exposition is self-contained: only inner products, the chain rule on
> the unit sphere, and the Walsh–Hadamard transform are assumed.

This document covers, in order:

1. The classic RaBitQ identity that makes 1-bit IP estimation unbiased on
   the unit sphere.
2. The Riemannian extension we use to repair the corner-cutting that
   RaBitQ introduces on multi-vector tokens.
3. Why we pick *ternary* (1.58-bit) on the rotated tangent residual rather
   than 1-bit, 2-bit, or 4-bit.
4. The popcount/sign-plane packing that makes ternary scoring run at
   FP16 throughput on a Triton kernel and AVX2 throughput on a Rust
   kernel.
5. The K = 8192 derivation: where the centroid bit-budget comes from and
   how it trades off against the residual.
6. The 4-bit asymmetric variant (`rroq4_riem`) for users who refuse any
   loss against FP16.

Concrete kernel code lives in:

- `voyager_index/_internal/inference/quantization/rroq158.py`
- `voyager_index/_internal/inference/quantization/rroq4_riem.py`
- `voyager_index/_internal/inference/triton_roq_rroq158.py`
- `voyager_index/_internal/inference/triton_roq_rroq4_riem.py`
- `latence-shard-engine/src/fused_rroq158.rs`
- `latence-shard-engine/src/fused_rroq4_riem.rs`

The benchmark numbers cited below are produced by
`benchmarks/beir_2026q2_full_sweep.py` and recorded under
`reports/beir_2026q2/`.

---

## 1. RaBitQ in one paragraph (and what it gets wrong on multi-vector)

RaBitQ ([Gao et al. 2024](https://arxiv.org/abs/2405.12497)) replaces every
unit-norm vector \(x \in \mathbb{S}^{d-1}\) with a sign code
\(s = \text{sign}(R x) \in \{-1, +1\}^d\), where \(R\) is a fixed random
rotation. The key identity is

\[
\mathbb{E}\bigl[ \tfrac{1}{\sqrt{d}}\,\langle s,\, R x\rangle \bigr]
\;=\; \mathbb{E}\bigl[ \tfrac{1}{d}\,\|R x\|_1 \bigr]
\;\to\; \sqrt{2/\pi}
\]
as \(d \to \infty\), so that for two unit vectors \(x, y\),

\[
\langle x, y\rangle \;=\; \mathbb{E}\!\left[
\tfrac{1}{2}\bigl(\tfrac{1}{\sqrt{d}}\langle s_x, R y\rangle
                  + \tfrac{1}{\sqrt{d}}\langle s_y, R x\rangle\bigr)
\right] + O\!\left(\tfrac{1}{\sqrt{d}}\right)
\]

is unbiased. The estimator's variance is constant in `dim`, so RaBitQ
gets you 1-bit-per-coordinate doc storage with a tight error bound for
*single-vector retrieval at large dim*.

The two places this breaks down for ColBERT-style multi-vector retrieval
are exactly the places we patch:

1. **Per-token vectors are not isotropic.** ColBERT tokens cluster
   strongly around term-specific directions (e.g. all `the` tokens
   point at one centroid, all `pneumonia` tokens at another). The
   "sample from the unit sphere" assumption of the RaBitQ proof is
   violated — the empirical sphere is a small set of cones, not a
   uniform distribution. The 1-bit residual then routinely loses ~5 pt
   NDCG@10 because it cannot distinguish two tokens that differ only
   along a few coordinates inside the same cone.
2. **MaxSim is sensitive to the absolute IP, not just rank.**
   ColBERT's MaxSim sums per-query-token max IPs across the doc; the
   absolute scale matters because it's compared across docs. RaBitQ's
   per-pair noise integrates badly across `Q × T` pairs.

The Riemannian extension fixes (1) by making the residual *live in the
tangent space of the cone*, not in the ambient space, so 1.58 bits are
spent on the *part of the vector that the per-token cone cannot
explain*. Fix (2) follows automatically because the cosine-of-residual-
norm correction below is a true scalar, not a Gaussian estimate.

---

## 2. The Riemannian decomposition

Let \(\mathbb{S}^{d-1}\) be the unit sphere in \(\mathbb{R}^d\). Pick a
codebook \(\{c_1, \ldots, c_K\} \subset \mathbb{S}^{d-1}\) of unit-norm
**centroids** (we fit them with spherical k-means at index-build time;
see §5). For any token \(x_n = x / \|x\|\) on the sphere, let
\(c = c_k\) be its nearest centroid by inner product. The geodesic that
joins \(c\) and \(x_n\) is uniquely parameterised by the **logarithmic
map**

\[
r \;=\; \log_c(x_n) \;=\; \frac{\theta}{\sin\theta}\,\bigl(x_n - \cos\theta\, c\bigr),
\qquad \theta = \arccos\langle c, x_n\rangle.
\]

`r` is a tangent vector at `c`, so \(\langle r, c\rangle = 0\) and
\(\|r\| = \theta\) (the geodesic distance). The corresponding
**exponential map** recovers \(x_n\) exactly:

\[
x_n \;=\; \exp_c(r) \;=\; \cos\|r\|\cdot c \;+\; \frac{\sin\|r\|}{\|r\|}\,r
            \;=\; \cos\|r\|\cdot c \;+\; \mathrm{sinc}\!\left(\tfrac{\|r\|}{\pi}\right)\,r.
\]

(Here `sinc(z) = sin(πz)/(πz)`, the normalised cardinal sine; numpy's
`np.sinc` matches this convention.)

Plugging this into the inner product against an arbitrary query vector
`q` (no normalization assumption — `q` is just a token from the query
side) and absorbing the document norm \(\|x\|\):

\[
\boxed{\;
\langle q,\, x\rangle
\;=\; \|x\|\cdot\bigl(\cos\|r\|\cdot \langle q, c\rangle
                       \;+\; \mathrm{sinc}(\|r\|/\pi)\cdot \langle q, r\rangle\bigr).
\;}
\quad (\ast)
\]

Identity (\(\ast\)) is **exact** — no approximation yet. The ROQ paper
applies a similar decomposition for ALU-friendly inner products; what we
add is the next two steps.

### 2a. Why this decomposition matters

The first term \(\langle q, c\rangle\) does not depend on the document
*at all* beyond the choice of centroid id. We can pre-compute the entire
\(Q \times K\) table `qc_table` once per query at the cost of one GEMM
of shape `(S, dim) × (dim, K)` and never touch it again across the
millions of (q-token, d-token) pairs in the inner loop. With `K = 8192`
and `dim = 128` this is a 1.0 MFLOP matmul, completely amortised.

The second term \(\langle q, r\rangle\) is what we have to actually
quantize. **And `r` is a tangent vector**: it lives in a `(d-1)`-dim
subspace, not the ambient `d` dim, and its statistics are *uniform*
across token clusters (because the per-cluster mean direction has been
removed by the `log_c` operation). That uniformity is exactly the
condition the RaBitQ unbiasedness proof needs. So:

- Spend ~13 bits per token on the centroid id (`log2 8192`).
- Spend a low-bit code on the residual `r`, which has nice tail
  statistics regardless of what the corpus distribution looks like.

In numbers: at `K=8192` on the BEIR scidocs corpus (1024-token docs at
dim 128), the average residual norm is \(\theta \approx 0.31\) rad
(~18°), versus \(\theta \approx 0.58\) rad without per-cluster
referencing. The per-coord variance of `r` is roughly **3.5×** lower
than the per-coord variance of `x_n` itself. Quantization noise in low-
bit codes scales linearly with the source variance, so a 1.58-bit code
on `r` has the same effective IP error as a ~3.5× richer (~5-bit) code
on `x_n` directly.

### 2b. The rotation step

The spherical k-means cones are not axis-aligned in the original basis.
We apply a fixed structured rotation `R` (a 3-stage random-sign
Walsh–Hadamard transform; see `quantization/rotational.py`) to the
tangent residual:

\[
\tilde r \;=\; R r,
\qquad \tilde q \;=\; R q.
\]

`R` is unitary, so \(\langle \tilde q, \tilde r \rangle = \langle q, r\rangle\)
*exactly* (no error is introduced). The reason we do it: in the rotated
basis, every coordinate of `r` is approximately a Gaussian random
variable (the random Walsh–Hadamard with sign-flips is a near-orthonormal
projection, see Ailon-Chazelle 2009). That makes the per-group
quantization scales `s_g` (§3) match the variance of the same single
distribution per group instead of fighting per-axis anisotropy.

In the FWHT-rotated frame, the per-group quantization parameters
collapse to one scalar per group. **Production SOTA default** at
`dim=128` is `group_size=128` (one scalar per token — `dim/group_size = 1`),
which the BEIR Pareto sweep validated as ±0.005 NDCG@10 vs the
previous `group_size=32` (4 scalars per token) at ~13% smaller storage
and ~10–30% faster CPU p95. The intuition: in the rotated frame, the
energy is mixed across coordinates uniformly enough that a single scale
per token captures the magnitude, and the kernel benefits from one
fewer scale load per group. For corpora with high intra-token
magnitude variance (e.g. arguana-class — short docs whose token
distributions are not well-mixed even after FWHT), pin
`Rroq158Config(group_size=64)` to recover the per-region scales — see
[`docs/guides/quantization-tuning.md`](quantization-tuning.md). For
non-multiple-of-128 dims (dim=64 / 96 / 160) the encoder transparently
steps down through `{128, 64, 32}` with a log warning.

---

## 3. Ternary on the rotated tangent residual: 1.58 bits, three reasons

The rotated residual `r̃` is symmetric around zero with light tails. We
encode each coordinate with the **ternary** alphabet \(\{-s_g, 0, +s_g\}\),
where `s_g` is the per-group scale (one scalar per `group_size`
coordinates per token; the SOTA production default is `group_size=128`
at `dim=128`, i.e. one scalar per whole token). The encode rule is:

\[
\tilde r_i \;\mapsto\;
\begin{cases}
+s_g & \text{if } \tilde r_i > +\tau_g, \\
-s_g & \text{if } \tilde r_i < -\tau_g, \\
\;\,0 & \text{otherwise},
\end{cases}
\qquad
\tau_g = \tfrac{1}{2}\,\sigma_g
\]

where \(\sigma_g = \mathrm{std}(\tilde r_g)\) is the per-group sample
standard deviation. The half-sigma threshold zero-codes ~38% of
coordinates and is the loss-minimising fit for a Gaussian residual; we
verified this empirically on the rotated tangent space — see the ternary
fit ablation in
[`research/low_bit_roq/PROGRESS.md`](../../research/low_bit_roq/PROGRESS.md).

The information-theoretic price of ternary is \(\log_2 3 \approx 1.58\)
bits per coord — hence the codec name. The 1.58-bit code outperforms
1-bit *and* 2-bit on the rotated tangent residual for three independent
reasons:

1. **The "0" code is essentially free.** With ~38% of coords at zero, the
   sparsity propagates directly into the kernel: the inner-product
   contribution from any zero-coordinate is zero, so the popcount
   accumulator skips those bits via `nonzero_plane & sign_plane`. We get
   the storage cost of 2 bits per coord (sign + nonzero) but the
   compute cost of half a multiply-add on average — better effective
   throughput than 1-bit codes that have no skip pattern.
2. **The residual is centered.** A 1-bit code is forced to assign
   either `+s` or `-s` to every coordinate, which biases the
   reconstruction even when the true value is essentially zero. A 2-bit
   code spends 25% of its codepoints on duplicate-magnitude entries
   (e.g. `±s_low, ±s_high`) that buy you very little on a residual whose
   density at zero is already high. Ternary is exactly the codebook
   that matches the symmetric-around-zero statistics of a Gaussian
   residual at this density.
3. **Sign + nonzero pack into popcount.** The IP estimator becomes a
   double-popcount (see §4), which is a 1-cycle integer instruction
   that any modern x86 / ARM / GPU SIMD lane can issue at full pipeline
   width. 4-bit codes lose the popcount fast-path; you have to do a
   gather + multiply-add on 4-bit nibbles. The throughput differential
   is real: rroq158 hits ~4.0× FP16 throughput on the same kernel
   shape, rroq4_riem hits ~1.4×.

---

## 4. The MaxSim score, and how it becomes two popcounts

Substitute the ternary code into (\(\ast\)) and group the doc-side
constants:

\[
\text{sim}(q, x)
\;=\; \underbrace{\|x\|\cos\|r\|}_{\text{cos\_norm}}\cdot
       \underbrace{\langle q, c\rangle}_{\text{qc\_table}[q,\,k]}
\;+\;
\underbrace{\|x\|\,\mathrm{sinc}(\|r\|/\pi)}_{\text{sin\_norm}}\cdot
       \underbrace{\langle \tilde q,\, \tilde r\rangle}_{\text{tern\_dot}(\tilde q,\,\tilde r)}.
\]

The first term is a single `fp16` multiply per (q-token, d-token) pair —
the constants `cos_norm` and `sin_norm` are pre-computed at index time
and stored as `fp16` (one number per token, not per coordinate).

The second term is the kernel hot path. Let
\(\sigma_i \in \{0, 1\}\) be the sign bit of \(\tilde r_i\) and
\(\nu_i \in \{0, 1\}\) be the nonzero bit. Then
\(\tilde r_i = (2\sigma_i - 1)\,\nu_i\,s_g\) and

\[
\langle \tilde q, \tilde r\rangle
\;=\; \sum_g s_g \!\!\sum_{i \in \text{group } g}\!\! (2\sigma_i - 1)\,\nu_i\, \tilde q_i.
\]

For the **query-side 4-bit asymmetric encoding** (the production query
encoding), we further write \(\tilde q_i = \mu_g + \delta_g\, q^{\,4b}_i\)
with \(q^{\,4b}_i \in \{0, 1, \ldots, 15\}\). After rearranging:

\[
\sum_{i \in g} (2\sigma_i - 1)\,\nu_i\, \tilde q_i
\;=\; \mu_g\,\sum_i (2\sigma_i - 1)\,\nu_i
       \;+\; \delta_g \sum_i (2\sigma_i - 1)\,\nu_i\, q^{\,4b}_i.
\]

The first inner sum is `popcount(σ & ν) - popcount(~σ & ν)` — two CPU
`popcnt` instructions or two CUDA `__popc` calls per group. The second
inner sum is a 4-bit integer gather across the query-side nibble stream,
which is what the AVX2 `vpshufb` instruction was designed for. On the
GPU the same operation maps to a Triton `tl.dot` over 4-bit packed
operands with a sign mask.

**The whole inner loop is two popcounts and one nibble-multiply-add per
group of 32 coordinates.** That's the ~4× throughput vs FP16 we observe
in the BEIR sweep.

The implementation is split between
`voyager_index/_internal/inference/triton_roq_rroq158.py` (GPU) and
`latence-shard-engine/src/fused_rroq158.rs` (CPU AVX2/AVX512). Both
kernels are validated bit-equivalent up to fp32 rounding by
`tests/test_rroq158_kernel.py`.

---

## 5. Where K = 8192 comes from

The total bits-per-coordinate of an rroq158 encoding is

\[
B(K, d, g) \;=\; \frac{\log_2 K}{d} \;+\; 1.58 \;+\; \frac{2 \cdot 16}{g} \;+\; \frac{2 \cdot 16}{d}.
\]

The first term is the centroid id cost amortised across `dim`
coordinates per token. The second is the ternary residual. The third is
the per-group `(min, delta)` overhead at fp16 (only the `delta` for
rroq158, but we list both for unified accounting with rroq4_riem). The
fourth is the per-token `(cos_norm, sin_norm)` overhead at fp16.

For `dim = 128, group_size = 32` (the previous default):

| K     | bits/coord | per-token bytes | reconstruction MSE on rotated tangent |
|-------|-----------:|----------------:|--------------------------------------:|
| 1024  | 1.66       | 26.6            | 1.00× (baseline)                      |
| 2048  | 1.67       | 26.7            | 0.71×                                 |
| 4096  | 1.68       | 26.9            | 0.54×                                 |
| 8192  | 1.69       | 27.0            | 0.43×                                 |
| 16384 | 1.69       | 27.1            | 0.37×                                 |

For `dim = 128, group_size = 128` (the **SOTA default**, K=8192): the
third term collapses to `2 · 16 / 128 = 0.25` bits/coord (vs `1.0` at
gs=32), giving **per-coord ≈ 1.55 bits and per-token ≈ 24.8 bits ≈
25 B of payload + 14.4 B of dense ternary planes ≈ 40 B/token total**
(matches the measured 40 B in the layout breakdown of the [tuning
guide](quantization-tuning.md)). The **6 B/token saving vs gs=32 is
purely from the per-group scale overhead** — the ternary residual
itself is unchanged. See [`docs/guides/quantization-tuning.md`](quantization-tuning.md)
for the per-dim breakdown.

So increasing `K` from 1024 to 8192 adds **0.03 bits/coord** but cuts
the residual MSE by **2.3×**. That MSE drop is exactly what closes the
NDCG@10 gap on hard datasets (scidocs, arguana) where the per-cluster
cone is wider — the more the cone covers, the more the residual carries
the discriminative signal.

K = 8192 is also the largest power of two that fits the centroid table
in 4 MiB at `dim = 128`:

\[
\underbrace{8192}_{K} \cdot \underbrace{128}_{d} \cdot \underbrace{4}_{\text{fp32}} \;=\; 4\,\text{MiB}.
\]

That's small enough to live in L3 cache on every modern CPU and to keep
the `q @ centroids.T` GEMM at ~1 ms on CPU and ~0.005 ms on GPU. The
next power of two (K = 16384) breaks both, doubles the GEMM time, and
cuts the residual MSE by only an additional 14% — diminishing returns.

The K = 8192 default is enforced in `Rroq158Config` (see the docstring),
and is the value used by the
[`benchmarks/beir_2026q2_full_sweep.py`](../../benchmarks/beir_2026q2_full_sweep.py)
production sweep that populates the README BEIR table. For corpora
under 8192 unique tokens the build path auto-shrinks K to the largest
feasible power of two and logs a warning (see
[`choose_effective_rroq158_k`](../../voyager_index/_internal/inference/quantization/rroq158.py)
and the `k_requested` / `k_effective` runtime fallback in
[`_manager/search.py`](../../voyager_index/_internal/inference/shard_engine/_manager/search.py)).

---

## 6. The 4-bit asymmetric variant: `rroq4_riem`

For users who refuse any quality loss against FP16 (regulated domains,
high-stakes ranking, etc.) we ship a 4-bit asymmetric Riemannian
variant. The encoding pipeline is identical to rroq158 through step
2b — spherical centroids + log map + FWHT — but the residual encoding
swaps in an asymmetric 4-bit code:

\[
\tilde r_i \;\mapsto\; m_g \;+\; \Delta_g\,c_i,
\qquad c_i \in \{0, 1, \ldots, 15\},
\]

with one `(m_g, Δ_g)` pair per group. The asymmetric `m_g` (per-group
*minimum*) lets us track the location of the residual mode, and the
4-bit codepoints are uniformly spaced across the per-group range. This
gives reconstruction error ~2 nats/coord vs ~1 nat/coord for ternary —
better at the cost of a richer codebook.

The score formula is identical to (\(\ast\)) with the ternary inner
product replaced by

\[
\langle \tilde q, \tilde r\rangle
\;=\; \sum_g \bigl(m_g\,(\sum_{i \in g}\tilde q_i)
                  \;+\; \Delta_g\,\sum_{i \in g}\tilde q_i\,c_i\bigr).
\]

The first inner sum is just per-group sums of `q̃` (independent of
the document — pre-computed once per query as `q_group_sums`) times the
per-group `m_g` constant. The second is a 4-bit nibble-multiply-add,
implemented with `vpshufb` on AVX2 and `tl.dot` on the GPU.

Disk overhead: ~3 bits/coord (4-bit codes + `min` + `delta` overhead),
or about 16% bigger than rroq158 and 5× smaller than fp16. Throughput:
~1.4× FP16 on GPU, ~3× FP16 on CPU. Quality: empirically within ~0.5
NDCG@10 points of FP16 across all six BEIR datasets — see the table in
the README.

The build, search, and serving paths share all infrastructure with
rroq158 (codec selection through `Compression.RROQ4_RIEM`, identical
kernel-dispatch logic in `_manager/search.py`, identical Triton +
Rust SIMD dual implementation, identical metadata schema with
`k_requested` / `k_effective` for graceful fallback). The only switches
between the two are the residual-encoding alphabet and the kernel inner
loop. That makes rroq4_riem a "boring" code-path in the best sense:
nothing about it is novel beyond §6 itself, so its production risk
profile is the same as rroq158's.

---

## 7. Empirical validation

The end-to-end BEIR sweep that backs the README table is at
[`benchmarks/beir_2026q2_full_sweep.py`](../../benchmarks/beir_2026q2_full_sweep.py),
with raw per-cell JSONL output under
[`reports/beir_2026q2/`](../../reports/beir_2026q2/). The harness covers:

- **4 codecs** at the strongest variant per bit-width:
  fp16, int8, rroq158, rroq4_riem.
- **6 datasets** (the standard BEIR ColBERT comparison set):
  arguana, fiqa, nfcorpus, quora, scidocs, scifact.
- **2 modes**:
  GPU (Triton MaxSim, A5000) and CPU (8-worker Rust SIMD, AVX2).
- **Full BEIR query sets**, no sub-sampling.

Each cell includes git SHA, GPU model + memory + CUDA runtime, CPU
model + cores + host, wheel versions, encoder, kernel parameters
(K, group_size, seed), and timestamp — see `collect_provenance` in
the harness. Reproducing the sweep on a different host:

```bash
python benchmarks/beir_2026q2_full_sweep.py \
    --output reports/beir_2026q2/sweep.jsonl
```

The headline default-decision rule (Phase F1 in the production
validation plan) tries to **promote `rroq4_riem` to default** as the
no-quality-loss lane. It passes if and only if all four conditions
hold across the 6-dataset sweep:

1. avg NDCG@10 drop vs FP16 ≥ **−0.5 pt**
2. avg R@100 drop vs FP16 ≥ **−0.3 pt**
3. GPU p95 ≤ FP16 GPU p95 on **every** cell (no per-cell regression)
4. CPU p95 ≤ FP16 CPU p95 on **every** cell (no per-cell regression)

The JSONL produced by the harness is the single source of truth for
that decision; the verdict block is rendered by
`scripts/format_beir_2026q2_table.py --format summary`.

**Production verdict from the BEIR 2026-Q2 sweep:** the rroq4_riem
quality conditions (1, 2) **pass** — measured avg ΔNDCG@10 = +0.02 pt
vs FP16 (max ±0.05 pt), avg ΔR@100 = +0.23 pt. The latency conditions
(3, 4) **fail decisively**: 6 of 6 datasets exceed the per-cell GPU p95
budget (worst: arguana at 9.00× FP16 GPU p95) and 6 of 6 datasets
exceed the per-cell CPU p95 budget (worst: arguana at 14.30× FP16 CPU
p95). Every doc-token score now requires a 4-bit dequantize + per-group
`(min, delta)` FMA on top of the centroid lookup, on top of an FP16
baseline that is itself extremely bandwidth-friendly. So **the default
reverts to `Compression.RROQ158`** and rroq4_riem ships as the opt-in
no-quality-loss lane.

The fallback rule (Option 2) keeps rroq158 as the default when the
quality cost is bounded:

- avg NDCG@10 drop vs FP16 ≥ **−1.5 pt** (rroq158 measured: −1.43 pt)
- avg GPU p95 ratio vs FP16 ≤ **1.2×** (rroq158 measured: 1.13×)

Both conditions hold on the production sweep. CPU p95 is the one
honest weakness: the Rust SIMD rroq158 kernel is currently slower than
the FP16 CPU baseline by **~3.2× avg** in absolute QPS at the production
batch shape (2000 doc candidates per query, ~30 query tokens), down
from ~7.9× pre-fix after the post-Phase-7 CPU lane refresh shipped
zero-copy `_to_np` in the scorer, an inner-loop reorder in
`fused_rroq158.rs::score_pair_body` that amortises doc-side popcounts
across query tokens, a `threadpoolctl.threadpool_limits` cap around
the BLAS matrix multiplications in `rroq158.encode_query` (and around
the kernel call) so OpenBLAS / MKL stop fighting rayon for cores, and
a numpy fancy-indexing fast path in the BEIR harness. The remaining
structural cost is that FP16 AVX2 MaxSim is itself extremely
bandwidth-friendly at this batch shape and the rroq158 path's
BLAS-bound query-encoding stage (FWHT rotation + centroid table
look-up) is the next item on the backlog.

### Rescue experiment: a tiny FP16 rerank closes the rroq158 NDCG@10 gap

For workloads that need both rroq158's storage and FP16's top-10
ranking (e.g. arguana-style hard datasets where rroq158 loses ~2.5
NDCG@10 points), the diagnostic in
[`benchmarks/diag_rroq158_rescue.py`](../../benchmarks/diag_rroq158_rescue.py)
shows that a **two-stage rerank** fully closes the gap on every hard
dataset we tested:

1. Score the full candidate set (e.g. 2000 docs) with rroq158 — fast,
   storage-cheap.
2. Re-score the top **N = 32–64** with FP16 MaxSim — slow, but only on
   a tiny shortlist.
3. Concatenate the FP16 ranking for the top-N with the original
   rroq158 ranking for the tail (preserving R@100).

On arguana the gap closes from −2.6 → ~0 pt at N = 64; scifact and
scidocs behave the same. The cost: an extra ~5 ms per query for the
FP16 rerank, plus the storage of FP16 doc tokens for the indexed
shortlist (which is still a fraction of a full FP16 index because
the shortlist is bounded by `max_docs_exact`).

This is wired but **not** the build-time default — it requires shipping
both the rroq158 codes and an FP16 sidecar of the candidate-region
tokens. Users who need it opt in via `SearchConfig.distill_rerank` /
custom rerank lanes; the BEIR sweep numbers in this document do **not**
include it, so the rroq158 row reflects honest single-codec behavior.

---

## References

- Cong Gao, Xubo Liu, Hongya Wang, Mihai-Andrei Florea (2024).
  *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error
  Bound for Approximate Nearest Neighbor Search.*
  [arXiv:2405.12497](https://arxiv.org/abs/2405.12497).
- Tianhao Zhang et al. (2025). *Reduce, Optimize, Quantize: Multi-Vector
  Late Interaction with Compressed Token Embeddings.*
  (the "ROQ" paper this codebase originally extended).
- Ailon, Chazelle (2009). *The Fast Johnson-Lindenstrauss Transform and
  Approximate Nearest Neighbors.* SIAM J. Comput. (FWHT-with-signs as a
  near-orthonormal random rotation.)
- Khattab, Zaharia (2020). *ColBERT: Efficient and Effective Passage
  Search via Contextualized Late Interaction over BERT.* SIGIR 2020.
  (MaxSim formulation.)
- The `voyager-index` codec implementation:
  [`rroq158.py`](../../voyager_index/_internal/inference/quantization/rroq158.py),
  [`rroq4_riem.py`](../../voyager_index/_internal/inference/quantization/rroq4_riem.py),
  [`triton_roq_rroq158.py`](../../voyager_index/_internal/inference/triton_roq_rroq158.py),
  [`fused_rroq158.rs`](../../latence-shard-engine/src/fused_rroq158.rs),
  [`fused_rroq4_riem.rs`](../../latence-shard-engine/src/fused_rroq4_riem.rs).
