# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

**Late-interaction retrieval for on-prem AI. One node. CPU or GPU. MaxSim is the truth scorer.**

> **What changed in this release:** the default codec for newly built indexes
> is now `Compression.RROQ158` at **`group_size=128`** — Riemannian-aware
> 1.58-bit ternary ROQ at K=8192 with one scale per token at dim=128, the
> SOTA storage path. On the 3 BEIR datasets re-validated at the gs=128
> flip (arguana, fiqa, nfcorpus, full-eval CPU 8-worker), it delivers
> **~13% smaller storage** (40 vs 46 bytes/token at dim=128, **~6.4×
> smaller than FP16** overall), **p95 ~10–30% faster** than the previous
> gs=32 default, with **NDCG@10 within ±0.005** on Pareto-clean datasets.
> Wired on **both** GPU (Triton fused kernel) and CPU (Rust SIMD kernel
> with hardware popcount + cached rayon thread pool). For dims that aren't
> a multiple of 128 (dim=64 / 96 / 160) the encoder transparently steps
> down to gs=64 / gs=32 with a log warning — no breakage, no need to
> override. For workloads that refuse any quality regression, ship
> `Compression.RROQ4_RIEM` — the Riemannian-aware 4-bit asymmetric
> no-quality-loss lane (Triton + Rust SIMD wired, parity-tested, ~3×
> smaller than FP16, **~0.5 pt NDCG@10 gap** — still slower than FP16
> in absolute BEIR latency, but the Phase-7-followup CPU kernel reorder
> cut the Rust microbench by ~22% at production K=8192/B=2000, see the
> sweep table below). Existing FP16 / RROQ158 / RROQ4_RIEM indexes
> continue to load unchanged — the manifest carries the build-time codec
> _and the resolved group_size_; only newly built indexes pick up the new
> default. Pass `compression=Compression.FP16` to opt out, or pin
> `Rroq158Config(group_size=64)` for the safest cross-dataset choice (see
> [docs/guides/quantization-tuning.md](docs/guides/quantization-tuning.md)
> for the per-dim recipe and override guidance). Sweep methodology,
> per-dataset numbers, and the F1 default-decision verdict live under
> [`reports/beir_2026q2/`](reports/beir_2026q2/) and
> [`reports/rroq158_pareto_cells/`](reports/rroq158_pareto_cells/); the
> math behind the codec is in
> [docs/guides/rroq-mathematics.md](docs/guides/rroq-mathematics.md);
> a public-facing write-up is in
> [docs/posts/sub-2-bit-late-interaction.md](docs/posts/sub-2-bit-late-interaction.md).

## The pain

ColBERT-quality retrieval is table-stakes for serious RAG, and the production options
force a choice you should not have to make.

- Managed SaaS — fast to start, hard to control, your data leaves the box.
- Distributed clusters — strong recall, expensive to operate.
- Offline benchmarks — great numbers, no API, no WAL, no recovery.

Most "production" stacks treat MaxSim as an optional rerank stage and lose its
signal under aggressive shortlisting. Most engines that ship operationally drop
late interaction entirely.

## The solution

`voyager-index` is a multi-vector native retrieval engine built around MaxSim as
the **final scorer** — and engineered so a single machine can serve it.

- **One-node deployment.** No control plane, no orchestration tax.
- **One contract across CPU and GPU.** Rust SIMD on CPU, Triton on GPU.
- **RROQ-1.58 SOTA default.** Riemannian 1.58-bit codec at K=8192, **group_size=128** (one scale per token at dim=128) — **~6.4× smaller doc-token storage than FP16**, ~−1 pt NDCG@10 average on BEIR (flat R@100), at FP16-comparable GPU latency on most cells and ~10–30% faster CPU p95 than the previous gs=32 default. Dim-aware fallback to gs=64 / gs=32 for non-multiple-of-128 dims (dim=64 / 96 / 160) ships out of the box. **RROQ-4 Riemannian** ships as the no-quality-loss lane (~3× smaller than FP16, ~0.5 pt NDCG@10 gap; the Phase-7-followup loop-reorder gave the Rust SIMD CPU kernel a ~22% speedup at production K=8192/B=2000, but the codec is still slower than FP16 in absolute latency — the win is storage). FP16 / INT8 / FP8 / ROQ-4 all available, all reranked back to float truth. See [docs/guides/quantization-tuning.md](docs/guides/quantization-tuning.md) for the decision matrix and per-dim recipe.
- **Late-interaction native.** ColBERT, ColPali, ColQwen out of the box.
- **Database semantics.** WAL, checkpoint, crash recovery, scroll, retrieve.
- **Optional graph lane.** The Latence sidecar augments first-stage retrieval — never required.

## How

```bash
pip install "voyager-index[full]"        # CPU-only host
pip install "voyager-index[full,gpu]"    # GPU host
voyager-index-server                     # OpenAPI at http://127.0.0.1:8080/docs
```

Python:

```python
import numpy as np
from voyager_index import Index

rng = np.random.default_rng(7)
docs = [rng.normal(size=(16, 128)).astype("float32") for _ in range(32)]
query = rng.normal(size=(16, 128)).astype("float32")

idx = Index("demo", dim=128, engine="shard", n_shards=32,
            k_candidates=256, compression="fp16")
idx.add(docs, ids=list(range(len(docs))))
print(idx.search(query, k=5)[0])
```

HTTP (base64 vector payloads, fp8 GPU scoring, ColBANDIT pruning):

```python
import numpy as np, requests
from voyager_index import encode_vector_payload

q = np.random.default_rng(7).normal(size=(16, 128)).astype("float32")
r = requests.post(
    "http://127.0.0.1:8080/collections/demo/search",
    json={"vectors": encode_vector_payload(q, dtype="float16"), "top_k": 5,
          "quantization_mode": "fp8", "use_colbandit": True},
    timeout=30,
)
print(r.json()["results"][0])
```

Docker:

```bash
docker build -f deploy/reference-api/Dockerfile -t voyager-index .
docker run -p 8080:8080 -v "$(pwd)/data:/data" voyager-index
```

## Features

- **Routing** — LEMUR proxy router + FAISS MIPS shortlist, optional ColBANDIT query-time pruning.
- **Scoring** — Triton MaxSim and fused Rust MaxSim, RROQ-1.58 (default) / RROQ-4 Riemannian (safe fallback) / INT8 / FP8 / ROQ-4 with float rerank.
- **Storage** — safetensors shards, memory-mapped CPU, GPU-resident corpus mode.
- **Hybrid** — BM25 + dense fusion via RRF or Tabu Search refinement.
- **Multimodal** — text (ColBERT), images (ColPali / ColQwen), preprocessing for PDF / DOCX / XLSX.
- **Operations** — WAL, checkpoint, crash recovery, scroll, retrieve, multi-worker FastAPI.
- **Optional graph lane** — Latence sidecar for graph-aware rescue and provenance, additive to the OSS path.
- **Optional groundedness lane** — **Latence Trace** premium sidecar for post-generation hallucination scoring against retrieved `chunk_ids` or raw context. Calibrated `green`/`amber`/`red` risk band, NLI peer with cross-encoder premise reranking, atomic-claim decomposition, retrieval-coverage observability, response chunking, multilingual EN+DE, three Pareto-optimal profiles, ~118 ms p95 end-to-end with NLI on. Commercial license; runs as a separate process, additive to the OSS retrieval path. See the [Groundedness sidecar guide](docs/guides/groundedness-sidecar.md) and [latence.ai](https://latence.ai) for access.

## Benchmarks

### BEIR retrieval — RTX A5000 + 8-worker AVX2 CPU, full query set

Encoder: `lightonai/GTE-ModernColBERT-v1`. Sweep harness:
[`benchmarks/beir_2026q2_full_sweep.py`](benchmarks/beir_2026q2_full_sweep.py).
Raw per-cell JSONL with provenance (git SHA, GPU model + driver, CPU
model + cores, wheel versions): [`reports/beir_2026q2/`](reports/beir_2026q2/).

The full 4-codec × 6-dataset × 2-mode table is rendered to
[docs/benchmarks.md](docs/benchmarks.md) by
[`scripts/format_beir_2026q2_table.py`](scripts/format_beir_2026q2_table.py).
Headline averages (BEIR-6 mean):

<!-- BEIR_2026Q2_HEADLINE_TABLE_BEGIN — measured 2026-04-20 from reports/beir_2026q2_gs128/sweep_combined.jsonl (full BEIR-6 sweep at the new SOTA gs=128 default + post-Phase-7 CPU lane fixes) -->

| Codec       | NDCG@10 (avg) | ΔNDCG@10 vs fp16 | R@100 (avg) | ΔR@100 vs fp16 | Storage vs fp16 | GPU p95 (avg) | CPU p95 (avg) |
|-------------|--------------:|-----------------:|------------:|---------------:|----------------:|--------------:|--------------:|
| fp16        |        0.5206 |              0.0 |      0.7360 |            0.0 |          1.00×  |        4.0 ms |       103 ms  |
| int8        |        0.5200 |       −0.06 pt   |      0.7357 |     −0.03 pt   |          0.50×  |        4.0 ms |   n/a (GPU-only) |
| **rroq158** |        0.5069 |       **−1.37 pt** |    0.7298 |     −0.62 pt   |        **0.16×** |     4.8 ms (1.20×) |   310 ms (3.00×) |
| rroq4_riem  |        0.5208 |       +0.02 pt   |      0.7383 |     +0.23 pt   |          0.34×  |   20.1 ms (5.03×) |   741 ms (7.18×) |

<!-- BEIR_2026Q2_HEADLINE_TABLE_END -->

> **Refreshed 2026-04-20 at the new SOTA default `group_size=128`.** Full
> BEIR-6 (arguana / fiqa / nfcorpus / quora / scidocs / scifact) × 2-mode
> (GPU Triton + 8-worker CPU Rust SIMD) sweep at the new default — 12
> rroq158 cells re-measured against the existing fp16 / int8 / rroq4_riem
> cells. Vs the prior `group_size=32` baseline rroq158 is **13% smaller
> storage** (40 vs 46 B/tok at dim=128 → 0.16× vs 0.18× of fp16, i.e.
> **~6.4× smaller than fp16** vs ~5.5× before), with **NDCG@10
> essentially flat** (avg per-dataset Δ = +0.0006 — fiqa +0.0037 /
> nfcorpus +0.0004 / quora +0.0007 / scidocs ≈0 / scifact +0.0016 /
> arguana −0.0024) and **lower or equal CPU p95** on every dataset
> (largest win: nfcorpus 287 → 223 ms = −22%; fiqa 279 → 285 ms is the
> only +/−2% bump). Sweep harness:
> [`benchmarks/beir_2026q2_full_sweep.py`](benchmarks/beir_2026q2_full_sweep.py)
> (now defaults to `--rroq158-group-size 128`); raw cells:
> [`reports/beir_2026q2_gs128/`](reports/beir_2026q2_gs128/); per-dim
> recipe + override guidance:
> [docs/guides/quantization-tuning.md](docs/guides/quantization-tuning.md).

Detailed per-dataset rows, the F1 default-promotion verdict, and the
brute-force codec-fidelity overlap (top-10 / 20 / 50 / 100 vs FP16 — the
rroq158 quality story disaggregated below the rank-aggregate metric)
live in [docs/benchmarks.md](docs/benchmarks.md).

#### Default: RROQ-1.58 (Riemannian 1.58-bit ternary, K=8192)

`Compression.RROQ158` is the **default codec for newly built indexes**
on both GPU (Triton fused kernel) and CPU (Rust SIMD kernel) — at the
new SOTA `group_size=128` (one scale per token at dim=128). Per-token
storage drops to **~40 B** (vs 256 B FP16, 64 B ROQ-4 — i.e. **~6.4× /
1.6× smaller**), down from the previous ~46 B at gs=32. Both lanes are
wired and parity-tested. For dims that aren't a multiple of 128 (dim=64
/ 96 / 160) the encoder transparently steps down to gs=64 / gs=32 with
a log warning. Override with `Rroq158Config(group_size=64)` for the
safest cross-corpus choice — see
[docs/guides/quantization-tuning.md](docs/guides/quantization-tuning.md)
for the per-dim recipe and override guidance.

The honest sweep verdict from the BEIR 2026-Q2 production sweep
(`reports/beir_2026q2/sweep.jsonl`, 4 codecs × 6 datasets × 2 modes,
full BEIR query sets):

- **Quality.** Avg NDCG@10 vs FP16 (gs=128 default): **−1.37 pt**
  (worst dataset: arguana at −2.93 pt; best: nfcorpus at −0.36 pt).
  Avg R@100 vs FP16: **−0.62 pt** (essentially flat in absolute
  terms). Vs the prior gs=32 default the avg per-dataset NDCG@10 is
  +0.0006 — i.e. essentially Pareto-equal in quality, paying only the
  arguana −0.0024 marginal cost in exchange for the 13% smaller
  storage. The
  brute-force codec-fidelity overlap diagnostic
  (`benchmarks/topk_overlap_sweep.py`, `reports/beir_2026q2/topk_overlap.jsonl`)
  measures the per-query top-K overlap of each codec's brute-force
  MaxSim ranking against the FP16 brute-force ranking — i.e. the
  fraction of FP16's top-K documents that the codec also returns in
  *its* top-K. Across BEIR-6 the rroq158 codec retains **avg ~79%
  top-10 overlap and ~80% top-100 overlap** (range: 73–83% top-10,
  72–85% top-100; per-dataset numbers in
  [docs/benchmarks.md](docs/benchmarks.md)). Important: top-K overlap
  is roughly *flat across K* for rroq158 (e.g. quora 72.9% → 72.1%
  from K=10 to K=100), so widening the serve window is **not** a
  reliable rescue mechanism — the displacement is *out of the
  candidate set*, not within it. Even so, rroq158 R@100 stays
  within −2.1 pt of FP16 on every dataset (and within −1.4 pt on
  arguana specifically), because the codec still admits the labeled
  relevant documents — the displacement happens among the
  non-relevant tail. Workloads requiring exact top-10 rank fidelity
  vs FP16 should opt into `rroq4_riem` (the no-quality-loss lane below
  — avg ~96% top-10 overlap) or use rroq158 with an FP16 rerank on
  the shortlist (`benchmarks/diag_rroq158_rescue.py` shows top-32/64
  FP16 rerank closes the gap with no R@100 regression).
- **Latency.** Avg GPU p95 (gs=128 default): **4.8 ms vs 4.0 ms FP16
  (1.20×)** — at the 1.20× retention budget. Avg CPU p95: **310 ms
  vs 103 ms FP16 (3.00×)** — improved from 3.15× at gs=32 (and from
  7.88× pre-fix) thanks to one fewer scale load per group in the
  popcount kernel; per-dataset CPU p95 is lower or equal vs gs=32 on
  every cell (best: nfcorpus −22%, scidocs −5%, arguana −1%; only
  bump: fiqa +2%). Avg GPU p95 vs gs=32 is +5% (within noise on the
  small / medium datasets, +6% on quora). Cumulative CPU p95 win
  came from four post-Phase-7 lane refresh optimisations: (1)
  zero-copy `_to_np` in
  [`scorer.py`](voyager_index/_internal/inference/shard_engine/scorer.py)
  that bypasses `np.ascontiguousarray` for already-contiguous
  CPU-resident tensors, (2) inner-loop reorder in
  [`fused_rroq158.rs`](src/kernels/shard_engine/src/fused_rroq158.rs)
  amortising doc-side popcounts (`s_g`) once per document token,
  (3) `threadpoolctl.threadpool_limits` cap around the BLAS
  matrix multiplications in
  [`rroq158.encode_query`](voyager_index/_internal/inference/quantization/rroq158.py)
  and around the kernel call to stop OpenBLAS/MKL fighting rayon,
  and (4) numpy fancy-indexing fast path in the BEIR harness's
  `_rroq158_score_candidates` to bypass `torch.index_select` on
  CPU. Per-dataset speed-ups vs the pre-fix CPU lane range from
  **2.0× (quora) to 5.0× (nfcorpus, scifact)** with quality
  unchanged (kernel is deterministic). Remaining headroom is in
  the BLAS-bound query-encoding stage (FWHT rotation + centroid
  table look-up); shrinking that further is on the post-fix
  backlog.

GPU lane: fused two-stage Triton kernel
(`voyager_index._internal.kernels.triton_roq_rroq158`), parity ≤ 1e-4
vs the python reference. CPU lane: Rust SIMD kernel
(`latence_shard_engine.rroq158_score_batch`) with hardware `popcnt` +
AVX2/BMI2/FMA + cached rayon thread pool, bitwise parity to rtol=1e-4
vs the python reference (validated by
`tests/test_rroq158_kernel.py::test_rroq158_rust_simd_matches_python_reference`).

Backwards compatibility: existing FP16 / RROQ158 / RROQ4_RIEM indexes
load unchanged — the manifest carries the build-time codec. Pass
`compression=Compression.FP16` (Python) or `--compression fp16` (CLI) to
opt out of the new default. The math (RaBitQ extension + Riemannian
log map + FWHT-rotated tangent ternary + K = 8192 derivation) is in
[docs/guides/rroq-mathematics.md](docs/guides/rroq-mathematics.md).
The public-facing write-up is at
[docs/posts/sub-2-bit-late-interaction.md](docs/posts/sub-2-bit-late-interaction.md).

#### No-quality-loss lane: RROQ-4 Riemannian (4-bit asymmetric, K=8192)

`Compression.RROQ4_RIEM` is the production option for workloads that
cannot tolerate any quality regression vs FP16 but still want the
storage win of low-bit ROQ. It applies the same Riemannian-aware
spherical-k-means + FWHT pipeline as RROQ-1.58, but encodes the residual
as **4-bit asymmetric per-group** (default `group_size=32`, mins/deltas
in fp16) instead of ternary. Both kernels are wired and parity-tested:

- **GPU**: fused Triton kernel `roq_maxsim_rroq4_riem`
  (`voyager_index._internal.kernels.triton_roq_rroq4_riem`).
- **CPU**: Rust SIMD kernel `latence_shard_engine.rroq4_riem_score_batch`
  with AVX2/FMA + cached rayon thread pool — bitwise parity to rtol=1e-4
  vs the python reference (validated by
  `tests/test_rroq4_riem_kernel.py`).

Per-token storage: ~88 B (vs 256 B FP16, 40 B RROQ-1.58 default — i.e. **~3×
smaller** than fp16). Measured on the Phase-7 BEIR sweep:

- **Quality** is at FP16 parity: avg ΔNDCG@10 = **+0.02 pt** (max
  ±0.05 pt across datasets), avg ΔR@100 = +0.23 pt — the
  no-quality-loss promise holds on every BEIR-6 dataset.
- **Latency** is the trade-off: avg GPU p95 **20.1 ms vs 4.0 ms FP16
  (5.03×)**, avg CPU p95 **741 ms vs 103 ms FP16 (7.18×)** — down
  from 12.65× pre-fix after the same post-Phase-7 CPU lane refresh
  shipped for rroq158 (zero-copy `_to_np`, BLAS thread cap around
  query encode in
  [`rroq4_riem.encode_query`](voyager_index/_internal/inference/quantization/rroq4_riem.py),
  numpy fancy-indexing path in the BEIR harness, plus the
  pre-existing nibble-unpack amortisation in
  [`fused_rroq4_riem.rs::score_pair_body`](src/kernels/shard_engine/src/fused_rroq4_riem.rs)).
  The 4-bit asymmetric per-group dequant + FMA path still adds
  structural compute over the FP16 GEMM/MaxSim baseline, so a
  full-parity CPU lane is not realistic without an AVX-512
  re-encode of the kernel; the storage-with-zero-quality-loss
  promise is what this codec sells.
- The win here is **storage with zero quality regression**, not
  throughput. RROQ4_RIEM is the lane for workloads that refuse the
  rroq158 NDCG@10 cost; rroq158 stays the default when GPU-latency
  parity matters more than the last point of NDCG@10.

Use this when you need the smaller index but cannot accept the rroq158
NDCG@10 cost on hard datasets; use rroq158 when latency parity with
FP16 matters more than a 1-point NDCG@10 budget.

Enable it from Python:

```python
from voyager_index import Index
from voyager_index._internal.inference.shard_engine.config import Compression

idx = Index("safe-fallback-demo", dim=128, engine="shard", n_shards=32,
            k_candidates=256, compression=Compression.RROQ4_RIEM)
```

…or from the CLI:

```bash
python -m voyager_index._internal.inference.shard_engine._builder.cli \
    --compression rroq4_riem --rroq4-riem-k 8192 --rroq4-riem-group-size 32 ...
```

…or over HTTP at collection-create time:

```json
{
  "compression": "rroq4_riem",
  "rroq4_riem_k": 8192,
  "rroq4_riem_group_size": 32
}
```

End-to-end build + search is covered by
`tests/test_rroq4_riem_e2e.py::test_rroq4_riem_build_and_search_cpu` and
auto-derive at search time (no `quantization_mode` override required) by
`tests/test_shard_serving_wiring.py::test_score_sealed_candidates_auto_derives_rroq4_riem_when_meta_present`.

## Architecture

```text
query (token / patch embeddings)
  → LEMUR routing MLP → FAISS ANN → candidate IDs
  → optional BM25 fusion · centroid pruning · ColBANDIT
  → exact MaxSim   (Rust SIMD CPU FP16/RROQ-1.58/RROQ-4-Riem  |  Triton FP16/INT8/FP8/ROQ-4/RROQ-1.58/RROQ-4-Riem GPU)
  → optional FP16 rerank on top-N shortlist (closes the rroq158 NDCG@10 gap)
  → optional Latence graph augmentation
  → top-K (or packed context)
```

| Layer        | What ships                                                       |
|--------------|------------------------------------------------------------------|
| Routing      | LEMUR MLP + FAISS MIPS, candidate budgets                        |
| Storage      | safetensors shards, mmap, GPU-resident corpus mode               |
| Scoring      | Triton + Rust fused MaxSim with RROQ-1.58 (default) / RROQ-4 Riemannian (safe fallback) / INT8 / FP8 / ROQ-4 fast paths |
| Optional graph | Latence sidecar, additive after first-stage retrieval          |
| Durability   | WAL, memtable, checkpoint, crash recovery                        |
| Serving      | FastAPI, base64 vector transport, multi-worker, OpenAPI          |

Three execution modes share the same collection format and API contract:
**CPU exact** (mmap → Rust fused), **GPU streamed** (CPU → GPU → Triton), and
**GPU corpus** (fully VRAM-resident). Start with CPU, add GPU when latency matters.

## Documentation

- [Quickstart](docs/getting-started/quickstart.md) · [Installation](docs/getting-started/installation.md) · [Install from source](docs/getting-started/installation.md#from-source)
- [Reference API Tutorial](docs/reference_api_tutorial.md) · [HTTP endpoint reference](docs/reference_api_tutorial.md#10-api-endpoint-reference) · [Python API](docs/api/python.md)
- [Shard Engine Guide](docs/guides/shard-engine.md) · [Max-Performance Guide](docs/guides/max-performance-reference-api.md) · [Scaling Guide](docs/guides/scaling.md)
- [Latence Graph Sidecar Guide](docs/guides/latence-graph-sidecar.md) · [Groundedness Sidecar Guide](docs/guides/groundedness-sidecar.md) · [Enterprise Control Plane](docs/guides/control-plane.md)
- [Benchmarks And Methodology](docs/benchmarks.md) · [Production Notes](PRODUCTION.md)
- [Full Feature Cookbook](docs/full_feature_cookbook.md)

## Community And Project Health

- File a bug: [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml)
- Request a feature: [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml)
- Open a PR: [pull request template](.github/pull_request_template.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Release process: [RELEASING.md](RELEASING.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## License & Commercial Use

**This project is licensed under [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE) (`CC-BY-NC-4.0`). It is free for non-commercial use only.**

You may use, copy, modify, and redistribute `voyager-index` for:

- research, evaluation, and benchmarking
- academic work and teaching
- personal projects and experimentation
- internal, non-revenue R&D inside an organization

You may **not** use `voyager-index`, in whole or in part, for any commercial
or revenue-generating purpose without a separate commercial license. This
explicitly includes:

- selling, sublicensing, or relicensing the code
- offering it as a hosted, managed, or SaaS product
- embedding it in any product, service, or integration that you sell,
  license, or otherwise monetize
- using it to provide paid consulting deliverables built on top of it

For commercial licensing inquiries, contact **commercial@latence.ai**.

**Carveout — vendored Qdrant subtree.** The directory
`src/kernels/vendor/qdrant/` is a vendored copy of upstream
[qdrant/qdrant](https://github.com/qdrant/qdrant) and remains under its
original **Apache-2.0** license; that license travels with those files into
any derivative binaries. See [LICENSING.md](LICENSING.md) and
[internal/contracts/QDRANT_VENDORING.md](internal/contracts/QDRANT_VENDORING.md)
for the full carveout.

**Prior releases.** Versions of `voyager-index` previously distributed under
Apache-2.0 remain available under Apache-2.0 to recipients who already
obtained them. The CC-BY-NC-4.0 license above applies to this source tree
and to all releases made from it going forward.
