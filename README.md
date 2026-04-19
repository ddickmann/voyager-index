# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Late-interaction retrieval for on-prem AI. One node. CPU or GPU. MaxSim is the truth scorer.**

> **What changed in this release:** the default codec for newly built indexes
> is now `Compression.RROQ158` — Riemannian-aware 1.58-bit ROQ at K=8192 —
> on **both** GPU (Triton fused kernel) and CPU (Rust SIMD kernel with
> hardware popcount + cached rayon thread pool). At the kernel level it is
> **5.8× faster than the legacy fp16 lane at p95 in the production 8-worker
> CPU layout** while using ~5.5× less disk than fp16. Existing fp16 indexes
> continue to load — the manifest carries the build-time codec; only newly
> built indexes pick up the new default. Pass `compression=Compression.FP16`
> to opt out. Methodology + per-dataset numbers in
> [research/low_bit_roq/PROGRESS.md](research/low_bit_roq/PROGRESS.md)
> (`[2026-04-19] phase-1.5-cpu-kernel-perf-pass`).

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
- **RROQ-1.58 default.** Riemannian 1.58-bit codec — strictly faster than fp16 on both lanes at ~5.5× smaller storage. FP16 / INT8 / FP8 / ROQ-4 all available, all reranked back to float truth.
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
- **Scoring** — Triton MaxSim and fused Rust MaxSim, RROQ-1.58 (default) / INT8 / FP8 / ROQ-4 with float rerank.
- **Storage** — safetensors shards, memory-mapped CPU, GPU-resident corpus mode.
- **Hybrid** — BM25 + dense fusion via RRF or Tabu Search refinement.
- **Multimodal** — text (ColBERT), images (ColPali / ColQwen), preprocessing for PDF / DOCX / XLSX.
- **Operations** — WAL, checkpoint, crash recovery, scroll, retrieve, multi-worker FastAPI.
- **Optional graph lane** — Latence sidecar for graph-aware rescue and provenance, additive to the OSS path.
- **Optional groundedness lane** — **Latence Trace** premium sidecar for post-generation hallucination scoring against retrieved `chunk_ids` or raw context. Calibrated `green`/`amber`/`red` risk band, NLI peer with cross-encoder premise reranking, atomic-claim decomposition, retrieval-coverage observability, response chunking, multilingual EN+DE, three Pareto-optimal profiles, ~118 ms p95 end-to-end with NLI on. Commercial license; runs as a separate process, additive to the OSS retrieval path. See the [Groundedness sidecar guide](docs/guides/groundedness-sidecar.md) and [latence.ai](https://latence.ai) for access.

## Benchmarks

### BEIR retrieval — RTX A5000, search-only, full query set

Encoder: `lightonai/GTE-ModernColBERT-v1`, codec: `Compression.RROQ158`
(K=8192 — the new default). CPU lane uses 8 native Rust workers.

| Dataset  | Docs   | NDCG@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|----------|-------:|--------:|-----------:|--------:|-------------:|--------:|-------------:|
| arguana  | 8,674  | 0.3679  | 0.9586     | 270.0   | 4.1          | 41.6    | 202.7        |
| fiqa     | 57,638 | 0.4436  | 0.7297     | 164.8   | 5.0          | 80.2    | 115.7        |
| nfcorpus | 3,633  | 0.3833  | 0.3348     | 282.6   | 3.8          | 123.3   | 84.4         |
| quora    | 15,675 | 0.9766  | 0.9993     | 346.8   | 2.6          | 271.7   | 46.9         |
| scidocs  | 25,657 | 0.1977  | 0.4369     | 246.8   | 4.3          | 83.9    | 111.8        |
| scifact  | 5,183  | 0.7544  | 0.9567     | 263.4   | 4.0          | 69.1    | 138.4        |

GPU P95 stays under 6 ms across every dataset. The full per-dataset
[head-to-head against `next-plaid`](docs/benchmarks.md#comparison-vs-next-plaid)
(same model, H100, encoding included), methodology, and caveats live in
[docs/benchmarks.md](docs/benchmarks.md).

#### New default: RROQ-1.58 (Riemannian 1.58-bit ternary, K=8192)

`Compression.RROQ158` is now the **default codec for newly built
indexes** on both GPU (Triton fused kernel) and CPU (Rust SIMD kernel).
Per-token storage drops to **46 B** (vs 256 B FP16, 64 B ROQ-4 — i.e.
**5.5× / 1.4× smaller**), and both lanes are wired and tested.

CPU lane microbench at production K=8192 (8 native workers × 16 threads,
which is the production server layout):

| codec   | p50         | p95         | docs·s⁻¹     |
| ------- | ----------: | ----------: | -----------: |
| fp16    | 199.8 ms    | 498.1 ms    | baseline     |
| rroq158 | **14.1 ms** | **86.3 ms** | **5.8× faster p95** |

GPU lane: fused two-stage Triton kernel — **0.15 ms p50 / 3.4 M docs·s⁻¹**
at the 32×32×512 microbench, parity ≤ 1e-4 vs the python reference. CPU
lane: Rust SIMD kernel (`latence_shard_engine.rroq158_score_batch`) with
hardware `popcnt` + AVX2/BMI2/FMA + cached rayon thread pool, bitwise
parity to rtol=1e-4 vs the python reference (validated by
`tests/test_rroq158_kernel.py::test_rroq158_rust_simd_matches_python_reference`).

Backwards compatibility: existing FP16 indexes load unchanged — the
manifest carries the build-time codec. Pass
`compression=Compression.FP16` (Python) or `--compression fp16` (CLI) to
opt out of the new default. Full audit and per-phase verdicts in
[research/low_bit_roq/PROGRESS.md](research/low_bit_roq/PROGRESS.md)
(`[2026-04-19] phase-2-to-6-rroq158-default` and
`[2026-04-19] phase-1.5-cpu-kernel-perf-pass`).

## Architecture

```text
query (token / patch embeddings)
  → LEMUR routing MLP → FAISS ANN → candidate IDs
  → optional BM25 fusion · centroid pruning · ColBANDIT
  → exact MaxSim   (Rust SIMD CPU FP16/RROQ-1.58  |  Triton FP16/INT8/FP8/ROQ-4/RROQ-1.58 GPU)
  → optional Latence graph augmentation
  → top-K (or packed context)
```

| Layer        | What ships                                                       |
|--------------|------------------------------------------------------------------|
| Routing      | LEMUR MLP + FAISS MIPS, candidate budgets                        |
| Storage      | safetensors shards, mmap, GPU-resident corpus mode               |
| Scoring      | Triton + Rust fused MaxSim with RROQ-1.58 (default) / INT8 / FP8 / ROQ-4 fast paths |
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

## License

Apache-2.0. See [LICENSE](LICENSE).
