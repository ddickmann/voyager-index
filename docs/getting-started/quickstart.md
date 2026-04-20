# Quickstart

## Install

Pick the profile you need:

```bash
pip install "voyager-index[full]"
pip install "voyager-index[full,gpu]"
pip install "voyager-index[shard]"
pip install "voyager-index[shard,shard-native]"
pip install "voyager-index[server,shard,solver]"        # adds Tabu Search solver
pip install "voyager-index[server,shard,latence-graph]"  # adds the optional Latence graph lane
```

`engine="shard"` is the mainline production path. `gem` and `hnsw` remain
compatibility backends, but they are not the default user journey in these docs.
The Latence graph sidecar is optional and additive; it is not required for the
base retrieval flow.

## Minimal Local Index

Each document is a matrix of token or patch embeddings shaped
`(n_tokens, dim)`.

```python
import numpy as np

from voyager_index import Index

rng = np.random.default_rng(7)
docs = [rng.normal(size=(16, 128)).astype("float32") for _ in range(64)]
query = rng.normal(size=(16, 128)).astype("float32")

idx = Index(
    "quickstart-index",
    dim=128,
    engine="shard",
    n_shards=32,
    k_candidates=256,
    # The new production default is `compression="rroq158"` (Riemannian
    # 1.58-bit, K=8192) on both GPU (Triton) and CPU (Rust SIMD). It needs
    # at least K tokens (8192 by default) to train the codebook, so the
    # 64-doc demo here uses fp16. For real corpora drop the `compression`
    # argument to pick up the default — see `docs/api/python.md` for the
    # full knob set (`rroq158_k`, `rroq158_group_size`, ...).
    compression="fp16",
)
idx.add(
    docs,
    ids=list(range(len(docs))),
    payloads=[{"title": f"Document {i}", "tenant": "demo"} for i in range(len(docs))],
)

results = idx.search(query, k=5)
print(results[0])
idx.close()
```

## Builder Form

```python
from voyager_index import IndexBuilder

idx = (
    IndexBuilder("builder-index", dim=128)
    .with_shard(
        n_shards=64,
        k_candidates=512,
        # compression defaults to "rroq158" (K=8192). Override with "fp16" /
        # "int8" / "roq4" if required.
        quantization_mode="fp8",
        transfer_mode="pinned",
    )
    .with_wal(enabled=True)
    .build()
)
```

## CRUD And Recovery

```python
idx.upsert([docs[0]], ids=[0], payloads=[{"title": "updated"}])
idx.update_payload(0, {"title": "updated", "tenant": "demo"})
idx.delete([1, 2, 3])
idx.flush()
idx.close()

reopened = Index("builder-index", dim=128, engine="shard")
print(reopened.stats())
reopened.close()
```

## Multimodal Start

```python
from voyager_index.preprocessing import enumerate_renderable_documents, render_documents

documents = enumerate_renderable_documents("./docs-to-index")
rendered = render_documents(documents["documents"], "./rendered-pages")
print(rendered["rendered"][0]["image_path"])
```

Supported render inputs: PDF, DOCX, XLSX, PNG, JPG, WebP, and GIF.

## HTTP Start

Start the reference server:

```bash
HOST=0.0.0.0 WORKERS=4 voyager-index-server
```

Use base64 transport for new clients:

```python
import numpy as np
import requests

from voyager_index import encode_vector_payload

query = np.random.default_rng(7).normal(size=(16, 128)).astype("float32")

response = requests.post(
    "http://127.0.0.1:8080/collections/demo/search",
    json={
        "vectors": encode_vector_payload(query, dtype="float16"),
        "top_k": 5,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["results"][0])
```

JSON float arrays still work, but base64 is the preferred/default transport for
large dense and multivector payloads.

## Optional Graph Start

If the Latence graph dependency is installed, the same search contract can
enable the premium graph lane:

```json
{
  "graph_mode": "auto",
  "graph_local_budget": 4,
  "graph_community_budget": 4,
  "graph_evidence_budget": 8,
  "graph_explain": true
}
```

Graph augmentation runs after first-stage retrieval and is merged additively.

## What To Tune First

- `n_shards`: controls shard granularity
- `k_candidates`: router frontier before exact scoring
- `compression`: stored representation. Default is `rroq158` (Riemannian
  1.58-bit, K=8192) on both GPU and CPU; opt-outs include `fp16`, `int8`,
  `roq4`, and `rroq4_riem` (Riemannian 4-bit asymmetric — the safe-fallback
  lane for zero-regression workloads). Existing indexes load against their
  build-time codec via the manifest, so flipping the default is
  non-breaking for deployed clusters.
- `rroq158_k` / `rroq158_seed` / `rroq158_group_size`: tuning knobs for the
  default codec. Defaults are `K=8192`, `seed=42`, **`group_size=128`**
  (SOTA — one scale per token at dim=128, ~13% smaller storage and
  ~10–30% faster CPU p95 vs the previous `32` default; for non-multiple-of-128
  dims the encoder transparently steps down to gs=64 / gs=32 with a log
  warning). See [`docs/guides/quantization-tuning.md`](../guides/quantization-tuning.md).
- `rroq4_riem_k` / `rroq4_riem_seed` / `rroq4_riem_group_size`: tuning knobs
  for the safe-fallback codec when `compression="rroq4_riem"`. Defaults are
  `K=8192`, `seed=42`, `group_size=32`.
- `quantization_mode`: active GPU scoring mode such as `int8`, `fp8`, or `roq4`
- `transfer_mode`: CPU->GPU fetch strategy for streamed GPU scoring

## Next Steps

- [Python API Reference](../api/python.md)
- [Reference API Tutorial](../reference_api_tutorial.md)
- [Shard Engine Guide](../guides/shard-engine.md)
- [Max-Performance Reference API Guide](../guides/max-performance-reference-api.md)
