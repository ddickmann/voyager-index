# Quickstart

## Install

Pick the profile you need:

```bash
pip install "voyager-index[shard]"
pip install "voyager-index[shard,gpu]"
pip install "voyager-index[server,shard]"
pip install "voyager-index[server,shard,native]"  # adds Tabu Search solver
```

`engine="shard"` is the mainline production path. `gem` and `hnsw` remain
compatibility backends, but they are not the default user journey in these docs.

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
        compression="fp16",
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

## What To Tune First

- `n_shards`: controls shard granularity
- `k_candidates`: router frontier before exact scoring
- `compression`: stored representation such as `fp16`, `int8`, or `roq4`
- `quantization_mode`: active GPU scoring mode such as `int8`, `fp8`, or `roq4`
- `transfer_mode`: CPU->GPU fetch strategy for streamed GPU scoring

## Next Steps

- [Python API Reference](../api/python.md)
- [Reference API Tutorial](../reference_api_tutorial.md)
- [Shard Engine Guide](../guides/shard-engine.md)
- [Max-Performance Reference API Guide](../guides/max-performance-reference-api.md)
