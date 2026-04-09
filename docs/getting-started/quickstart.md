# Quickstart

## Installation

```bash
pip install voyager-index
```

For native Rust acceleration (recommended):

```bash
pip install voyager-index[native]
```

## Create an Index

```python
import numpy as np
from voyager_index import Index

idx = Index(
    "my_index",
    dim=128,
    engine="gem",
    seed_batch_size=64,   # train codebook after 64 docs
    n_fine=128,           # fine centroids
    n_coarse=16,          # coarse clusters
)
```

## Add Documents

Each document is a matrix of token embeddings `(n_tokens, dim)`:

```python
n_docs = 100
embeddings = [np.random.randn(32, 128).astype(np.float32) for _ in range(n_docs)]
ids = list(range(n_docs))
payloads = [{"title": f"Document {i}"} for i in range(n_docs)]

idx.add(embeddings, ids=ids, payloads=payloads)
```

## Search

```python
query = np.random.randn(32, 128).astype(np.float32)
results = idx.search(query, k=10)

for r in results:
    print(f"  Doc {r.doc_id}: score={r.score:.4f}, payload={r.payload}")
```

## Update and Delete

```python
idx.update_payload(0, {"title": "Updated Document 0"})
idx.delete([1, 2, 3])
```

## Scroll (Pagination)

```python
page = idx.scroll(limit=20, offset=0)
for r in page.results:
    print(f"  Doc {r.doc_id}: {r.payload}")
if page.next_offset:
    print(f"  Next page at offset {page.next_offset}")
```

## Cleanup

```python
idx.close()
```

Or use a context manager:

```python
with Index("my_index", dim=128, engine="gem", seed_batch_size=64) as idx:
    idx.add(embeddings, ids=ids)
    results = idx.search(query, k=10)
```
