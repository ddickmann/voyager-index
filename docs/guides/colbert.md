# ColBERT Search with colsearch

## Overview

[ColBERT](https://arxiv.org/abs/2004.12832) produces multi-vector document
representations where each token gets its own embedding. `colsearch`
supports these multivector documents through the shard engine, which is the
mainline production path in this repo.

## End-to-End Example

```python
import numpy as np
from colsearch import Index

DIM = 128  # ColBERT default

# 1. Create index
with Index("colbert_index", dim=DIM, engine="shard", n_shards=64, k_candidates=512) as idx:

    # 2. Add documents (each is a (n_tokens, 128) matrix)
    doc_embeddings = [
        np.random.randn(64, DIM).astype(np.float32),   # 64-token doc
        np.random.randn(128, DIM).astype(np.float32),   # 128-token doc
        np.random.randn(32, DIM).astype(np.float32),    # 32-token doc
    ]
    idx.add(
        doc_embeddings,
        ids=[1, 2, 3],
        payloads=[
            {"title": "Introduction to IR"},
            {"title": "Neural Retrieval Survey"},
            {"title": "ColBERT Architecture"},
        ],
    )

    # 3. Search with a query
    query = np.random.randn(32, DIM).astype(np.float32)
    results = idx.search(query, k=3, ef=64)

    for r in results:
        print(f"  {r.doc_id}: {r.score:.4f} — {r.payload['title']}")
```

## With a Real ColBERT Model

```python
from colbert import Searcher

# Encode your query and documents with ColBERT
# Then pass the embeddings to colsearch

query_embs = model.encode_query("What is late interaction?")
doc_embs = [model.encode_doc(doc) for doc in documents]

idx.add(doc_embs, ids=range(len(doc_embs)))
results = idx.search(query_embs, k=10)
```

## Performance Tips

- Start with `engine="shard"` and only tune `n_shards` and `k_candidates` first
- Use GPU scoring when exact-stage latency matters more than deployment simplicity
- Prefer base64 transport in the HTTP API when ColBERT queries or documents get large
