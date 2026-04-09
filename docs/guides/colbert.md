# ColBERT Search with voyager-index

## Overview

[ColBERT](https://arxiv.org/abs/2004.12832) produces multi-vector document
representations where each token gets its own embedding. voyager-index's GEM
engine natively indexes these multi-vector documents.

## End-to-End Example

```python
import numpy as np
from voyager_index import Index

DIM = 128  # ColBERT default

# 1. Create index
with Index("colbert_index", dim=DIM, engine="gem", seed_batch_size=32) as idx:

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
# Then pass the embeddings to voyager-index

query_embs = model.encode_query("What is late interaction?")
doc_embs = [model.encode_doc(doc) for doc in documents]

idx.add(doc_embs, ids=range(len(doc_embs)))
results = idx.search(query_embs, k=10)
```

## Performance Tips

- Use `seed_batch_size=64` for small collections (< 1000 docs)
- Increase `n_fine` for longer documents (512+ tokens)
- Set `ef=128` or higher for high-recall applications
