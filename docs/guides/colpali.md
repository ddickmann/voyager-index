# ColPali Multimodal Search with voyager-index

## Overview

[ColPali](https://arxiv.org/abs/2407.01449) extends the late-interaction
paradigm to multimodal retrieval: document pages are encoded as grids of patch
embeddings. voyager-index indexes these multi-vector representations natively.

## Example

```python
import numpy as np
from voyager_index import Index

DIM = 128  # ColPali embedding dimension

with Index("colpali_index", dim=DIM, engine="gem", seed_batch_size=16) as idx:

    # Each page produces a (n_patches, 128) matrix
    # e.g., a 1024-patch ViT output
    page_embeddings = [
        np.random.randn(1024, DIM).astype(np.float32),  # page 1
        np.random.randn(1024, DIM).astype(np.float32),  # page 2
        np.random.randn(1024, DIM).astype(np.float32),  # page 3
    ]

    idx.add(
        page_embeddings,
        ids=[1, 2, 3],
        payloads=[
            {"filename": "report.pdf", "page": 1},
            {"filename": "report.pdf", "page": 2},
            {"filename": "slides.pdf", "page": 1},
        ],
    )

    # Query with text embeddings
    query = np.random.randn(32, DIM).astype(np.float32)
    results = idx.search(query, k=3)

    for r in results:
        print(f"  Page {r.doc_id}: score={r.score:.4f}, file={r.payload['filename']}")
```

## Why GEM for ColPali

ColPali pages produce 1024+ patch embeddings per document. Traditional HNSW
indexes each patch separately, requiring per-patch search + aggregation.
GEM's proximity graph operates at the document level, making it **52x faster**
at 1024-patch sequences.

## Payload Filtering

Filter results by document metadata:

```python
results = idx.search(
    query, k=10,
    filters={"filename": {"$eq": "report.pdf"}}
)
```
