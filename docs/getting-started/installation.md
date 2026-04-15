# Installation

## Recommended Extras

```bash
pip install "voyager-index[shard]"
pip install "voyager-index[shard,gpu]"
pip install "voyager-index[server,shard]"
pip install "voyager-index[server,shard,native]"
pip install "voyager-index[server,shard,latence-graph]"
```

What they mean:

- `shard`: FAISS + safetensors for the shard retrieval path
- `gpu`: Triton GPU kernels
- `server`: FastAPI reference API plus document rendering dependencies
- `native`: optional `latence_solver` wheel for `tabu` refinement and `/reference/optimize`
- `latence-graph`: optional LatenceAI SDK dependency for the premium graph-aware retrieval lane

## Mainline Install Profiles

| Goal | Install |
|---|---|
| Local shard retrieval on CPU | `pip install "voyager-index[shard]"` |
| Shard retrieval with Triton MaxSim | `pip install "voyager-index[shard,gpu]"` |
| Reference API on CPU | `pip install "voyager-index[server,shard]"` |
| Reference API + solver refinement | `pip install "voyager-index[server,shard,native]"` |
| Reference API + optional Latence graph lane | `pip install "voyager-index[server,shard,latence-graph]"` |
| Reference API + solver + optional graph lane | `pip install "voyager-index[server,shard,native,latence-graph]"` |

## From Source

Requires Python 3.10+ and a Rust toolchain only if you want to build the solver
wheel from source.

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

## Optional Native Package

The only native package in the supported PyPI story today is:

| Package | Purpose |
|---|---|
| `latence-solver` | Tabu Search solver for `dense_hybrid_mode="tabu"` and `/reference/optimize` |

The optional premium graph lane is Python-only from the `voyager-index` side:

| Package | Purpose |
|---|---|
| `latence` via `voyager-index[latence-graph]` | LatenceAI Dataset Intelligence and graph sidecar integration |

If the graph dependency is missing or unavailable, the runtime falls back to the
OSS retrieval path and reports the graph lane as unavailable or skipped.

Build it directly if needed:

```bash
python -m pip install ./src/kernels/knapsack_solver
```

## Verify

```python
import voyager_index
from voyager_index import Index, encode_vector_payload

print(voyager_index.__version__)
print(Index, encode_vector_payload)
```
