# Installation

## Recommended Extras

```bash
pip install "voyager-index[shard]"
pip install "voyager-index[shard,gpu]"
pip install "voyager-index[server,shard]"
pip install "voyager-index[server,shard,native]"
```

What they mean:

- `shard`: FAISS + safetensors for the shard retrieval path
- `gpu`: Triton GPU kernels
- `server`: FastAPI reference API plus document rendering dependencies
- `native`: optional `latence_solver` wheel for `tabu` refinement and `/reference/optimize`

## Mainline Install Profiles

| Goal | Install |
|---|---|
| Local shard retrieval on CPU | `pip install "voyager-index[shard]"` |
| Shard retrieval with Triton MaxSim | `pip install "voyager-index[shard,gpu]"` |
| Reference API on CPU | `pip install "voyager-index[server,shard]"` |
| Reference API + solver refinement | `pip install "voyager-index[server,shard,native]"` |

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
