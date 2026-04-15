# Installation

## Recommended Extras

```bash
pip install "voyager-index[full]"
pip install "voyager-index[full,gpu]"
pip install "voyager-index[shard]"
pip install "voyager-index[shard,shard-native]"
pip install "voyager-index[shard,gpu]"
pip install "voyager-index[server,shard]"
pip install "voyager-index[server,shard,solver]"
pip install "voyager-index[server,shard,native]"
pip install "voyager-index[server,shard,latence-graph]"
```

What they mean:

- `full`: the full public CPU install profile: server, shard, multimodal helpers, preprocessing stack, supported native wheels, and the graph SDK
- `shard`: FAISS + safetensors for the shard retrieval path
- `shard-native`: optional `latence_shard_engine` wheel for the fused Rust CPU shard path
- `gpu`: Triton GPU kernels
- `server`: FastAPI reference API plus document rendering dependencies
- `solver`: optional `latence_solver` wheel for `tabu` refinement and `/reference/optimize`
- `native`: both supported native wheels: `latence_shard_engine` plus `latence_solver`
- `latence-graph`: optional LatenceAI SDK dependency for the premium graph-aware retrieval lane and compatible prebuilt graph data

## Mainline Install Profiles

| Goal | Install |
|---|---|
| Full public CPU surface | `pip install "voyager-index[full]"` |
| Full public surface + Triton GPU | `pip install "voyager-index[full,gpu]"` |
| Local shard retrieval on CPU | `pip install "voyager-index[shard]"` |
| Local shard retrieval + Rust CPU fast-path | `pip install "voyager-index[shard,shard-native]"` |
| Shard retrieval with Triton MaxSim | `pip install "voyager-index[shard,gpu]"` |
| Reference API on CPU | `pip install "voyager-index[server,shard]"` |
| Reference API + solver refinement | `pip install "voyager-index[server,shard,solver]"` |
| Reference API + both public native wheels | `pip install "voyager-index[server,shard,native]"` |
| Reference API + optional Latence graph lane | `pip install "voyager-index[server,shard,latence-graph]"` |
| Reference API + solver + optional graph lane | `pip install "voyager-index[server,shard,native,latence-graph]"` |

## From Source

Requires Python 3.10+ and a Rust toolchain only if you want to build the public
native wheels from source.

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

## Supported Native Packages

The supported native packages in the public PyPI story are:

| Package | Purpose |
|---|---|
| `latence-shard-engine` | Rust shard CPU fast-path used by the shard production lane when installed |
| `latence-solver` | Tabu Search solver for `dense_hybrid_mode="tabu"` and `/reference/optimize` |

Published prebuilt native wheels target:

- Linux `x86_64`
- macOS `x86_64`
- macOS `arm64`

On other platforms, keep the same extras and build the native packages from
source with a Rust toolchain.

The optional premium graph lane is Python-only from the `voyager-index` side:

| Package | Purpose |
|---|---|
| `latence` via `voyager-index[latence-graph]` | LatenceAI Dataset Intelligence and graph sidecar integration |

If the graph dependency is missing or unavailable, the runtime falls back to the
OSS retrieval path and reports the graph lane as unavailable or skipped.

Build them directly if needed:

```bash
python -m pip install ./src/kernels/shard_engine
python -m pip install ./src/kernels/knapsack_solver
```

## Verify

```python
import voyager_index
from voyager_index import Index, encode_vector_payload

print(voyager_index.__version__)
print(Index, encode_vector_payload)
```
