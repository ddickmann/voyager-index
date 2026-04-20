# rroq158 Pareto compression probe

- **Dataset**: `arguana` (n_queries=300, n_docs=8,674, dim=128)
- **Mode**: CPU-8w (8 native Rust workers, n_eval=300)
- **Baseline cell**: `K=8192, group_size=32` → 46.0 B/tok, 2.875 bits/coord, NDCG@10=0.3713, p95=666ms
- **Pareto criteria**: storage strictly better, NDCG@10 within -0.0050 of baseline, p95 within +10% of baseline

## Per-cell results

| K | gs | B/tok | bits/coord | QPS | p50 ms | p95 ms | NDCG@10 | ΔNDCG@10 | Recall@100 | Pareto |
|---|----|-------|-----------:|----:|-------:|-------:|--------:|---------:|-----------:|--------|
| 8192 | 128 | 40.0 | 2.500 | 15.9 | 499 | 530 | 0.3655 | -0.0058 | 0.9633 | no |
| 4096 | 128 | 40.0 | 2.500 | 16.1 | 498 | 538 | 0.3626 | -0.0088 | 0.9467 | no |
| 1024 | 128 | 40.0 | 2.500 | 16.4 | 487 | 520 | 0.3524 | -0.0189 | 0.9567 | no |
| 8192 | 64 | 42.0 | 2.625 | 15.7 | 502 | 555 | 0.3726 | +0.0013 | 0.9633 | **YES** |
| 4096 | 64 | 42.0 | 2.625 | 15.9 | 500 | 538 | 0.3670 | -0.0043 | 0.9500 | **YES** |
| 1024 | 64 | 42.0 | 2.625 | 15.9 | 499 | 527 | 0.3429 | -0.0284 | 0.9533 | no |
| 8192 | 32 | 46.0 | 2.875 | 13.8 | 565 | 666 | 0.3713 | +0.0000 | 0.9633 | _baseline_ |
| 4096 | 32 | 46.0 | 2.875 | 14.8 | 530 | 571 | 0.3620 | -0.0093 | 0.9533 | no |
| 1024 | 32 | 46.0 | 2.875 | 13.8 | 579 | 620 | 0.3608 | -0.0105 | 0.9600 | no |

## Verdict

**Pareto-dominant config found**: `K=8192, group_size=64`.

- Storage: 42.0 B/tok (2.625 bits/coord) vs. baseline 46.0 B/tok → **8.7% smaller**.
- Quality: NDCG@10=0.3726 vs. baseline 0.3713 (Δ=+0.0013, within tolerance -0.0050).
- Latency: p95=555ms vs. baseline 666ms (ratio=0.83×).

_All 2 Pareto-dominant cells (storage ASC, NDCG DESC):_

| K | gs | B/tok | NDCG@10 | ΔNDCG | p95 ms | notes |
|---|----|------:|--------:|------:|-------:|-------|
| 8192 | 64 | 42.0 | 0.3726 | +0.0013 | 555 | **recommended** |
| 4096 | 64 | 42.0 | 0.3670 | -0.0043 | 538 | alternate |

**Recommended next step**: validate this config on the remaining 5 BEIR datasets (`fiqa, hotpotqa, nfcorpus, scifact, scidocs`); if quality holds within tolerance across the suite, open a follow-up PR to flip the production default in `voyager_index/_internal/inference/quantization/rroq158.py:Rroq158Config`.

## Beyond the uniform floor

The 9-cell uniform sweep can only ever buy us ~13% storage on dim=128, because
the dense ternary sign+nz pair (32 of the 46 baseline B/tok = 70%) is fixed
regardless of `(K, group_size)`. Going below the 40 B/tok floor (the most
aggressive uniform corner) requires changing the codec itself.

The natural unlock is the **outlier-rescue / sign-only hybrid** pattern (same
trick the user's KV-cache prototype validated at p=0.30 → 1.85 b/d):

* Per-token `regime` byte selects between
    * `rich` (today: sign + nz + scales@gs=32, **46 B/tok**)
    * `cheap` (sign-only + scales@gs=128, **24 B/tok**)
* Outlier selector (recommended: top-p by `sin_norm`) keeps the highest-residual
  fraction `p` at full fidelity.
* Predicted blended density: `B/tok(p) = 25 + 22·p`

| p (rescue) | B/tok | bits/coord | overhead vs. all-cheap |
|---:|---:|---:|---:|
| 0.00 | 25.0 | 1.563 | floor |
| 0.10 | 27.2 | 1.700 | +9% |
| 0.20 | 29.4 | 1.838 | +18% |
| 0.30 | 31.6 | 1.975 | +26% |

That's ~31-46% smaller than today's 46 B/tok at the same quality (assuming the
rescue defends quality, which the KV-cache results suggest it does). Full design
is in [`docs/posts/rroq158-outlier-rescue-design.md`](../docs/posts/rroq158-outlier-rescue-design.md). The recommended de-risking
path is **Phase A** in that doc: a Python-only score-merge prototype that
validates ranking quality before any kernel work.
