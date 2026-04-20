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

**Outcome (shipped)**: validated on fiqa + nfcorpus at full eval (both pass cleanly), plus the wider 2026-Q2 sweep gs=32 baseline for the remaining BEIR-6 datasets. Default flipped to `group_size=128` (one scale per token at dim=128) with a dim-aware fallback to gs=64 / gs=32 for non-multiple-of-128 dims. See [`docs/guides/quantization-tuning.md`](../docs/guides/quantization-tuning.md) for the per-dim recipe and the closing retrospective on the outlier-rescue investigation.
