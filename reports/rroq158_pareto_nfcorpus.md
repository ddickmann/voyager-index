# rroq158 Pareto compression probe

- **Dataset**: `nfcorpus` (n_queries=323, n_docs=3,633, dim=128)
- **Mode**: CPU-8w (8 native Rust workers, n_eval=full)
- **Baseline cell**: `K=8192, group_size=32` → 46.0 B/tok, 2.875 bits/coord, NDCG@10=0.3799, p95=286ms
- **Pareto criteria**: storage strictly better, NDCG@10 within -0.0050 of baseline, p95 within +10% of baseline

## Per-cell results

| K | gs | B/tok | bits/coord | QPS | p50 ms | p95 ms | NDCG@10 | ΔNDCG@10 | Recall@100 | Pareto |
|---|----|-------|-----------:|----:|-------:|-------:|--------:|---------:|-----------:|--------|
| 8192 | 128 | 40.0 | 2.500 | 57.8 | 121 | 193 | 0.3790 | -0.0009 | 0.3329 | **YES** |
| 8192 | 64 | 42.0 | 2.625 | 44.6 | 179 | 233 | 0.3796 | -0.0003 | 0.3318 | **YES** |
| 8192 | 32 | 46.0 | 2.875 | 40.9 | 187 | 286 | 0.3799 | +0.0000 | 0.3354 | _baseline_ |

## Verdict

**Pareto-dominant config found**: `K=8192, group_size=128`.

- Storage: 40.0 B/tok (2.500 bits/coord) vs. baseline 46.0 B/tok → **13.0% smaller**.
- Quality: NDCG@10=0.3790 vs. baseline 0.3799 (Δ=-0.0009, within tolerance -0.0050).
- Latency: p95=193ms vs. baseline 286ms (ratio=0.67×).

_All 2 Pareto-dominant cells (storage ASC, NDCG DESC):_

| K | gs | B/tok | NDCG@10 | ΔNDCG | p95 ms | notes |
|---|----|------:|--------:|------:|-------:|-------|
| 8192 | 128 | 40.0 | 0.3790 | -0.0009 | 193 | **recommended** |
| 8192 | 64 | 42.0 | 0.3796 | -0.0003 | 233 | alternate |

**Recommended next step**: validate this config on the remaining 5 BEIR datasets (`fiqa, hotpotqa, nfcorpus, scifact, scidocs`); if quality holds within tolerance across the suite, open a follow-up PR to flip the production default in `voyager_index/_internal/inference/quantization/rroq158.py:Rroq158Config`.
