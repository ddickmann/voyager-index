# rroq158 Pareto compression probe

- **Dataset**: `fiqa` (n_queries=648, n_docs=57,638, dim=128)
- **Mode**: CPU-8w (8 native Rust workers, n_eval=full)
- **Baseline cell**: `K=8192, group_size=32` → 46.0 B/tok, 2.875 bits/coord, NDCG@10=0.4223, p95=279ms
- **Pareto criteria**: storage strictly better, NDCG@10 within -0.0050 of baseline, p95 within +10% of baseline

## Per-cell results

| K | gs | B/tok | bits/coord | QPS | p50 ms | p95 ms | NDCG@10 | ΔNDCG@10 | Recall@100 | Pareto |
|---|----|-------|-----------:|----:|-------:|-------:|--------:|---------:|-----------:|--------|
| 8192 | 128 | 40.0 | 2.500 | 40.4 | 197 | 253 | 0.4260 | +0.0037 | 0.7118 | **YES** |
| 8192 | 64 | 42.0 | 2.625 | 40.8 | 195 | 255 | 0.4219 | -0.0004 | 0.7164 | **YES** |
| 8192 | 32 | 46.0 | 2.875 | 35.1 | 221 | 279 | 0.4223 | +0.0000 | 0.7151 | _baseline_ |

## Verdict

**Pareto-dominant config found**: `K=8192, group_size=128`.

- Storage: 40.0 B/tok (2.500 bits/coord) vs. baseline 46.0 B/tok → **13.0% smaller**.
- Quality: NDCG@10=0.4260 vs. baseline 0.4223 (Δ=+0.0037, within tolerance -0.0050).
- Latency: p95=253ms vs. baseline 279ms (ratio=0.91×).

_All 2 Pareto-dominant cells (storage ASC, NDCG DESC):_

| K | gs | B/tok | NDCG@10 | ΔNDCG | p95 ms | notes |
|---|----|------:|--------:|------:|-------:|-------|
| 8192 | 128 | 40.0 | 0.4260 | +0.0037 | 253 | **recommended** |
| 8192 | 64 | 42.0 | 0.4219 | -0.0004 | 255 | alternate |

**Recommended next step**: validate this config on the remaining 5 BEIR datasets (`fiqa, hotpotqa, nfcorpus, scifact, scidocs`); if quality holds within tolerance across the suite, open a follow-up PR to flip the production default in `voyager_index/_internal/inference/quantization/rroq158.py:Rroq158Config`.
