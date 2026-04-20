## BEIR 2026-Q2 codec averages (6-dataset mean)

| Codec | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | GPU P95 (ms) | CPU P95 (ms) | GPU QPS | CPU QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.5206 | 0.5498 | 0.6105 | 0.7360 | 4.0 | 103.1 | 271.2 | 127.1 |
| int8 | 0.5200 | 0.5491 | 0.6106 | 0.7357 | 4.0 | n/a | 277.9 | n/a |
| rroq158 | 0.5069 | 0.5373 | 0.5965 | 0.7298 | 4.8 | 309.6 | 249.9 | 43.4 |
| rroq4_riem | 0.5208 | 0.5505 | 0.6105 | 0.7383 | 20.1 | 740.5 | 119.0 | 24.8 |

## Decision rule 1 — rroq4_riem default promotion (Phase F1)

- avg ΔNDCG@10 vs fp16: **+0.02 pt** (budget ≥ -0.50 pt)
- avg ΔR@100 vs fp16: **+0.23 pt** (budget ≥ -0.30 pt)
- per-cell GPU p95 vs fp16: **6 violation(s)** (budget ≤ 1.00× on every cell)
- per-cell CPU p95 vs fp16: **6 violation(s)** (budget ≤ 1.00× on every cell)

### Verdict: **REVERT default to `Compression.RROQ158`** (ship Option-2 honesty-doc framing)

Failed condition(s):
- 6 cell(s) exceed GPU p95 budget ≤1.00× fp16 [arguana: 9.00× (4.0→36.3ms), fiqa: 4.14× (5.0→20.6ms), nfcorpus: 2.99× (4.1→12.2ms), quora: 1.20× (2.7→3.2ms), scidocs: 4.79× (4.4→20.9ms), scifact: 6.58× (4.1→27.3ms)]
- 6 cell(s) exceed CPU p95 budget ≤1.00× fp16 [arguana: 6.81× (154.5→1051.2ms), fiqa: 3.64× (111.6→406.4ms), nfcorpus: 4.00× (103.6→414.0ms), quora: 13.50× (37.5→505.8ms), scidocs: 14.16× (91.5→1295.4ms), scifact: 6.42× (119.9→770.2ms)]

## Decision rule 2 — rroq158 retention as opt-in storage-saver

- avg ΔNDCG@10 vs fp16: **-1.37 pt** (budget ≥ -1.50 pt)
- avg GPU p95 ratio vs fp16: **1.19×** (budget ≤ 1.20×)

### Verdict: **KEEP rroq158 as opt-in storage-saver alternative**

