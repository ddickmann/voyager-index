## BEIR 2026-Q2 codec averages (6-dataset mean)

| Codec | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | GPU P95 (ms) | CPU P95 (ms) | GPU QPS | CPU QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.5206 | 0.5498 | 0.6105 | 0.7360 | 4.0 | 103.1 | 271.2 | 127.1 |
| int8 | 0.5200 | 0.5491 | 0.6106 | 0.7357 | 4.0 | n/a | 277.9 | n/a |
| rroq158 | 0.5063 | 0.5376 | 0.5950 | 0.7312 | 4.6 | 812.3 | 285.9 | 13.9 |
| rroq4_riem | 0.5208 | 0.5505 | 0.6105 | 0.7383 | 20.1 | 1303.6 | 119.0 | 10.0 |

## Decision rule 1 — rroq4_riem default promotion (Phase F1)

- avg ΔNDCG@10 vs fp16: **+0.02 pt** (budget ≥ -0.50 pt)
- avg ΔR@100 vs fp16: **+0.23 pt** (budget ≥ -0.30 pt)
- per-cell GPU p95 vs fp16: **6 violation(s)** (budget ≤ 1.00× on every cell)
- per-cell CPU p95 vs fp16: **6 violation(s)** (budget ≤ 1.00× on every cell)

### Verdict: **REVERT default to `Compression.RROQ158`** (ship Option-2 honesty-doc framing)

Failed condition(s):
- 6 cell(s) exceed GPU p95 budget ≤1.00× fp16 [arguana: 9.00× (4.0→36.3ms), fiqa: 4.14× (5.0→20.6ms), nfcorpus: 2.99× (4.1→12.2ms), quora: 1.20× (2.7→3.2ms), scidocs: 4.79× (4.4→20.9ms), scifact: 6.58× (4.1→27.3ms)]
- 6 cell(s) exceed CPU p95 budget ≤1.00× fp16 [arguana: 14.30× (154.5→2209.0ms), fiqa: 10.57× (111.6→1179.5ms), nfcorpus: 10.11× (103.6→1046.6ms), quora: 12.85× (37.5→481.3ms), scidocs: 14.22× (91.5→1300.4ms), scifact: 13.38× (119.9→1604.9ms)]

## Decision rule 2 — rroq158 retention as opt-in storage-saver

- avg ΔNDCG@10 vs fp16: **-1.43 pt** (budget ≥ -1.50 pt)
- avg GPU p95 ratio vs fp16: **1.13×** (budget ≤ 1.20×)

### Verdict: **KEEP rroq158 as opt-in storage-saver alternative**

## Codec-fidelity overlap (brute-force, vs fp16)

Each cell shows the average per-query top-K overlap of the codec's brute-force MaxSim ranking with fp16 brute-force MaxSim. 100% means the codec returns exactly the same documents in the top-K as fp16.

| Dataset | Codec | overlap@10 | overlap@20 | overlap@50 | overlap@100 |
|---|---|---:|---:|---:|---:|
| arguana | rroq158 | 0.8223 | 0.8185 | 0.7997 | 0.7851 |
| arguana | rroq4_riem | 0.9680 | 0.9688 | 0.9681 | 0.9665 |
| fiqa | rroq158 | 0.7508 | 0.7612 | 0.7735 | 0.7840 |
| fiqa | rroq4_riem | 0.9495 | 0.9540 | 0.9584 | 0.9616 |
| nfcorpus | rroq158 | 0.7954 | 0.8014 | 0.8046 | 0.8038 |
| nfcorpus | rroq4_riem | 0.9576 | 0.9591 | 0.9611 | 0.9629 |
| quora | rroq158 | 0.7292 | 0.7246 | 0.7220 | 0.7206 |
| quora | rroq4_riem | 0.9487 | 0.9491 | 0.9510 | 0.9515 |
| scidocs | rroq158 | 0.8213 | 0.8308 | 0.8377 | 0.8462 |
| scidocs | rroq4_riem | 0.9638 | 0.9662 | 0.9695 | 0.9718 |
| scifact | rroq158 | 0.8280 | 0.8248 | 0.8251 | 0.8299 |
| scifact | rroq4_riem | 0.9643 | 0.9658 | 0.9661 | 0.9675 |

