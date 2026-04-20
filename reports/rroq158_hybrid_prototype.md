# rroq158 outlier-rescue hybrid prototype — Phase A results

- **Dataset**: `arguana` (n_docs=8,674, n_queries_eval=1,401, n_tokens=1,541,060, dim=128)
- **Rich regime**: K=8192, group_size=32, sign + nz planes (today's production config).
- **Cheap regime**: K=8192 (shared), group_size=128, sign-only (no nz plane).
- **Rescue selection**: top-`p` tokens by ambient residual energy ``||r_amb||``.
- **Scoring**: pure-numpy MaxSim over float32 reconstructions; the kernel adds at most fp16 rounding on the cached scales/norms.
- **Seed**: 42

## Per-`p` results

| variant | predicted B/tok | bits/coord | NDCG@10 | ΔNDCG@10 | Recall@100 | ΔRecall@100 | gate verdict |
|---------|----------------:|-----------:|--------:|---------:|-----------:|------------:|--------------|
| FP32 (no quant) — ceiling | 256.0 | 16.000 | 0.3688 | +0.0286 | 0.9593 | +0.0143 | out of gate range |
| p=0.00 | 25.0 | 1.562 | 0.3079 | -0.0323 | 0.9158 | -0.0293 | all cheap (floor) |
| p=0.05 | 26.1 | 1.631 | 0.3127 | -0.0275 | 0.9186 | -0.0264 | fail (NDCG@10 Δ=-0.0275 < -0.005; Recall@100 Δ=-0.0264 < -0.010) |
| p=0.10 | 27.2 | 1.700 | 0.3167 | -0.0235 | 0.9229 | -0.0221 | fail (NDCG@10 Δ=-0.0235 < -0.005; Recall@100 Δ=-0.0221 < -0.010) |
| p=0.20 | 29.4 | 1.838 | 0.3217 | -0.0185 | 0.9265 | -0.0186 | fail (NDCG@10 Δ=-0.0185 < -0.005; Recall@100 Δ=-0.0186 < -0.010) |
| p=0.30 | 31.6 | 1.975 | 0.3206 | -0.0196 | 0.9286 | -0.0164 | fail (NDCG@10 Δ=-0.0196 < -0.005; Recall@100 Δ=-0.0164 < -0.010) |
| p=0.50 | 36.0 | 2.250 | 0.3259 | -0.0143 | 0.9293 | -0.0157 | out of gate range |
| p=1.00 | 47.0 | 2.938 | 0.3402 | +0.0000 | 0.9450 | +0.0000 | _baseline (all rich)_ |

## Decision Gate 2 verdict

**FAIL**: no `p ≤ 0.30` variant passes both quality (NDCG@10 within −0.005, Recall@100 within −0.010) and storage (B/tok ≤ 35) constraints simultaneously.

**Recommended next step**: skip the hybrid kernel work and ship the uniform Pareto win from Decision Gate 1 only. The dense ternary planes are intrinsically hard to compress further on ModernColBERT (dim=128) without a higher-bit rescue tier.

## Predicted blended-storage table (design doc Table 4.1)

| p (rescue) | B/tok (incl. 1-B regime tag) | bits/coord | notes |
|-----------:|-----------------------------:|-----------:|-------|
| 0.00 | 25.0 | 1.562 | all cheap (floor) |
| 0.05 | 26.1 | 1.631 | blended |
| 0.10 | 27.2 | 1.700 | blended |
| 0.20 | 29.4 | 1.838 | blended |
| 0.30 | 31.6 | 1.975 | blended |
| 0.50 | 36.0 | 2.250 | blended |
| 1.00 | 47.0 | 2.938 | all rich (today's baseline ≈) |

_Total wall: 97.1 min._

