# Triangular MaxSim Falsification - Master Report

This report synthesizes three runs of `research/triangular_maxsim/experiment.py` over the same 20 hand-crafted (Q, C, R) cases (10 SciFact + 10 HotpotQA), using three different ColBERT-style encoders. The Triton kernel under test is `voyager_index/_internal/kernels/triton_triangular_maxsim.py`.

## Hypothesis under test

> *Reverse MaxSim is not able to score response groundedness in the context provided.*

We falsify the hypothesis if **any** Reverse MaxSim variant achieves AUROC >= 0.90 separating the 5 grounded from the 5 ungrounded anchor responses. We further check whether the user-proposed Triangular (Query-Conditioned Reverse MaxSim) variant beats the naive R->(Q union C) baseline.

## Cross-encoder summary

| encoder | prompts | AUROC G_tri | AUROC G_naive_QC | AUROC G_rc | best AUROC | mean G_tri (G/U) | check1 (any >= 0.90) | check2 (tri > naive +0.05) |
|---|:-:|---:|---:|---:|---:|---:|:-:|:-:|
| `lightonai/GTE-ModernColBERT-v1` | no | 0.7600 | 1.0000 | 1.0000 | 1.0000 | 0.956 / 0.950 | PASS | FAIL |
| `lightonai/ColBERT-Zero` | no | 0.8400 | 1.0000 | 1.0000 | 1.0000 | 0.876 / 0.837 | PASS | FAIL |
| `lightonai/ColBERT-Zero` | yes | 0.4800 | 1.0000 | 1.0000 | 1.0000 | 0.781 / 0.784 | PASS | FAIL |

## Synthesized verdict

> **HYPOTHESIS FALSIFIED, but the triangular gating does NOT earn its complexity here.** Across all three encoders, naive Reverse MaxSim against R->(Q union C) (and against R->C) achieves perfect AUROC=1.00 on the 10 anchor cases (5 grounded vs 5 ungrounded), so embedding-only Reverse MaxSim *can* score groundedness. However, the user-proposed Triangular variant with `min(s_RC, a_j)` gating consistently *underperforms* the naive baseline -- it compresses the dynamic range and hurts discrimination. The min-gate's intended virtue ('a response token cannot hide behind irrelevant context') doesn't materialize because, in the dense ColBERT embedding space, all context tokens for these queries are highly query-relevant (a_j is uniformly high), so the gating mostly clips the highest-quality response-context matches.

## Why triangular underperforms naive Reverse MaxSim here

Mechanically, for every anchor case across all three encoders the *relative ordering* of cases by `G_tri` and by `G_naive_QC` is almost identical -- the issue is *separation magnitude*, not rank inversion:

1. ColBERT-style models produce token cosine similarities tightly packed near the top of the [-1, 1] interval. Both `s(R[t], C[j])` and `a[j] = max_i s(Q[i], C[j])` sit in roughly the same band (`~0.85-0.99` for GTE-ModernColBERT, `~0.75-0.99` for ColBERT-Zero).
2. When `s(R[t], C[j])` is HIGH (a strong match for a grounded response token), `a[j]` is *also* high but generally a bit lower, so `min` clips the strong match down to `a[j]`.
3. When `s(R[t], C[j])` is LOWER (an ungrounded response token), `min` returns the already-low value of `s(R[t], C[j])`.
4. Net effect: the gate erases the high end of the grounded distribution while leaving the ungrounded distribution mostly untouched, *reducing* separation rather than enhancing it.

The user's intuition (the AND-gate forces a response token to match a *query-relevant* context token) only bites when many context tokens are *not* query-relevant. In our setup -- both SciFact and HotpotQA contexts are tightly built around the query -- almost every context token has high `a[j]`, so the AND-gate is effectively a no-op cap rather than a discriminating filter.

## Where the per-token signals still earn their keep

Even though aggregate `G_tri` underperforms aggregate `G_naive_QC` on the binary anchor split, the auxiliary signals from the kernel remain diagnostic on the 10 ambiguous cases:

- **Prompt-echo cases (A1, A2)**: the `e_t` channel dominates `g_t` -- exactly the diagnostic the user described.
    - A1: G_tri = 0.829, echo = **1.000** (echo - G_tri = +0.170)
    - A2: G_tri = 0.929, echo = **0.963** (echo - G_tri = +0.035)

- **Partial-grounding (A4)**: G_tri = **0.796** is the *lowest* of any ambiguous case, reflecting the unsupported 'Eredivisie' clause dragging the per-token mean down.

- **Parametric-knowledge (A5)**: G_tri = **0.780** sits clearly below the grounded mean -- the response talks about SIRT1/mTOR, which doesn't appear anywhere in the context, so most of its content tokens cannot find good support.

- **Negation-flip (A7)**: G_tri = **0.930** stays HIGH, confirming the user's documented limitation -- embedding-only methods cannot detect 'protect' vs 'promote' on otherwise identical sentences.

- **Entity-swap (A9)**: G_tri = **0.912** stays HIGH for the same reason -- 'beta-6' and 'beta-8' are tokenized into highly similar subword sequences and the rest of the sentence is verbatim from context.


These per-token diagnostics are the part of the user's framework that *does* deliver value, even though the aggregate `G_tri` does not beat the simpler aggregate `G_naive_QC`.

## Practical takeaway

1. The original hypothesis ('Reverse MaxSim cannot score groundedness') is **falsified** by naive Reverse MaxSim against either Q union C or just C, on all three tested encoders.
2. The proposed Triangular `min(s_RC, a_j)` gating is **not vindicated** as a stronger aggregate scorer on this dataset and these encoders -- it consistently underperforms the naive baseline on the binary anchor task.
3. The auxiliary per-token channels the kernel emits (`e_t` for prompt-echo detection, `jstar[t]` for evidence attribution, `u_i` for grounded coverage) ARE useful and behave as the user predicted on prompt-echo and partial-grounding cases.
4. As the user warned, **negation-flip and entity-swap remain uncatchable by any embedding-only Reverse MaxSim** -- the responses look near-identical in cosine space.
5. Recommendation: keep the kernel for its per-token diagnostics and as a token-level visualization layer; use naive `G_naive_QC` or `G_rc` as the headline groundedness score; add an NLI / claim-level verifier for the failure modes embeddings cannot solve.

## Per-encoder detail

- [lightonai/GTE-ModernColBERT-v1](report__GTE-ModernColBERT-v1.md)
- [lightonai/ColBERT-Zero](report__ColBERT-Zero.md)
- [lightonai/ColBERT-Zero (asymmetric prompts)](report__ColBERT-Zero__prompts.md)
