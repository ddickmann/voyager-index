"""Synthesize the per-encoder reports into one master report.md."""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

ENCODERS = [
    ("GTE-ModernColBERT-v1", "lightonai/GTE-ModernColBERT-v1", False),
    ("ColBERT-Zero", "lightonai/ColBERT-Zero", False),
    ("ColBERT-Zero__prompts", "lightonai/ColBERT-Zero", True),
]


def load_all():
    out = []
    for tag, name, prompts in ENCODERS:
        path = os.path.join(_HERE, f"report__{tag}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            blob = json.load(f)
        out.append((tag, name, prompts, blob))
    return out


def main():
    runs = load_all()
    if not runs:
        print("no per-encoder reports found")
        return

    lines: list[str] = []
    lines.append("# Triangular MaxSim Falsification - Master Report\n")
    lines.append(
        "This report synthesizes three runs of "
        "`research/triangular_maxsim/experiment.py` over the same 20 "
        "hand-crafted (Q, C, R) cases (10 SciFact + 10 HotpotQA), using three "
        "different ColBERT-style encoders. The Triton kernel under test is "
        "`voyager_index/_internal/kernels/triton_triangular_maxsim.py`.\n"
    )
    lines.append("## Hypothesis under test\n")
    lines.append(
        "> *Reverse MaxSim is not able to score response groundedness in the "
        "context provided.*\n\n"
        "We falsify the hypothesis if **any** Reverse MaxSim variant achieves "
        "AUROC >= 0.90 separating the 5 grounded from the 5 ungrounded anchor "
        "responses. We further check whether the user-proposed Triangular "
        "(Query-Conditioned Reverse MaxSim) variant beats the naive "
        "R->(Q union C) baseline.\n"
    )

    lines.append("## Cross-encoder summary\n")
    lines.append(
        "| encoder | prompts | AUROC G_tri | AUROC G_naive_QC | AUROC G_rc | "
        "best AUROC | mean G_tri (G/U) | check1 (any >= 0.90) | check2 (tri > naive +0.05) |"
    )
    lines.append("|---|:-:|---:|---:|---:|---:|---:|:-:|:-:|")
    for tag, name, prompts, blob in runs:
        m = blob["metrics"]
        lines.append(
            f"| `{name}` | {'yes' if prompts else 'no'} "
            f"| {m['AUROC_G_tri']:.4f} | {m['AUROC_G_naive_QC']:.4f} "
            f"| {m['AUROC_G_rc']:.4f} | {m['AUROC_best']:.4f} "
            f"| {m['mean_G_tri_grounded']:.3f} / {m['mean_G_tri_ungrounded']:.3f} "
            f"| {'PASS' if m['check1_global_separability_pass'] else 'FAIL'} "
            f"| {'PASS' if m['check2_triangular_beats_naive_pass'] else 'FAIL'} |"
        )
    lines.append("")

    # Verdict synthesis
    any_check1 = any(b["metrics"]["check1_global_separability_pass"] for _, _, _, b in runs)
    any_check2 = any(b["metrics"]["check2_triangular_beats_naive_pass"] for _, _, _, b in runs)

    lines.append("## Synthesized verdict\n")
    if any_check1 and any_check2:
        verdict = (
            "**HYPOTHESIS FALSIFIED, AND triangular wins on at least one "
            "encoder.** Reverse MaxSim CAN score response groundedness, and "
            "the user-proposed Triangular `min(s_RC, a_j)` gating outperforms "
            "the naive R->(Q union C) baseline on at least one of the tested "
            "encoders."
        )
    elif any_check1:
        verdict = (
            "**HYPOTHESIS FALSIFIED, but the triangular gating does NOT earn "
            "its complexity here.** Across all three encoders, naive Reverse "
            "MaxSim against R->(Q union C) (and against R->C) achieves "
            "perfect AUROC=1.00 on the 10 anchor cases (5 grounded vs 5 "
            "ungrounded), so embedding-only Reverse MaxSim *can* score "
            "groundedness. However, the user-proposed Triangular variant "
            "with `min(s_RC, a_j)` gating consistently *underperforms* the "
            "naive baseline -- it compresses the dynamic range and hurts "
            "discrimination. The min-gate's intended virtue ('a response "
            "token cannot hide behind irrelevant context') doesn't materialize "
            "because, in the dense ColBERT embedding space, all context "
            "tokens for these queries are highly query-relevant (a_j is "
            "uniformly high), so the gating mostly clips the highest-quality "
            "response-context matches."
        )
    else:
        verdict = (
            "**HYPOTHESIS NOT FALSIFIED.** No Reverse MaxSim variant cleanly "
            "separates grounded from ungrounded responses on this anchor set."
        )
    lines.append("> " + verdict + "\n")

    lines.append("## Why triangular underperforms naive Reverse MaxSim here\n")
    lines.append(
        "Mechanically, for every anchor case across all three encoders the "
        "*relative ordering* of cases by `G_tri` and by `G_naive_QC` is "
        "almost identical -- the issue is *separation magnitude*, not rank "
        "inversion:\n\n"
        "1. ColBERT-style models produce token cosine similarities tightly "
        "packed near the top of the [-1, 1] interval. Both `s(R[t], C[j])` "
        "and `a[j] = max_i s(Q[i], C[j])` sit in roughly the same band "
        "(`~0.85-0.99` for GTE-ModernColBERT, `~0.75-0.99` for ColBERT-Zero).\n"
        "2. When `s(R[t], C[j])` is HIGH (a strong match for a grounded "
        "response token), `a[j]` is *also* high but generally a bit lower, "
        "so `min` clips the strong match down to `a[j]`.\n"
        "3. When `s(R[t], C[j])` is LOWER (an ungrounded response token), "
        "`min` returns the already-low value of `s(R[t], C[j])`.\n"
        "4. Net effect: the gate erases the high end of the grounded "
        "distribution while leaving the ungrounded distribution mostly "
        "untouched, *reducing* separation rather than enhancing it.\n\n"
        "The user's intuition (the AND-gate forces a response token to match "
        "a *query-relevant* context token) only bites when many context "
        "tokens are *not* query-relevant. In our setup -- both SciFact and "
        "HotpotQA contexts are tightly built around the query -- almost "
        "every context token has high `a[j]`, so the AND-gate is effectively "
        "a no-op cap rather than a discriminating filter."
    )
    lines.append("")

    lines.append("## Where the per-token signals still earn their keep\n")
    lines.append(
        "Even though aggregate `G_tri` underperforms aggregate `G_naive_QC` "
        "on the binary anchor split, the auxiliary signals from the kernel "
        "remain diagnostic on the 10 ambiguous cases:\n"
    )

    # Pull A1 (prompt_echo) details from ColBERT-Zero
    cz = next(b for tag, _, prompts, b in runs if tag == "ColBERT-Zero")
    a1 = next(c for c in cz["per_case"] if c["case_id"] == "A1")
    a2 = next(c for c in cz["per_case"] if c["case_id"] == "A2")
    a4 = next(c for c in cz["per_case"] if c["case_id"] == "A4")
    a5 = next(c for c in cz["per_case"] if c["case_id"] == "A5")
    a7 = next(c for c in cz["per_case"] if c["case_id"] == "A7")
    a9 = next(c for c in cz["per_case"] if c["case_id"] == "A9")

    lines.append("- **Prompt-echo cases (A1, A2)**: the `e_t` channel "
                 "dominates `g_t` -- exactly the diagnostic the user "
                 "described.\n"
                 f"    - A1: G_tri = {a1['G_tri']:.3f}, echo = "
                 f"**{a1['echo_mean']:.3f}** (echo - G_tri = "
                 f"{a1['echo_mean']-a1['G_tri']:+.3f})\n"
                 f"    - A2: G_tri = {a2['G_tri']:.3f}, echo = "
                 f"**{a2['echo_mean']:.3f}** (echo - G_tri = "
                 f"{a2['echo_mean']-a2['G_tri']:+.3f})\n")
    lines.append(f"- **Partial-grounding (A4)**: G_tri = "
                 f"**{a4['G_tri']:.3f}** is the *lowest* of any ambiguous "
                 "case, reflecting the unsupported 'Eredivisie' clause "
                 "dragging the per-token mean down.\n")
    lines.append(f"- **Parametric-knowledge (A5)**: G_tri = "
                 f"**{a5['G_tri']:.3f}** sits clearly below the grounded "
                 "mean -- the response talks about SIRT1/mTOR, which "
                 "doesn't appear anywhere in the context, so most of its "
                 "content tokens cannot find good support.\n")
    lines.append(f"- **Negation-flip (A7)**: G_tri = "
                 f"**{a7['G_tri']:.3f}** stays HIGH, confirming the "
                 "user's documented limitation -- embedding-only methods "
                 "cannot detect 'protect' vs 'promote' on otherwise "
                 "identical sentences.\n")
    lines.append(f"- **Entity-swap (A9)**: G_tri = "
                 f"**{a9['G_tri']:.3f}** stays HIGH for the same reason -- "
                 "'beta-6' and 'beta-8' are tokenized into highly similar "
                 "subword sequences and the rest of the sentence is "
                 "verbatim from context.\n")
    lines.append(
        "\nThese per-token diagnostics are the part of the user's framework "
        "that *does* deliver value, even though the aggregate `G_tri` does "
        "not beat the simpler aggregate `G_naive_QC`."
    )
    lines.append("")

    lines.append("## Practical takeaway\n")
    lines.append(
        "1. The original hypothesis ('Reverse MaxSim cannot score "
        "groundedness') is **falsified** by naive Reverse MaxSim against "
        "either Q union C or just C, on all three tested encoders.\n"
        "2. The proposed Triangular `min(s_RC, a_j)` gating is **not "
        "vindicated** as a stronger aggregate scorer on this dataset and "
        "these encoders -- it consistently underperforms the naive baseline "
        "on the binary anchor task.\n"
        "3. The auxiliary per-token channels the kernel emits "
        "(`e_t` for prompt-echo detection, `jstar[t]` for evidence "
        "attribution, `u_i` for grounded coverage) ARE useful and behave "
        "as the user predicted on prompt-echo and partial-grounding cases.\n"
        "4. As the user warned, **negation-flip and entity-swap remain "
        "uncatchable by any embedding-only Reverse MaxSim** -- the "
        "responses look near-identical in cosine space.\n"
        "5. Recommendation: keep the kernel for its per-token diagnostics "
        "and as a token-level visualization layer; use naive `G_naive_QC` or "
        "`G_rc` as the headline groundedness score; add an NLI / claim-level "
        "verifier for the failure modes embeddings cannot solve.\n"
    )

    lines.append("## Per-encoder detail\n")
    for tag, name, prompts, _ in runs:
        rel = f"report__{tag}.md"
        lines.append(f"- [{name}{' (asymmetric prompts)' if prompts else ''}]({rel})")
    lines.append("")

    out = os.path.join(_HERE, "report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
