"""Run the Triangular MaxSim falsification experiment.

For each of 20 hand-crafted (Q, C, R) cases:
  1. Compute Triangular MaxSim signals (a, g, e, u, jstar) via the Triton
     kernel and the PyTorch reference.
  2. Compute baselines: naive Reverse MaxSim against Q union C, and Reverse
     MaxSim against just C.
  3. Aggregate to a single G(R|Q,C) per case using a content-token weighting.
  4. Evaluate AUROC on the 10 grounded-vs-ungrounded anchors and inspect the
     10 ambiguous cases qualitatively.

Outputs `report.md` and `report.json` next to this script.
"""
from __future__ import annotations

import json
import math
import os
import string
import sys
from dataclasses import asdict
from typing import List, Sequence, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from voyager_index._internal.kernels.triton_triangular_maxsim import (  # noqa: E402
    triangular_maxsim,
    triangular_maxsim_reference,
    weighted_groundedness,
    grounded_coverage,
    naive_reverse_maxsim_qc,
    reverse_maxsim_rc,
)
from research.triangular_maxsim.cases import CASES  # noqa: E402
from research.triangular_maxsim.dataset import (  # noqa: E402
    DEFAULT_MODEL, EmbeddedCase, load_or_build,
)


# ---------------------------------------------------------------------------
# Token weighting (dependency-free; spaCy not installed in this env)
# ---------------------------------------------------------------------------

_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]", "<s>", "</s>", "<pad>"}

# A small, conventional English stopword set.
_STOPWORDS = set("""
a an the of in on at to for from by with as is are was were be been being am
do does did doing have has had having and or but if then else than that this
these those it its it's they them their there here he she his her hers we us
our you your yours i me my mine not no nor so very can could should would may
might must will shall just only also too into about over under up down out
which who whom whose what when where why how all any both each few more most
other some such own same own own
""".split())


def _strip_marker(tok: str) -> str:
    # ModernBERT/GPT BPE uses 'Ġ' prefix to mark word boundaries.
    if tok.startswith("Ġ"):
        return tok[1:]
    if tok.startswith("##"):
        return tok[2:]
    return tok


def token_weights(tokens: Sequence[str]) -> torch.Tensor:
    """Content-token weights: special/punct/stopwords -> 0, content -> 1.

    Numeric tokens get a small boost (1.5) since dates/numbers are
    high-information content per the user's spec.
    """
    out = []
    for t in tokens:
        if t in _SPECIAL_TOKENS:
            out.append(0.0); continue
        s = _strip_marker(t)
        if not s:
            out.append(0.0); continue
        if all(ch in string.punctuation for ch in s):
            out.append(0.0); continue
        if s.lower() in _STOPWORDS:
            out.append(0.0); continue
        if any(ch.isdigit() for ch in s):
            out.append(1.5); continue
        out.append(1.0)
    return torch.tensor(out, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------

def score_case(ec: EmbeddedCase, device: torch.device) -> dict:
    Q = ec.Q.to(device)
    C = ec.C.to(device)
    R = ec.R.to(device)

    # Triangular MaxSim (kernel)
    tri_kernel = triangular_maxsim(Q, C, R, normalize=True, use_kernel=device.type == "cuda")
    # Reference for sanity
    tri_ref = triangular_maxsim_reference(Q, C, R, normalize=True)

    g_kernel = tri_kernel.g.detach().float().cpu()
    g_ref = tri_ref.g.detach().float().cpu()
    kernel_match_max_err = (g_kernel - g_ref).abs().max().item()

    e_per = tri_kernel.e.detach().float().cpu()
    g_per = g_kernel
    a_per = tri_kernel.a.detach().float().cpu()
    u_per = tri_kernel.u.detach().float().cpu()
    jstar = tri_kernel.jstar.detach().long().cpu()

    w_R = token_weights(ec.r_tokens)
    w_Q = token_weights(ec.q_tokens)

    G_tri = weighted_groundedness(g_per, w_R)
    echo_mean = weighted_groundedness(e_per, w_R)
    GC = grounded_coverage(u_per, w_Q if w_Q.sum() > 0 else None)

    # Baselines
    G_naive, g_naive_per = naive_reverse_maxsim_qc(Q, C, R, weights=w_R, normalize=True)
    G_rc, g_rc_per = reverse_maxsim_rc(C, R, weights=w_R, normalize=True)

    # Top-3 evidence excerpt: for each of the highest-weighted response tokens,
    # show (response_token -> context_token via jstar)
    evidence = []
    if w_R.sum() > 0:
        weighted = g_per * w_R
        top_idx = torch.topk(weighted, k=min(5, weighted.numel())).indices.tolist()
        for t_idx in top_idx:
            j = int(jstar[t_idx].item())
            r_tok = ec.r_tokens[t_idx] if t_idx < len(ec.r_tokens) else "?"
            c_tok = ec.c_tokens[j] if 0 <= j < len(ec.c_tokens) else "?"
            evidence.append({
                "r_idx": int(t_idx), "r_tok": r_tok,
                "c_idx": j, "c_tok": c_tok,
                "g": float(g_per[t_idx].item()),
                "weight": float(w_R[t_idx].item()),
            })

    return {
        "case_id": ec.case.id,
        "label": ec.case.label,
        "subcategory": ec.case.subcategory,
        "source": ec.case.source,
        "G_tri": G_tri,
        "G_naive_QC": G_naive,
        "G_rc": G_rc,
        "echo_mean": echo_mean,
        "GC": GC,
        "kernel_vs_ref_max_err": kernel_match_max_err,
        "g_per_token": g_per.tolist(),
        "e_per_token": e_per.tolist(),
        "a_per_context_token": a_per.tolist(),
        "u_per_query_token": u_per.tolist(),
        "jstar": jstar.tolist(),
        "r_tokens": ec.r_tokens,
        "q_tokens": ec.q_tokens,
        "c_tokens": ec.c_tokens,
        "r_weights": w_R.tolist(),
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUROC with ties handled (average rank)."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-sum approach
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def best_threshold(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, int]:
    """Threshold with highest accuracy on the binary task."""
    candidates = sorted(set(scores))
    best_acc = -1
    best_thr = 0.0
    for thr in candidates:
        preds = [1 if s >= thr else 0 for s in scores]
        correct = sum(1 for p, l in zip(preds, labels) if p == l)
        if correct > best_acc:
            best_acc = correct
            best_thr = thr
    return best_thr, best_acc


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(model_name: str = DEFAULT_MODEL, use_prompts: bool = False) -> dict:
    embedded = load_or_build(rebuild=False, model_name=model_name, use_prompts=use_prompts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_case = [score_case(ec, device) for ec in embedded]

    # Anchor (grounded vs ungrounded) AUROCs
    anchors = [r for r in per_case if r["label"] in ("grounded", "ungrounded")]
    labels = [1 if r["label"] == "grounded" else 0 for r in anchors]

    metrics = {}
    for key in ("G_tri", "G_naive_QC", "G_rc"):
        scores = [r[key] for r in anchors]
        metrics[f"AUROC_{key}"] = auroc(scores, labels)
        thr, correct = best_threshold(scores, labels)
        metrics[f"best_thr_{key}"] = thr
        metrics[f"best_acc_{key}"] = correct / max(len(labels), 1)

    # Means by label
    by_label = {"grounded": [], "ungrounded": []}
    for r in anchors:
        by_label[r["label"]].append(r["G_tri"])
    metrics["mean_G_tri_grounded"] = sum(by_label["grounded"]) / len(by_label["grounded"])
    metrics["mean_G_tri_ungrounded"] = sum(by_label["ungrounded"]) / len(by_label["ungrounded"])
    metrics["G_tri_separation_margin"] = (
        metrics["mean_G_tri_grounded"] - metrics["mean_G_tri_ungrounded"]
    )

    # Falsification verdict
    auroc_tri = metrics["AUROC_G_tri"]
    auroc_naive = metrics["AUROC_G_naive_QC"]
    auroc_rc = metrics["AUROC_G_rc"]
    auroc_best = max(auroc_tri, auroc_naive, auroc_rc)
    metrics["AUROC_best"] = auroc_best

    # The user's hypothesis to falsify: "Reverse MaxSim cannot score response
    # groundedness in the context." That hypothesis is falsified the moment
    # ANY Reverse MaxSim variant cleanly separates grounded vs ungrounded.
    check1_pass = auroc_best >= 0.90
    # The user's secondary claim: the triangular variant is better than the
    # naive R->(Q union C) baseline.
    check2_pass = (auroc_tri - auroc_naive) >= 0.05
    metrics["check1_global_separability_pass"] = bool(check1_pass)
    metrics["check2_triangular_beats_naive_pass"] = bool(check2_pass)

    if check1_pass and check2_pass:
        verdict = (
            "HYPOTHESIS FALSIFIED, AND triangular wins. Reverse MaxSim cleanly "
            "separates grounded from ungrounded responses (best AUROC >= 0.90), "
            "and the Triangular variant beats naive Reverse MaxSim against "
            "Q union C by >= 0.05 AUROC. The triangular gating earns its "
            "complexity on this dataset/encoder."
        )
    elif check1_pass and not check2_pass:
        verdict = (
            "HYPOTHESIS FALSIFIED, but naive Reverse MaxSim suffices. "
            "Naive R->(Q union C) and/or R->C MaxSim cleanly separate "
            "grounded from ungrounded responses (AUROC >= 0.90), so "
            "embedding-only Reverse MaxSim CAN score groundedness on this "
            "anchor set. However, the Triangular `min(s_RC, a_j)` gating "
            "does NOT outperform the naive baseline here -- in fact it "
            "compresses the dynamic range and reduces separation. The "
            "triangular structure is not earning its keep on this "
            "dataset/encoder."
        )
    else:
        verdict = (
            "HYPOTHESIS NOT FALSIFIED. Embedding-only Reverse MaxSim of any "
            "flavor (triangular, naive R->Q union C, R->C) fails to cleanly "
            "separate grounded from ungrounded responses on this anchor set."
        )
    metrics["verdict"] = verdict

    return {
        "model": model_name,
        "use_prompts": use_prompts,
        "metrics": metrics,
        "per_case": per_case,
        "cases": [asdict(c) for c in CASES],
    }


def write_outputs(result: dict, out_md: str, out_json: str) -> None:
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    md = render_markdown(result)
    with open(out_md, "w") as f:
        f.write(md)


def render_markdown(result: dict) -> str:
    m = result["metrics"]
    per = result["per_case"]
    lines: list[str] = []
    lines.append("# Triangular MaxSim Falsification Report\n")
    lines.append(
        f"Run of `research/triangular_maxsim/experiment.py` on the 20 "
        f"hand-crafted SciFact + HotpotQA cases. Each (Q, C, R) triple is "
        f"embedded with `{result['model']}` (use_prompts={result['use_prompts']}) "
        f"and scored with the Triton kernel in "
        f"`voyager_index/_internal/kernels/triton_triangular_maxsim.py`.\n"
    )
    lines.append("## Verdict\n\n> " + m["verdict"] + "\n")

    lines.append("## Anchor metrics (grounded=10 cases? actually 5+5)\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in (
        "AUROC_G_tri", "AUROC_G_naive_QC", "AUROC_G_rc", "AUROC_best",
        "mean_G_tri_grounded", "mean_G_tri_ungrounded",
        "G_tri_separation_margin",
        "best_thr_G_tri", "best_acc_G_tri",
        "best_thr_G_naive_QC", "best_acc_G_naive_QC",
        "check1_global_separability_pass", "check2_triangular_beats_naive_pass",
    ):
        v = m[k]
        if isinstance(v, float):
            lines.append(f"| `{k}` | {v:.4f} |")
        else:
            lines.append(f"| `{k}` | {v} |")
    lines.append("")

    lines.append("## Per-case scores\n")
    lines.append("| id | label | sub | G_tri | G_naive_QC | G_rc | echo | GC | kernel-vs-ref err |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in per:
        lines.append(
            f"| {r['case_id']} | {r['label']} | {r['subcategory'] or ''} "
            f"| {r['G_tri']:.4f} | {r['G_naive_QC']:.4f} | {r['G_rc']:.4f} "
            f"| {r['echo_mean']:.4f} | {r['GC']:.4f} | {r['kernel_vs_ref_max_err']:.2e} |"
        )
    lines.append("")

    lines.append("## Anchor cases - sorted by G_tri\n")
    anchors = [r for r in per if r["label"] in ("grounded", "ungrounded")]
    anchors_sorted = sorted(anchors, key=lambda r: -r["G_tri"])
    lines.append("| rank | id | label | G_tri | G_naive_QC | G_rc |")
    lines.append("|---:|---|---|---:|---:|---:|")
    for i, r in enumerate(anchors_sorted, 1):
        marker = "OK" if r["label"] == "grounded" else "X "
        lines.append(
            f"| {i} | {r['case_id']} `{marker}` | {r['label']} "
            f"| {r['G_tri']:.4f} | {r['G_naive_QC']:.4f} | {r['G_rc']:.4f} |"
        )
    lines.append("")

    lines.append("## Ambiguous cases - per-subcategory diagnostics\n")
    sub_order = ["prompt_echo", "partial", "parametric", "negation_flip", "entity_swap"]
    by_sub: dict[str, list] = {s: [] for s in sub_order}
    for r in per:
        if r["label"] == "ambiguous" and r["subcategory"] in by_sub:
            by_sub[r["subcategory"]].append(r)

    for sub in sub_order:
        lines.append(f"### {sub}\n")
        lines.append("| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for r in by_sub[sub]:
            ev = "; ".join(
                f"`{e['r_tok']}`->`{e['c_tok']}` ({e['g']:.2f})" for e in r["evidence"][:3]
            )
            lines.append(
                f"| {r['case_id']} | {r['G_tri']:.4f} | {r['echo_mean']:.4f} "
                f"| {r['G_naive_QC']:.4f} | {r['G_rc']:.4f} | {ev} |"
            )
        lines.append("")

    lines.append("## Case detail and evidence pointers\n")
    for r in per:
        c = next(x for x in result["cases"] if x["id"] == r["case_id"])
        lines.append(f"### {r['case_id']} - {r['label']}{(' / ' + (r['subcategory'] or '')) if r['subcategory'] else ''} - {r['source']}")
        lines.append(f"- Q: {c['query']}")
        lines.append(f"- R: {c['response']}")
        lines.append(
            f"- G_tri = **{r['G_tri']:.4f}**, G_naive_QC = {r['G_naive_QC']:.4f}, "
            f"G_rc = {r['G_rc']:.4f}, echo = {r['echo_mean']:.4f}"
        )
        if r["evidence"]:
            lines.append("- top response tokens -> grounding context tokens:")
            for e in r["evidence"]:
                lines.append(
                    f"    - `{e['r_tok']}` (resp t={e['r_idx']}, w={e['weight']:.1f}) "
                    f"-> `{e['c_tok']}` (ctx j={e['c_idx']}, g={e['g']:.3f})"
                )
        lines.append(f"- notes: {c['notes']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompts", action="store_true")
    p.add_argument("--tag", default=None,
                   help="suffix for report.{md,json}; defaults to model basename.")
    args = p.parse_args()
    tag = args.tag or args.model.split("/")[-1] + ("__prompts" if args.prompts else "")
    out = run(model_name=args.model, use_prompts=args.prompts)
    out_json = os.path.join(_HERE, f"report__{tag}.json")
    out_md = os.path.join(_HERE, f"report__{tag}.md")
    write_outputs(out, out_md, out_json)
    m = out["metrics"]
    print("=" * 60)
    print(m["verdict"])
    print("=" * 60)
    print(f"AUROC G_tri        = {m['AUROC_G_tri']:.4f}")
    print(f"AUROC G_naive_QC   = {m['AUROC_G_naive_QC']:.4f}")
    print(f"AUROC G_rc         = {m['AUROC_G_rc']:.4f}")
    print(f"mean G_tri grounded   = {m['mean_G_tri_grounded']:.4f}")
    print(f"mean G_tri ungrounded = {m['mean_G_tri_ungrounded']:.4f}")
    print(f"separation margin     = {m['G_tri_separation_margin']:.4f}")
    print()
    print(f"report.md  -> {out_md}")
    print(f"report.json -> {out_json}")
