"""Correctness tests for the triangular MaxSim Triton kernel.

Run directly:  python research/triangular_maxsim/test_kernel.py
Not registered with CI -- this is a research-side correctness gate.
"""
from __future__ import annotations

import os
import sys

# When invoked as a script (`python research/.../test_kernel.py`), Python
# inserts the script's own dir at sys.path[0], which can shadow the workspace.
# Make sure the workspace root is on sys.path so the in-tree voyager_index wins.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from voyager_index._internal.kernels.triton_triangular_maxsim import (
    triangular_maxsim,
    triangular_maxsim_reference,
)


def _make_inputs(S: int, T: int, U: int, H: int, device: str, seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(S, H, generator=g, device=device)
    C = torch.randn(T, H, generator=g, device=device)
    R = torch.randn(U, H, generator=g, device=device)
    return Q, C, R


def _check(name: str, kernel_t: torch.Tensor, ref_t: torch.Tensor, atol: float):
    diff = (kernel_t.float().cpu() - ref_t.float().cpu()).abs().max().item()
    if diff > atol:
        raise AssertionError(f"{name}: max abs error {diff:.3e} > tol {atol:.3e}")
    return diff


def _check_jstar_consistent(jstar_kernel: torch.Tensor, ref_g: torch.Tensor,
                            kernel_g: torch.Tensor, sim_RC: torch.Tensor,
                            a: torch.Tensor, atol: float):
    """jstar may differ on ties. Verify the value at the chosen index matches g."""
    gated = torch.minimum(sim_RC, a[None, :])
    chosen = gated.gather(1, jstar_kernel.long().cpu().unsqueeze(1)).squeeze(1)
    diff = (chosen - kernel_g.float().cpu()).abs().max().item()
    if diff > atol:
        raise AssertionError(
            f"jstar consistency: chosen-vs-g diff {diff:.3e} > {atol:.3e}"
        )


def run_one(shape, device="cuda", seed=0):
    S, T, U, H = shape
    Q, C, R = _make_inputs(S, T, U, H, device=device, seed=seed)

    ref = triangular_maxsim_reference(Q, C, R, normalize=True)
    out = triangular_maxsim(Q, C, R, normalize=True, use_kernel=True)

    atol = 1e-4
    _check("a", out.a, ref.a, atol)
    _check("g", out.g, ref.g, atol)
    _check("e", out.e, ref.e, atol)
    _check("u", out.u, ref.u, atol)

    Qn = torch.nn.functional.normalize(Q.float(), dim=-1)
    Cn = torch.nn.functional.normalize(C.float(), dim=-1)
    Rn = torch.nn.functional.normalize(R.float(), dim=-1)
    sim_RC = (Rn @ Cn.T).cpu()
    _check_jstar_consistent(out.jstar, ref.g, out.g, sim_RC, ref.a.cpu(), atol=2e-4)

    print(f"  ok shape={shape}")


def run_with_masks(device="cuda", seed=1):
    S, T, U, H = 12, 80, 24, 128
    Q, C, R = _make_inputs(S, T, U, H, device=device, seed=seed)
    Q_mask = torch.ones(S, device=device); Q_mask[8:] = 0
    C_mask = torch.ones(T, device=device); C_mask[60:] = 0
    R_mask = torch.ones(U, device=device); R_mask[20:] = 0

    ref = triangular_maxsim_reference(Q, C, R, Q_mask, C_mask, R_mask)
    out = triangular_maxsim(Q, C, R, Q_mask, C_mask, R_mask, use_kernel=True)

    atol = 1e-4
    # only valid positions need to match exactly
    _check("a (masked)", out.a[C_mask.bool()], ref.a[C_mask.bool()], atol)
    _check("g (masked)", out.g[R_mask.bool()], ref.g[R_mask.bool()], atol)
    _check("e (masked)", out.e[R_mask.bool()], ref.e[R_mask.bool()], atol)
    _check("u (masked)", out.u[Q_mask.bool()], ref.u[Q_mask.bool()], atol)
    print(f"  ok masked shape=({S},{T},{U},{H})")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping kernel tests")
        sys.exit(0)

    shapes = [
        (8, 64, 16, 128),
        (32, 256, 64, 128),
        (16, 192, 48, 96),
        (8, 512, 32, 128),
        (24, 96, 100, 128),
        (4, 33, 7, 64),     # awkward sizes
    ]
    for sh in shapes:
        run_one(sh, seed=hash(sh) & 0xFFFF)
    run_with_masks()
    print("All triangular MaxSim kernel correctness checks passed.")


if __name__ == "__main__":
    main()
