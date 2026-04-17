"""Embed the 20 hand-crafted cases with the same ColBERT model used in the
voyager-index BEIR benchmark (lightonai/GTE-ModernColBERT-v1).

Caches per-case (tokens, embeddings) tuples to disk so the experiment script
does not have to load the model on every run.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

# Ensure we import the in-tree voyager_index when running as a script
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.triangular_maxsim.cases import CASES, Case  # noqa: E402


DEFAULT_MODEL = "lightonai/GTE-ModernColBERT-v1"


def cache_path_for(model_name: str) -> str:
    safe = model_name.replace("/", "__")
    return os.path.join(_HERE, f"embeddings__{safe}.pt")


# Back-compat default
MODEL_NAME = DEFAULT_MODEL
CACHE_PATH = cache_path_for(DEFAULT_MODEL)


@dataclass
class EmbeddedCase:
    case: Case
    q_tokens: List[str]
    c_tokens: List[str]
    r_tokens: List[str]
    Q: torch.Tensor   # (S, H)
    C: torch.Tensor   # (T, H)
    R: torch.Tensor   # (U, H)


def _tokenize(model, text: str) -> List[str]:
    ids = model.tokenizer(text, add_special_tokens=True)["input_ids"]
    return model.tokenizer.convert_ids_to_tokens(ids)


def _encode_one(model, text: str, is_query: bool, prompt_name: Optional[str] = None) -> torch.Tensor:
    kwargs = {"is_query": is_query}
    if prompt_name is not None:
        # Pylate / sentence-transformers style: select an asymmetric prompt
        kwargs["prompt_name"] = prompt_name
    try:
        out = model.encode([text], **kwargs)
    except TypeError:
        # Older pylate without prompt_name kwarg
        out = model.encode([text], is_query=is_query)
    arr = out[0]
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(np.asarray(arr))
    return arr.float()


def embed_cases(
    cases: Sequence[Case] = CASES,
    device: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    use_prompts: bool = False,
) -> List[EmbeddedCase]:
    """Embed every case.

    use_prompts=True activates ColBERT-Zero-style asymmetric prompts
    (prompt_name='query' for Q, 'document' for C and R).
    """
    from pylate import models

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.ColBERT(
        model_name_or_path=model_name,
        device=device,
        do_query_expansion=False,
    )
    q_prompt = "query" if use_prompts else None
    d_prompt = "document" if use_prompts else None

    embedded: List[EmbeddedCase] = []
    for c in cases:
        q_tok = _tokenize(model, c.query)
        c_tok = _tokenize(model, c.context)
        r_tok = _tokenize(model, c.response)

        Q = _encode_one(model, c.query, is_query=True, prompt_name=q_prompt)
        C = _encode_one(model, c.context, is_query=False, prompt_name=d_prompt)
        R = _encode_one(model, c.response, is_query=False, prompt_name=d_prompt)

        # token-emb length sanity
        if Q.shape[0] != len(q_tok):
            q_tok = q_tok[: Q.shape[0]] if len(q_tok) > Q.shape[0] else q_tok + ["?"] * (Q.shape[0] - len(q_tok))
        if C.shape[0] != len(c_tok):
            c_tok = c_tok[: C.shape[0]] if len(c_tok) > C.shape[0] else c_tok + ["?"] * (C.shape[0] - len(c_tok))
        if R.shape[0] != len(r_tok):
            r_tok = r_tok[: R.shape[0]] if len(r_tok) > R.shape[0] else r_tok + ["?"] * (R.shape[0] - len(r_tok))

        embedded.append(
            EmbeddedCase(case=c, q_tokens=q_tok, c_tokens=c_tok, r_tokens=r_tok,
                         Q=Q.cpu(), C=C.cpu(), R=R.cpu())
        )
    return embedded


def load_or_build(
    cases: Sequence[Case] = CASES,
    cache_path: Optional[str] = None,
    rebuild: bool = False,
    device: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    use_prompts: bool = False,
) -> List[EmbeddedCase]:
    if cache_path is None:
        cache_path = cache_path_for(model_name + ("__prompts" if use_prompts else ""))
    if not rebuild and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("model") == model_name and blob.get("n_cases") == len(cases):
            return blob["embedded"]
    embedded = embed_cases(cases, device=device, model_name=model_name, use_prompts=use_prompts)
    with open(cache_path, "wb") as f:
        pickle.dump({"model": model_name, "n_cases": len(cases), "embedded": embedded}, f)
    return embedded


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompts", action="store_true",
                   help="Use asymmetric query/document prompts (required for ColBERT-Zero).")
    args = p.parse_args()
    embedded = load_or_build(rebuild=True, model_name=args.model, use_prompts=args.prompts)
    print(f"Embedded {len(embedded)} cases with {args.model} prompts={args.prompts}. (S, T, U):")
    for ec in embedded:
        print(f"  {ec.case.id:4s}  S={ec.Q.shape[0]:>3d}  T={ec.C.shape[0]:>4d}  U={ec.R.shape[0]:>3d}  ({ec.case.source})")
