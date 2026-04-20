"""
Phase A4 — token-salience pruning sweep.

For each prune rate ∈ {0, 10, 20, 30, 50}%, drop the bottom-X% of tokens
by ``norm`` salience (the cheap-and-fast signal — IDF + attention_mass
require the production embedder and are out-of-scope here), then for the
remaining tokens measure two things on the held-out fixture:

1. ``token_recall@K`` — what fraction of each query's true top-K
   nearest-neighbour tokens survive the prune. K ∈ {1, 5, 50}. This is
   the upstream-of-rerank recall: a 5-point drop here is the absolute
   ceiling on what the downstream BEIR Recall@10 can degrade by.
2. ``disk_fraction`` — surviving token count / original token count;
   maps directly to disk footprint of the persisted shard.

Memory plan: ~10 MB peak per rate. Runs in <2 s on CPU. Writes a JSON
report to ``reports/a4_salience.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import salience

log = logging.getLogger(__name__)


@dataclass
class SalienceRow:
    signal: str
    prune_rate: float
    kept_tokens: int
    n_docs: int
    disk_fraction: float
    token_recall_at_1: float
    token_recall_at_5: float
    token_recall_at_50: float


def _token_recall(
    queries: np.ndarray, tokens_full: np.ndarray, kept_idx: np.ndarray, k: int
) -> float:
    """Fraction of true top-K (q, t) NN pairs whose retrieved token survived
    the prune. K is capped to ``min(k, n_tokens)``.

    Implementation note: we materialize one ``(n_queries, n_tokens_full)``
    fp32 matrix once for the unpruned reference and reuse it across all
    rates and K values. Cost on the 8K-token sample is ~4 MB.
    """
    n_q, n_t = queries.shape[0], tokens_full.shape[0]
    if n_t == 0:
        return 0.0
    k_eff = min(k, n_t)
    sims = queries @ tokens_full.T
    true_top = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
    kept_set = set(int(i) for i in kept_idx)
    hits = 0
    total = 0
    for i in range(n_q):
        for j in true_top[i]:
            total += 1
            if int(j) in kept_set:
                hits += 1
    return hits / total if total > 0 else 0.0


def _make_doc_offsets(n_tokens: int, n_docs: int) -> np.ndarray:
    """Synthesize evenly-spaced document boundaries; the per-doc floor in
    ``SalienceConfig.min_tokens_per_doc`` then enforces realistic shapes.
    """
    sizes = np.full(n_docs, n_tokens // n_docs, dtype=np.int64)
    sizes[: n_tokens % n_docs] += 1
    offsets = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return offsets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-path", type=Path,
        default=Path("research/low_bit_roq/tests/fixtures/token_sample_1m.npy"),
    )
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--n-queries", type=int, default=128)
    parser.add_argument("--n-docs", type=int, default=80)
    parser.add_argument("--min-tokens-per-doc", type=int, default=4)
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--signal", default="norm",
                        choices=["norm", "idf", "attention_mass"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("research/low_bit_roq/reports/a4_salience.json"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.signal != "norm":
        log.error("signal=%s requires production-side metadata; only 'norm' is "
                  "available in this offline harness.", args.signal)
        return 1

    arr = np.load(str(args.sample_path), mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    take = min(args.n_tokens, arr.shape[0])
    idx = rng.choice(arr.shape[0], size=take, replace=False)
    tokens = np.array(arr[idx], dtype=np.float32, order="C")
    del arr
    q_idx = rng.choice(tokens.shape[0], size=min(args.n_queries, tokens.shape[0]), replace=False)
    queries = tokens[q_idx].copy()

    doc_offsets = _make_doc_offsets(tokens.shape[0], args.n_docs)
    sal = salience.compute_salience(tokens, signal=args.signal)

    rows: list[SalienceRow] = []
    for rate in args.rates:
        cfg = salience.SalienceConfig(
            signal=args.signal,
            prune_quantile=rate,
            min_tokens_per_doc=args.min_tokens_per_doc,
        )
        kept_tokens, kept_offsets = salience.prune_by_quantile(
            tokens, sal, doc_offsets, cfg
        )
        if rate == 0.0:
            kept_idx = np.arange(tokens.shape[0])
        else:
            threshold = float(np.quantile(sal, rate))
            kept_idx = np.flatnonzero(sal > threshold)

        rows.append(SalienceRow(
            signal=args.signal,
            prune_rate=float(rate),
            kept_tokens=int(kept_tokens.shape[0]),
            n_docs=args.n_docs,
            disk_fraction=float(kept_tokens.shape[0] / tokens.shape[0]),
            token_recall_at_1=_token_recall(queries, tokens, kept_idx, k=1),
            token_recall_at_5=_token_recall(queries, tokens, kept_idx, k=5),
            token_recall_at_50=_token_recall(queries, tokens, kept_idx, k=50),
        ))

    out = {"config": vars(args) | {"out": str(args.out), "sample_path": str(args.sample_path)},
           "rows": [asdict(r) for r in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for r in rows:
        log.info(
            "prune=%.0f%% disk=%.2f kept=%d  token_recall@1=%.3f @5=%.3f @50=%.3f",
            100 * r.prune_rate, r.disk_fraction, r.kept_tokens,
            r.token_recall_at_1, r.token_recall_at_5, r.token_recall_at_50,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
