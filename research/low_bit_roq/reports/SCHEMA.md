# Report JSON schema

Structured artifacts cited from `PROGRESS.md` entries. The harness writes
one of these per experiment; `progress_md.append_stub_from_report` consumes
it.

## Top-level keys

```json
{
  "id": "a1-cell-17",
  "summary": "FWHT-on x group_size=16 x 2-bit doc x 6-bit query, normalize-on",
  "config": {
    "doc_bits": 2,
    "group_size": 16,
    "query_bits": 6,
    "fwht": true,
    "normalize": true,
    "codebook": "lloyd_per_group",
    "norm_correction": "4term"
  },
  "datasets": ["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"],
  "seeds": 5,
  "baseline": "roq4",
  "metrics": [
    {
      "name": "Recall@10 (rerank)",
      "baseline": 0.521,
      "this": 0.518,
      "delta": -0.003,
      "p_value": 0.42
    }
  ],
  "artifacts": [
    "reports/a1-cell-17.json",
    "reports/a1-cell-17_per_dataset.json"
  ],
  "gate": "informs A6 / advances A6 / blocks A6 / decides C1.5 / informs nothing"
}
```

## Required keys

| key            | type          | notes                                            |
| -------------- | ------------- | ------------------------------------------------ |
| `id`           | str           | matches a todo id from the plan frontmatter     |
| `summary`      | str           | one-line summary, ≤ 80 chars                    |
| `config`       | object        | flat key=value, primitives only                 |
| `datasets`     | list[str]     | BEIR dataset names                              |
| `seeds`        | int           | number of seeds aggregated                      |
| `baseline`     | str           | name of the baseline this row was compared to   |
| `metrics`      | list[object]  | each row matches `progress_md.MetricRow`         |
| `artifacts`    | list[str]     | paths the entry should link to                  |
| `gate`         | str           | one of the strings listed above                 |

## Per-metric row

```json
{
  "name": "Recall@10 (rerank)",
  "baseline": 0.521,
  "this": 0.518,
  "delta": -0.003,
  "p_value": 0.42
}
```

`baseline` / `this` are the macro-averaged values across datasets+seeds.
`delta` is `this - baseline` (or relative when expressed as %).
`p_value` is the macro paired-bootstrap p-value vs the named baseline.

Per-dataset rows live in a sibling `*_per_dataset.json` artifact, one row per
(dataset, seed). PROGRESS.md never includes per-dataset breakdowns directly.
