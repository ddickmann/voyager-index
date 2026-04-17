# Groundedness Phase E–J evaluation harness

This folder hosts the deterministic minimal-pair fixture, the external
benchmark adapters, and the evaluation harness used to validate the
`voyager-index` Groundedness Tracker (Beta). Phase J adds three new
"HARD" minimal-pair families (`hard_compound_facts`, `hard_structured`,
`hard_dialogue_distributed`), tightens exit criteria, and extends the
report schema with `structured_diagnostics`, risk bands, and semantic
entropy peers.

## Quick run (defaults, no external data)

```bash
cd voyager-index
# Lane A: dense + literal guardrails, no NLI
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 \
  --out research/triangular_maxsim/reports/phase_j_no_nli.json

# Lane B: dense + literal + NLI peer (reranker + atomic + multi-premise)
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 \
  --enable-nli \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --concat-premises --atomic-claims \
  --out research/triangular_maxsim/reports/phase_j_nli.json

# Lane C: NLI peer + semantic entropy (synthetic 4 samples)
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 15 \
  --enable-nli \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --concat-premises --atomic-claims \
  --enable-semantic-entropy --semantic-entropy-samples 4 \
  --out research/triangular_maxsim/reports/phase_j_nli_sem.json
```

Each report carries:

- per-stratum paired ranking accuracy on 200 (or 150) minimal pairs
  across 10 strata, with bootstrap 95% confidence intervals
- encode and score latency split (`encode_p50/p95_ms`,
  `score_p50/p95_ms`, `latency_p50/p95_ms`)
- per-criterion verdict labels against `PREREGISTERED_TARGETS`
- a single human-readable `headline_verdict` string
- `all_targets_met` boolean

## Adding the external benchmarks

The harness will pick up real-world benchmarks when the corresponding
data directories are exported via environment variables. None of the
external datasets are bundled.

### RAGTruth

```bash
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git external_data/RAGTruth
```

The released layout is `dataset/source_info.jsonl` plus
`dataset/response.jsonl` keyed by `source_id`. The voyager loader
expects a per-stratum `qa/test.jsonl`, `summarization/test.jsonl`, and
`data2text/test.jsonl` in the configured base directory, with each row
carrying `source_info`, `response`, and span-level `labels`. Convert
once with the snippet at the bottom of this file (`convert_ragtruth.py`)
and point the harness at the converted layout:

```bash
export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=$PWD/external_data/RAGTruth/voyager_layout
```

### HaluEval

```bash
git clone --depth 1 https://github.com/RUCAIBox/HaluEval.git external_data/HaluEval
ln -sf qa_data.json            external_data/HaluEval/data/qa_data.jsonl
ln -sf summarization_data.json external_data/HaluEval/data/summarization_data.jsonl
ln -sf dialogue_data.json      external_data/HaluEval/data/dialogue_data.jsonl
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=$PWD/external_data/HaluEval/data
```

### FActScore

Follow the upstream instructions at
<https://github.com/shmsw25/FActScore> to assemble
`biographies.jsonl` (or `factscore.jsonl`), then:

```bash
export VOYAGER_GROUNDEDNESS_FACTSCORE_DIR=/path/to/factscore
```

## Real-world run (RAGTruth + HaluEval, three lanes)

```bash
export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=$PWD/research/triangular_maxsim/external_data/RAGTruth/voyager_layout
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=$PWD/research/triangular_maxsim/external_data/HaluEval/data

# No-NLI lane
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 --max-external-per-stratum 20 \
  --out research/triangular_maxsim/reports/phase_j_no_nli.json

# NLI lane (reranker + atomic + multi-premise)
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 --max-external-per-stratum 20 \
  --enable-nli --reranker-model BAAI/bge-reranker-v2-m3 \
  --concat-premises --atomic-claims \
  --out research/triangular_maxsim/reports/phase_j_nli.json

# NLI + semantic entropy lane
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 15 --max-external-per-stratum 10 \
  --enable-nli --reranker-model BAAI/bge-reranker-v2-m3 \
  --concat-premises --atomic-claims \
  --enable-semantic-entropy --semantic-entropy-samples 4 \
  --out research/triangular_maxsim/reports/phase_j_nli_sem.json
```

## Latest published results — Phase J (single A5000, batch 1)

| Lane                                     | Internal lex | Internal sem | Internal partial | RAGTruth macro F1 | HaluEval QA F1 | Latency p95 |
|------------------------------------------|-------------:|-------------:|-----------------:|------------------:|---------------:|------------:|
| Dense + literal only                     |         0.80 |         0.93 |             0.95 |              0.48 |           0.75 |       92 ms |
| Dense + literal + NLI (reranker+atomic)  |         0.99 |         1.00 |             1.00 |              0.49 |       **0.90** |      102 ms |
| + Semantic entropy (synthetic peers)     |         0.98 |         1.00 |             1.00 |          **0.60** |           0.80 |      125 ms |
| Pre-registered exit (Phase J)            |      ≥ 0.80  |      ≥ 0.70  |          ≥ 0.65  |          ≥ 0.55   |        ≥ 0.75  | ≤ 400 ms    |

The NLI lane lifts HaluEval QA F1 from `0.75` to `0.90` (+15 absolute
points) and sweeps the internal minimal pairs at 1.00 across 9 of 10
strata (hard_structured lands at 0.80). The NLI + semantic-entropy lane
is what pushes **RAGTruth macro F1 to `0.60`** (pass ≥ 0.55). All 10
strata now include the new HARD families (`hard_compound_facts`,
`hard_structured`, `hard_dialogue_distributed`); paired accuracy on
those in the NLI lane is `1.00 / 0.80 / 1.00`. HaluEval dialogue
(F1 `0.40`) and RAGTruth data2text (F1 `0.25`) remain the structurally
hard tails — see `groundedness_algorithm_audit.md` for the full
diagnostics.

## Committed reports

The canonical reports for Phase J live under
`research/triangular_maxsim/reports/`:

- `phase_j_no_nli.json` — dense + literal only, 200 minimal pairs,
  60 external RAGTruth + 60 external HaluEval samples
- `phase_j_nli.json` — NLI peer with reranker, atomic decomposition,
  multi-premise concatenation, and structured-source verification
- `phase_j_nli_sem.json` — NLI peer + semantic-entropy with 4 synthetic
  verification samples per case

## RAGTruth conversion script

```python
import json
from collections import defaultdict
from pathlib import Path

base = Path("external_data/RAGTruth")
sources = {}
for line in (base / "dataset" / "source_info.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    sources[str(obj["source_id"])] = obj

TASK = {"QA": "qa", "Summary": "summarization", "Data2txt": "data2text"}
bucket: dict = defaultdict(list)
for line in (base / "dataset" / "response.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if rec.get("split") != "test":
        continue
    src = sources.get(str(rec["source_id"]))
    if not src:
        continue
    stratum = TASK.get(src.get("task_type"))
    if not stratum:
        continue
    src_info = src.get("source_info")
    if isinstance(src_info, dict):
        parts = []
        if src_info.get("question"):
            parts.append("Question: " + str(src_info["question"]).strip())
        psgs = src_info.get("passages") or src_info.get("context") or []
        if isinstance(psgs, list):
            parts.extend(str(p).strip() for p in psgs)
        elif isinstance(psgs, str):
            parts.append(psgs.strip())
        context = "\n\n".join(p for p in parts if p)
        query = src_info.get("question")
    else:
        context = str(src_info or "")
        query = None
    bucket[stratum].append({
        "id": rec["id"],
        "source_id": str(rec["source_id"]),
        "source_info": context,
        "response": rec.get("response", ""),
        "labels": rec.get("labels", []),
        "query": query,
        "model": rec.get("model"),
        "task_type": src.get("task_type"),
    })

for stratum, rows in bucket.items():
    out = base / "voyager_layout" / stratum / "test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(stratum, len(rows), "->", out)
```
