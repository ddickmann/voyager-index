# Groundedness Service Validation

- profile: `long_ambiguous`
- model: `lightonai/GTE-ModernColBERT-v1`
- prompts: `False`
- packed raw_context chunk tokens: `299`
- encoder token limit: `299`
- anchor count: `4`
- anchor AUROC (reverse_context): `1.0000`
- anchor AUROC (reverse_query_context): `1.0000`
- anchor AUROC (triangular): `0.5000`
- latency p50/p95 ms: `472.52` / `511.89`
- mean/max context tokens: `7847` / `7888`
- mean packed support units: `28.0`
- user-facing go/no-go: `False`

## Difficulty Summary

| bucket | count | mean reverse_context | mean triangular | mean context tokens |
|---|---:|---:|---:|---:|
| entity_swap | 2 | 0.9846 | 0.9597 | 7858 |
| grounded | 2 | 0.9874 | 0.9574 | 7816 |
| partial | 2 | 0.9804 | 0.9507 | 7867 |

## Hardest Previous Case Rerun

- selected case: `LG3` (ungrounded)
- rationale: highest previous non-grounded reverse_context score (`0.9908`) in the earlier long-context report
- before: reverse_context `0.9908`, support_units `66`
- after packed-299: reverse_context `0.9831`, support_units `28`
- verification per-token max abs diff: `0.00000000`
- verification scalar abs diff: `0.00000000`
- evidence mapping exact match: `True`
- top-evidence exact match: `True`

## Example Evidence

### entity_swap
- `LG3`: Construction on the Royal Arena in Copenhagen, where the 2017 European Short Course Swimming Championships will be held, broke ground on 26 July 2013.
  - notes: Date-swap near miss inside an ultra-long context; hard for embeddings and easy for a human to miss.
  - token `Ġ26` -> `rops` score `0.9903`
  - token `Ġ2013` -> `ĠSw` score `0.9899`
  - token `Ġ2017` -> `Ġ26` score `0.9876`
- `LG4`: Treg-cell-specific deletion of integrin αvβ6 did not result in a spontaneous inflammatory phenotype in the studied mice.
  - notes: Single-character entity swap inside a long scientific context block.
  - token `6` -> `8` score `0.9984`
  - token `Î²` -> `ol` score `0.9533`
  - token `Ġresult` -> `Ġresult` score `0.9970`

### grounded
- `LG1`: Teardrops, the second single from George Harrison's album Somewhere in England, was released in the United States on 20 July 1981.
  - notes: Long distractor-heavy context with the George Harrison evidence paragraph buried near the middle.
  - token `Ġ20` -> `Ġsong` score `0.9946`
  - token `Ġ1981` -> `aff` score `0.9885`
  - token `ĠStates` -> `George` score `0.9957`
- `LG2`: Aptamer-functionalized lipid nanoparticles can target specific cell types such as osteoblasts, as demonstrated with the CH6 aptamer-functionalized LNP system.
  - notes: Long biomedical context where the relevant aptamer paragraph must survive thousands of unrelated tokens.
  - token `6` -> `Ġboth` score `0.9911`
  - token `Ġdemonstrated` -> `le` score `0.9951`
  - token `ized` -> `ĠA` score `0.9944`

### partial
- `LG5`: SHP2 signal-deficient knockin mice display lymphadenopathy and splenomegaly, and were the first transgenic model used in human clinical trials for lupus nephritis.
  - notes: First clause is supported, second clause is a plausible biomedical extrapolation hidden inside a long context.
  - token `2` -> `2` score `0.9894`
  - token `Ġused` -> `Ġstress` score `0.9950`
  - token `Ġtrials` -> `Ġ2` score `0.9935`
- `LG6`: Be Quick '28 is a football club from Zwolle in the province of Overijssel, and it currently plays in the Dutch Eredivisie top flight.
  - notes: Supported province fact mixed with an unsupported league claim under heavy distractor load.
  - token `28` -> `Ġmunicipality` score `0.9887`
  - token `Ġfootball` -> `Ġprovince` score `0.9964`
  - token `ĠZ` -> `ij` score `0.9957`

## Per-Case Scores

| id | label | subcategory | context_tokens | support_units | reverse_context | reverse_query_context | triangular | latency_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LG1 | grounded | - | 7808 | 28 | 0.9866 | 0.9866 | 0.9588 | 507.49 |
| LG2 | grounded | - | 7824 | 28 | 0.9882 | 0.9882 | 0.9559 | 467.41 |
| LG3 | ungrounded | entity_swap | 7828 | 28 | 0.9831 | 0.9843 | 0.9551 | 477.63 |
| LG4 | ungrounded | entity_swap | 7888 | 28 | 0.9861 | 0.9861 | 0.9642 | 382.89 |
| LG5 | ambiguous | partial | 7884 | 28 | 0.9809 | 0.9809 | 0.9541 | 513.35 |
| LG6 | ambiguous | partial | 7850 | 28 | 0.9800 | 0.9800 | 0.9473 | 447.29 |
