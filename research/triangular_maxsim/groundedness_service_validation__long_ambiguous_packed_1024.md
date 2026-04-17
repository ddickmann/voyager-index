# Groundedness Service Validation

- profile: `long_ambiguous`
- model: `lightonai/GTE-ModernColBERT-v1`
- prompts: `False`
- packed raw_context chunk tokens: `1024`
- encoder token limit: `299`
- anchor count: `4`
- anchor AUROC (reverse_context): `0.0000`
- anchor AUROC (reverse_query_context): `0.0000`
- anchor AUROC (triangular): `0.0000`
- latency p50/p95 ms: `82.60` / `108.73`
- mean/max context tokens: `7847` / `7888`
- mean packed support units: `8.0`
- user-facing go/no-go: `False`

## Difficulty Summary

| bucket | count | mean reverse_context | mean triangular | mean context tokens |
|---|---:|---:|---:|---:|
| entity_swap | 2 | 0.9842 | 0.9615 | 7858 |
| grounded | 2 | 0.9444 | 0.9415 | 7816 |
| partial | 2 | 0.9494 | 0.9421 | 7867 |

## Hardest Previous Case Rerun

- note: the requested packed budget exceeds this encoder's tokenizer limit, so support windows can be truncated during encoding on this model.

- selected case: `LG3` (ungrounded)
- rationale: highest previous non-grounded reverse_context score (`0.9908`) in the earlier long-context report
- before: reverse_context `0.9908`, support_units `66`
- after packed-1024: reverse_context `0.9857`, support_units `8`
- verification per-token max abs diff: `0.00000000`
- verification scalar abs diff: `0.00000000`
- evidence mapping exact match: `True`
- top-evidence exact match: `True`

## Example Evidence

### entity_swap
- `LG3`: Construction on the Royal Arena in Copenhagen, where the 2017 European Short Course Swimming Championships will be held, broke ground on 26 July 2013.
  - notes: Date-swap near miss inside an ultra-long context; hard for embeddings and easy for a human to miss.
  - token `Ġ2017` -> `Ġ2017` score `0.9906`
  - token `Ġ26` -> `ae` score `0.9902`
  - token `Ġ2013` -> `ĠThe` score `0.9895`
- `LG4`: Treg-cell-specific deletion of integrin αvβ6 did not result in a spontaneous inflammatory phenotype in the studied mice.
  - notes: Single-character entity swap inside a long scientific context block.
  - token `6` -> `ĠÎ±` score `0.9977`
  - token `Î²` -> `6` score `0.9527`
  - token `Ġresult` -> `8` score `0.9965`

### grounded
- `LG1`: Teardrops, the second single from George Harrison's album Somewhere in England, was released in the United States on 20 July 1981.
  - notes: Long distractor-heavy context with the George Harrison evidence paragraph buried near the middle.
  - token `Ġ20` -> `Ġat` score `0.9850`
  - token `Ġ1981` -> `Ġis` score `0.9840`
  - token `ĠStates` -> `Ġis` score `0.9895`
- `LG2`: Aptamer-functionalized lipid nanoparticles can target specific cell types such as osteoblasts, as demonstrated with the CH6 aptamer-functionalized LNP system.
  - notes: Long biomedical context where the relevant aptamer paragraph must survive thousands of unrelated tokens.
  - token `6` -> `X` score `0.9060`
  - token `Ġdemonstrated` -> `Ġof` score `0.9945`
  - token `Ġsuch` -> `Ġresponses` score `0.9939`

### partial
- `LG5`: SHP2 signal-deficient knockin mice display lymphadenopathy and splenomegaly, and were the first transgenic model used in human clinical trials for lupus nephritis.
  - notes: First clause is supported, second clause is a plausible biomedical extrapolation hidden inside a long context.
  - token `2` -> `Ġhelper` score `0.9079`
  - token `Ġused` -> `Ġa` score `0.9941`
  - token `Ġtrials` -> `Ġlupus` score `0.9917`
- `LG6`: Be Quick '28 is a football club from Zwolle in the province of Overijssel, and it currently plays in the Dutch Eredivisie top flight.
  - notes: Supported province fact mixed with an unsupported league claim under heavy distractor load.
  - token `28` -> `rops` score `0.9781`
  - token `Ġfootball` -> `]` score `0.9947`
  - token `Ġcurrently` -> `Ġplay` score `0.9933`

## Per-Case Scores

| id | label | subcategory | context_tokens | support_units | reverse_context | reverse_query_context | triangular | latency_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LG1 | grounded | - | 7808 | 8 | 0.9470 | 0.9564 | 0.9433 | 84.11 |
| LG2 | grounded | - | 7824 | 8 | 0.9417 | 0.9545 | 0.9397 | 65.83 |
| LG3 | ungrounded | entity_swap | 7828 | 8 | 0.9857 | 0.9867 | 0.9606 | 81.09 |
| LG4 | ungrounded | entity_swap | 7888 | 8 | 0.9826 | 0.9826 | 0.9624 | 108.11 |
| LG5 | ambiguous | partial | 7884 | 8 | 0.9515 | 0.9578 | 0.9439 | 108.93 |
| LG6 | ambiguous | partial | 7850 | 8 | 0.9473 | 0.9509 | 0.9402 | 27.40 |
