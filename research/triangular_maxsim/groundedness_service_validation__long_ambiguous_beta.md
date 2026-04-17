# Groundedness Service Validation

- profile: `long_ambiguous`
- model: `lightonai/GTE-ModernColBERT-v1`
- prompts: `False`
- anchor count: `4`
- anchor AUROC (reverse_context): `0.7500`
- anchor AUROC (reverse_query_context): `0.7500`
- anchor AUROC (triangular): `0.5000`
- latency p50/p95 ms: `8.09` / `14.21`
- mean/max context tokens: `7847` / `7888`
- user-facing go/no-go: `False`

## Difficulty Summary

| bucket | count | mean reverse_context | mean triangular | mean context tokens |
|---|---:|---:|---:|---:|
| entity_swap | 2 | 0.9875 | 0.9623 | 7858 |
| grounded | 2 | 0.9909 | 0.9618 | 7816 |
| partial | 2 | 0.9835 | 0.9562 | 7867 |

## Example Evidence

### entity_swap
- `LG3`: Construction on the Royal Arena in Copenhagen, where the 2017 European Short Course Swimming Championships will be held, broke ground on 26 July 2013.
  - notes: Date-swap near miss inside an ultra-long context; hard for embeddings and easy for a human to miss.
  - token `tok_19` -> `tok_76` score `0.9979`
  - token `tok_18` -> `tok_67` score `0.9976`
  - token `tok_3` -> `tok_26` score `0.9971`
- `LG4`: Treg-cell-specific deletion of integrin αvβ6 did not result in a spontaneous inflammatory phenotype in the studied mice.
  - notes: Single-character entity swap inside a long scientific context block.
  - token `tok_16` -> `tok_83` score `0.9973`
  - token `tok_13` -> `tok_80` score `0.9971`
  - token `tok_22` -> `tok_28` score `0.9968`

### grounded
- `LG1`: Teardrops, the second single from George Harrison's album Somewhere in England, was released in the United States on 20 July 1981.
  - notes: Long distractor-heavy context with the George Harrison evidence paragraph buried near the middle.
  - token `tok_17` -> `tok_44` score `0.9984`
  - token `tok_5` -> `tok_22` score `0.9981`
  - token `tok_20` -> `tok_34` score `0.9981`
- `LG2`: Aptamer-functionalized lipid nanoparticles can target specific cell types such as osteoblasts, as demonstrated with the CH6 aptamer-functionalized LNP system.
  - notes: Long biomedical context where the relevant aptamer paragraph must survive thousands of unrelated tokens.
  - token `tok_18` -> `tok_85` score `0.9986`
  - token `tok_6` -> `tok_12` score `0.9960`
  - token `tok_20` -> `tok_134` score `0.9960`

### partial
- `LG5`: SHP2 signal-deficient knockin mice display lymphadenopathy and splenomegaly, and were the first transgenic model used in human clinical trials for lupus nephritis.
  - notes: First clause is supported, second clause is a plausible biomedical extrapolation hidden inside a long context.
  - token `tok_19` -> `tok_48` score `0.9982`
  - token `tok_20` -> `tok_91` score `0.9969`
  - token `tok_21` -> `tok_36` score `0.9963`
- `LG6`: Be Quick '28 is a football club from Zwolle in the province of Overijssel, and it currently plays in the Dutch Eredivisie top flight.
  - notes: Supported province fact mixed with an unsupported league claim under heavy distractor load.
  - token `tok_6` -> `tok_50` score `0.9991`
  - token `tok_7` -> `tok_51` score `0.9989`
  - token `tok_23` -> `tok_24` score `0.9987`

## Per-Case Scores

| id | label | subcategory | context_tokens | support_units | reverse_context | reverse_query_context | triangular | latency_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LG1 | grounded | - | 7808 | 67 | 0.9944 | 0.9944 | 0.9646 | 6.99 |
| LG2 | grounded | - | 7824 | 69 | 0.9874 | 0.9874 | 0.9591 | 7.85 |
| LG3 | ungrounded | entity_swap | 7828 | 66 | 0.9908 | 0.9908 | 0.9612 | 7.20 |
| LG4 | ungrounded | entity_swap | 7888 | 67 | 0.9842 | 0.9842 | 0.9634 | 8.76 |
| LG5 | ambiguous | partial | 7884 | 67 | 0.9834 | 0.9834 | 0.9595 | 8.33 |
| LG6 | ambiguous | partial | 7850 | 66 | 0.9835 | 0.9839 | 0.9529 | 16.03 |
