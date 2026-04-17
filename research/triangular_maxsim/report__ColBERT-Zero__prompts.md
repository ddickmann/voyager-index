# Triangular MaxSim Falsification Report

Run of `research/triangular_maxsim/experiment.py` on the 20 hand-crafted SciFact + HotpotQA cases. Each (Q, C, R) triple is embedded with `lightonai/ColBERT-Zero` (use_prompts=True) and scored with the Triton kernel in `voyager_index/_internal/kernels/triton_triangular_maxsim.py`.

## Verdict

> HYPOTHESIS FALSIFIED, but naive Reverse MaxSim suffices. Naive R->(Q union C) and/or R->C MaxSim cleanly separate grounded from ungrounded responses (AUROC >= 0.90), so embedding-only Reverse MaxSim CAN score groundedness on this anchor set. However, the Triangular `min(s_RC, a_j)` gating does NOT outperform the naive baseline here -- in fact it compresses the dynamic range and reduces separation. The triangular structure is not earning its keep on this dataset/encoder.

## Anchor metrics (grounded=10 cases? actually 5+5)

| metric | value |
|---|---|
| `AUROC_G_tri` | 0.4800 |
| `AUROC_G_naive_QC` | 1.0000 |
| `AUROC_G_rc` | 1.0000 |
| `AUROC_best` | 1.0000 |
| `mean_G_tri_grounded` | 0.7808 |
| `mean_G_tri_ungrounded` | 0.7841 |
| `G_tri_separation_margin` | -0.0033 |
| `best_thr_G_tri` | 0.7647 |
| `best_acc_G_tri` | 0.6000 |
| `best_thr_G_naive_QC` | 0.9564 |
| `best_acc_G_naive_QC` | 1.0000 |
| `check1_global_separability_pass` | True |
| `check2_triangular_beats_naive_pass` | False |

## Per-case scores

| id | label | sub | G_tri | G_naive_QC | G_rc | echo | GC | kernel-vs-ref err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| G1 | grounded |  | 0.7531 | 0.9564 | 0.9564 | 0.7158 | 0.7395 | 5.96e-08 |
| G2 | grounded |  | 0.7647 | 0.9778 | 0.9778 | 0.6736 | 0.7172 | 1.19e-07 |
| G3 | grounded |  | 0.8166 | 0.9796 | 0.9796 | 0.7562 | 0.7384 | 5.96e-08 |
| G4 | grounded |  | 0.8027 | 0.9788 | 0.9788 | 0.7444 | 0.7468 | 1.19e-07 |
| G5 | grounded |  | 0.7670 | 0.9577 | 0.9577 | 0.6880 | 0.6596 | 1.19e-07 |
| U1 | ungrounded |  | 0.7610 | 0.9040 | 0.9040 | 0.7066 | 0.7677 | 5.96e-08 |
| U2 | ungrounded |  | 0.8153 | 0.9358 | 0.9358 | 0.7531 | 0.7240 | 1.19e-07 |
| U3 | ungrounded |  | 0.7810 | 0.9123 | 0.9123 | 0.7183 | 0.7702 | 5.96e-08 |
| U4 | ungrounded |  | 0.7604 | 0.8756 | 0.8712 | 0.7093 | 0.7476 | 1.19e-07 |
| U5 | ungrounded |  | 0.8032 | 0.9035 | 0.9028 | 0.7557 | 0.7221 | 1.19e-07 |
| A1 | ambiguous | prompt_echo | 0.7258 | 0.8853 | 0.8491 | 0.7951 | 0.7235 | 1.19e-07 |
| A2 | ambiguous | prompt_echo | 0.8128 | 0.9373 | 0.9322 | 0.8256 | 0.7650 | 1.19e-07 |
| A3 | ambiguous | partial | 0.7585 | 0.9306 | 0.9306 | 0.6752 | 0.7371 | 1.19e-07 |
| A4 | ambiguous | partial | 0.7496 | 0.9307 | 0.9262 | 0.6936 | 0.7686 | 1.19e-07 |
| A5 | ambiguous | parametric | 0.7484 | 0.8235 | 0.8160 | 0.6556 | 0.7090 | 5.96e-08 |
| A6 | ambiguous | parametric | 0.7208 | 0.9164 | 0.9164 | 0.5925 | 0.6043 | 1.19e-07 |
| A7 | ambiguous | negation_flip | 0.8057 | 0.9636 | 0.9636 | 0.7534 | 0.7487 | 1.19e-07 |
| A8 | ambiguous | negation_flip | 0.7853 | 0.9483 | 0.9483 | 0.7215 | 0.7359 | 1.19e-07 |
| A9 | ambiguous | entity_swap | 0.7781 | 0.9258 | 0.9258 | 0.7019 | 0.7393 | 5.96e-08 |
| A10 | ambiguous | entity_swap | 0.7819 | 0.9627 | 0.9627 | 0.7062 | 0.7389 | 1.19e-07 |

## Anchor cases - sorted by G_tri

| rank | id | label | G_tri | G_naive_QC | G_rc |
|---:|---|---|---:|---:|---:|
| 1 | G3 `OK` | grounded | 0.8166 | 0.9796 | 0.9796 |
| 2 | U2 `X ` | ungrounded | 0.8153 | 0.9358 | 0.9358 |
| 3 | U5 `X ` | ungrounded | 0.8032 | 0.9035 | 0.9028 |
| 4 | G4 `OK` | grounded | 0.8027 | 0.9788 | 0.9788 |
| 5 | U3 `X ` | ungrounded | 0.7810 | 0.9123 | 0.9123 |
| 6 | G5 `OK` | grounded | 0.7670 | 0.9577 | 0.9577 |
| 7 | G2 `OK` | grounded | 0.7647 | 0.9778 | 0.9778 |
| 8 | U1 `X ` | ungrounded | 0.7610 | 0.9040 | 0.9040 |
| 9 | U4 `X ` | ungrounded | 0.7604 | 0.8756 | 0.8712 |
| 10 | G1 `OK` | grounded | 0.7531 | 0.9564 | 0.9564 |

## Ambiguous cases - per-subcategory diagnostics

### prompt_echo

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A1 | 0.7258 | 0.7951 | 0.8853 | 0.8491 | `5`->`Ġ(` (0.76); `Ġcentral`->`IG` (0.79); `ase`->`.` (0.79) |
| A2 | 0.8128 | 0.8256 | 0.9373 | 0.9322 | `Ġfather`->`Ġ19` (0.84); `'s`->`Ġ19` (0.84); `re`->`;` (0.84) |

### partial

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A3 | 0.7585 | 0.6752 | 0.9306 | 0.9306 | `2`->`Dis` (0.77); `Ġsignal`->`3` (0.85); `deficient`->`2` (0.83) |
| A4 | 0.7496 | 0.6936 | 0.9307 | 0.9262 | `28`->`000` (0.85); `Ġfootball`->`ĠQuick` (0.88); `ol`->`Ġprovince` (0.79) |

### parametric

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A5 | 0.7484 | 0.6556 | 0.8235 | 0.8160 | `1`->`ĠRNA` (0.65); `Ġsignaling`->`micro` (0.79); `ĠS`->`micro` (0.79) |
| A6 | 0.7208 | 0.5925 | 0.9164 | 0.9164 | `Kim`->`[CLS]` (0.80); `ĠAmerican`->`[CLS]` (0.80); `ĠLiverpool`->`[CLS]` (0.80) |

### negation_flip

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A7 | 0.8057 | 0.7534 | 0.9636 | 0.9636 | `Ġlupus`->`BAS` (0.82); `Ġproduction`->`BAS` (0.82); `Ġsuppressing`->`BAS` (0.82) |
| A8 | 0.7853 | 0.7215 | 0.9483 | 0.9483 | `Ġlongtime`->`Ġ2007` (0.84); `Ġnever`->`[CLS]` (0.83); `Ġprot`->`[CLS]` (0.83) |

### entity_swap

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A9 | 0.7781 | 0.7019 | 0.9258 | 0.9258 | `Î²`->`-` (0.81); `6`->`Ġin` (0.72); `v`->`-` (0.81) |
| A10 | 0.7819 | 0.7062 | 0.9627 | 0.9627 | `Ġ2017`->`Royal` (0.81); `Ġ26`->`[CLS]` (0.78); `Ġ2013`->`Ġfor` (0.75) |

## Case detail and evidence pointers

### G1 - grounded - SF:839
- Q: Nanoparticles can be targeted against specific cell types by incorporating aptamers into lipid nanoparticles.
- R: Aptamer-functionalized lipid nanoparticles can target specific cell types such as osteoblasts, as demonstrated with the CH6 aptamer-functionalized LNP system.
- G_tri = **0.7531**, G_naive_QC = 0.9564, G_rc = 0.9564, echo = 0.7158
- top response tokens -> grounding context tokens:
    - `6` (resp t=24, w=1.5) -> `ĠCH` (ctx j=71, g=0.628)
    - `amer` (resp t=26, w=1.0) -> `Ġwe` (ctx j=66, g=0.833)
    - `ized` (resp t=6, w=1.0) -> `Ġwe` (ctx j=66, g=0.833)
    - `functional` (resp t=5, w=1.0) -> `Ġwe` (ctx j=66, g=0.833)
    - `pt` (resp t=2, w=1.0) -> `pt` (ctx j=2, g=0.793)
- notes: Close paraphrase of opening sentences of the context.

### G2 - grounded - SF:536
- Q: Hypocretin neurones induce panic-prone state in rats.
- R: Orexin (hypocretin) neurons in the dorsomedial-perifornical hypothalamus play a key role in panic anxiety in a rat model of panic disorder.
- G_tri = **0.7647**, G_naive_QC = 0.9778, G_rc = 0.9778, echo = 0.6736
- top response tokens -> grounding context tokens:
    - `Ġdisorder` (resp t=34, w=1.0) -> `Ġpanic` (ctx j=40, g=0.834)
    - `Ġanxiety` (resp t=27, w=1.0) -> `Ġpanic` (ctx j=40, g=0.834)
    - `Ġpanic` (resp t=33, w=1.0) -> `Ġattacks` (ctx j=33, g=0.821)
    - `O` (resp t=1, w=1.0) -> `A` (ctx j=1, g=0.802)
    - `Ġrat` (resp t=30, w=1.0) -> `A` (ctx j=1, g=0.802)
- notes: Synthesis of two adjacent context sentences; entities preserved.

### G3 - grounded - HP:5ae48aff
- Q: When was George Harrison's second single from his album, that was made with rising conflicts with Warner Bros., released in the United States?
- R: Teardrops, the second single from George Harrison's album Somewhere in England, was released in the United States on 20 July 1981.
- G_tri = **0.8166**, G_naive_QC = 0.9796, G_rc = 0.9796, echo = 0.7562
- top response tokens -> grounding context tokens:
    - `Ġ20` (resp t=25, w=1.5) -> `Ġduring` (ctx j=113, g=0.840)
    - `Ġ1981` (resp t=27, w=1.5) -> `Ġ20` (ctx j=23, g=0.809)
    - `ĠUnited` (resp t=22, w=1.0) -> `[CLS]` (ctx j=0, g=0.873)
    - `Te` (resp t=1, w=1.0) -> `[CLS]` (ctx j=0, g=0.862)
    - `Ġalbum` (resp t=12, w=1.0) -> `Ġ"` (ctx j=9, g=0.850)
- notes: Direct factual answer; every claim has a context source token.

### G4 - grounded - HP:5ab2a54b
- Q: What is the size of the hydroelectric dam that will inundate part of Bui National Park?
- R: The Bui Dam, a 400 MW hydroelectric project on the Black Volta in Ghana, will inundate part of Bui National Park with its reservoir.
- G_tri = **0.8027**, G_naive_QC = 0.9788, G_rc = 0.9788, echo = 0.7444
- top response tokens -> grounding context tokens:
    - `Ġ400` (resp t=7, w=1.5) -> `Ġa` (ctx j=10, g=0.812)
    - `ui` (resp t=27, w=1.0) -> `ĠThe` (ctx j=5, g=0.887)
    - `ĠNational` (resp t=28, w=1.0) -> `ĠThe` (ctx j=5, g=0.887)
    - `ĠB` (resp t=26, w=1.0) -> `[CLS]` (ctx j=0, g=0.851)
    - `Ġreservoir` (resp t=32, w=1.0) -> `[CLS]` (ctx j=0, g=0.851)
- notes: Two-hop answer (dam size + which dam) with both hops in context.

### G5 - grounded - HP:5abce87e
- Q: What role in the military did the discoverer of the elliptical galaxy NGC 3610 have before reaching Britain?
- R: Before migrating to Great Britain in 1757, William Herschel served in the Military Band of Hanover, following his father into that role.
- G_tri = **0.7670**, G_naive_QC = 0.9577, G_rc = 0.9577, echo = 0.6880
- top response tokens -> grounding context tokens:
    - `57` (resp t=8, w=1.5) -> `Ġfather` (ctx j=49, g=0.763)
    - `Ġ17` (resp t=7, w=1.5) -> `Ġmigrating` (ctx j=59, g=0.758)
    - `Ġrole` (resp t=27, w=1.0) -> `[CLS]` (ctx j=0, g=0.830)
    - `ĠBand` (resp t=17, w=1.0) -> `[CLS]` (ctx j=0, g=0.830)
    - `Ġfather` (resp t=24, w=1.0) -> `[CLS]` (ctx j=0, g=0.830)
- notes: Multi-hop (NGC 3610 -> Herschel -> military role) all anchored in context.

### U1 - ungrounded - SF:957
- Q: Podocytes are motile and migrate in the presence of injury.
- R: Podocytes are entirely stationary cells that never migrate, even after injury, and serial imaging confirmed they remain locked to the glomerular basement membrane.
- G_tri = **0.7610**, G_naive_QC = 0.9040, G_rc = 0.9040, echo = 0.7066
- top response tokens -> grounding context tokens:
    - `Ġentirely` (resp t=4, w=1.0) -> `Ġare` (ctx j=28, g=0.873)
    - `Ġstationary` (resp t=5, w=1.0) -> `Ġcritical` (ctx j=29, g=0.865)
    - `Ġmigrate` (resp t=9, w=1.0) -> `ocin` (ctx j=99, g=0.821)
    - `ocytes` (resp t=2, w=1.0) -> `Ġof` (ctx j=83, g=0.815)
    - `Ġremain` (resp t=20, w=1.0) -> `ocin` (ctx j=99, g=0.794)
- notes: Direct contradiction; the context explicitly shows podocyte motility and migration.

### U2 - ungrounded - SF:362
- Q: During the primary early antibody response activated B cells migrate toward the inner-and outer paracortical areas where oxysterol accumulation is generated by stromal cells.
- R: During the early antibody response, activated B cells are confined to the bone marrow and never enter the lymph node follicles, because CXCL13 actively repels them.
- G_tri = **0.8153**, G_naive_QC = 0.9358, G_rc = 0.9358, echo = 0.7531
- top response tokens -> grounding context tokens:
    - `13` (resp t=26, w=1.5) -> `Ġplasma` (ctx j=64, g=0.752)
    - `Ġenter` (resp t=18, w=1.0) -> `[CLS]` (ctx j=0, g=0.876)
    - `Ġmarrow` (resp t=15, w=1.0) -> `[CLS]` (ctx j=0, g=0.876)
    - `Ġresponse` (resp t=5, w=1.0) -> `[CLS]` (ctx j=0, g=0.873)
    - `ĠCXCL` (resp t=25, w=1.0) -> `[CLS]` (ctx j=0, g=0.869)
- notes: Inverts the role of CXCL13 and contradicts the entire premise of B cell migration.

### U3 - ungrounded - SF:1216
- Q: The extracellular domain of TMEM27 is cleaved in human beta cells.
- R: TMEM27 is found exclusively inside the nucleus of beta cells, is never glycosylated, and its extracellular domain remains uncleaved throughout the cell's lifetime.
- G_tri = **0.7810**, G_naive_QC = 0.9123, G_rc = 0.9123, echo = 0.7183
- top response tokens -> grounding context tokens:
    - `27` (resp t=3, w=1.5) -> `Ġplasma` (ctx j=63, g=0.803)
    - `Ġexclusively` (resp t=6, w=1.0) -> `Ġcleaved` (ctx j=6, g=0.895)
    - `Ġfound` (resp t=5, w=1.0) -> `:` (ctx j=4, g=0.816)
    - `Ġnever` (resp t=15, w=1.0) -> `Ġplasma` (ctx j=63, g=0.803)
    - `Ġbeta` (resp t=11, w=1.0) -> `Ġplasma` (ctx j=63, g=0.803)
- notes: Contradicts three separate explicit facts about TMEM27 in the context.

### U4 - ungrounded - HP:5ab2da13
- Q: Who is the brother of the American transgender actor known for his work on MTV's Faking It?
- R: Elliot Fletcher's brother Marcus Fletcher is also a transgender actor, best known for his lead role in the HBO drama Euphoria.
- G_tri = **0.7604**, G_naive_QC = 0.8756, G_rc = 0.8712, echo = 0.7093
- top response tokens -> grounding context tokens:
    - `ĠFletcher` (resp t=3, w=1.0) -> `[CLS]` (ctx j=0, g=0.856)
    - `Ġactor` (resp t=12, w=1.0) -> `ĠAmerican` (ctx j=16, g=0.845)
    - `ĠMarcus` (resp t=6, w=1.0) -> `)` (ctx j=78, g=0.841)
    - `Ġlead` (resp t=18, w=1.0) -> `)` (ctx j=78, g=0.841)
    - `Ġtransgender` (resp t=11, w=1.0) -> `)` (ctx j=78, g=0.841)
- notes: Marcus, lead role, Euphoria all invented; context says brother is Connor.

### U5 - ungrounded - HP:5a797bd9
- Q: What age was Georgia Middleman when she started singing in the seventh-most populated city in the United States?
- R: Georgia Middleman started singing at age fifteen at the honky-tonks of downtown Houston, Texas, and recorded her first album at sixteen.
- G_tri = **0.8032**, G_naive_QC = 0.9035, G_rc = 0.9028, echo = 0.7557
- top response tokens -> grounding context tokens:
    - `Ġstarted` (resp t=5, w=1.0) -> `ĠGeorgia` (ctx j=29, g=0.897)
    - `Ġsinging` (resp t=6, w=1.0) -> `Ġan` (ctx j=45, g=0.868)
    - `Ġsixteen` (resp t=29, w=1.0) -> `[CLS]` (ctx j=0, g=0.856)
    - `ky` (resp t=13, w=1.0) -> `[CLS]` (ctx j=0, g=0.832)
    - `Georg` (resp t=1, w=1.0) -> `[CLS]` (ctx j=0, g=0.831)
- notes: Wrong age (15 vs 10), wrong venue, wrong city (Houston vs San Antonio), invented album.

### A1 - ambiguous / prompt_echo - SF:1179
- Q: The PRR MDA5 has a central DExD/H RNA helicase domain.
- R: The PRR MDA5 has a central DExD/H RNA helicase domain.
- G_tri = **0.7258**, G_naive_QC = 0.8853, G_rc = 0.8491, echo = 0.7951
- top response tokens -> grounding context tokens:
    - `5` (resp t=5, w=1.5) -> `Ġ(` (ctx j=21, g=0.764)
    - `Ġcentral` (resp t=8, w=1.0) -> `IG` (ctx j=26, g=0.794)
    - `ase` (resp t=16, w=1.0) -> `.` (ctx j=54, g=0.794)
    - `ĠMDA` (resp t=4, w=1.0) -> `[CLS]` (ctx j=0, g=0.739)
    - `Ex` (resp t=10, w=1.0) -> `[CLS]` (ctx j=0, g=0.739)
- notes: Verbatim restatement of the query; the context discusses MDA5 generally but does not mention the DExD/H domain at all.

### A2 - ambiguous / prompt_echo - HP:5ae66b05
- Q: What kind of comedian was Michael McIntyre's father?
- R: Michael McIntyre's father was a comedian.
- G_tri = **0.8128**, G_naive_QC = 0.9373, G_rc = 0.9322, echo = 0.8256
- top response tokens -> grounding context tokens:
    - `Ġfather` (resp t=7, w=1.0) -> `Ġ19` (ctx j=18, g=0.839)
    - `'s` (resp t=6, w=1.0) -> `Ġ19` (ctx j=18, g=0.838)
    - `re` (resp t=5, w=1.0) -> `;` (ctx j=17, g=0.836)
    - `Michael` (resp t=1, w=1.0) -> `Ġactor` (ctx j=68, g=0.822)
    - `ĠMc` (resp t=2, w=1.0) -> `ĠFebruary` (ctx j=60, g=0.791)
- notes: Echoes query verbatim; technically the word 'comedian' is in context but the answer omits the discriminating fact ('Canadian').

### A3 - ambiguous / partial - SF:729
- Q: Lymphadenopathy is observed in knockin mouse lacking the SHP-2 MAPK pathway.
- R: SHP2 signal-deficient knockin mice display lymphadenopathy and splenomegaly, and were the first transgenic model used in human clinical trials for lupus nephritis.
- G_tri = **0.7585**, G_naive_QC = 0.9306, G_rc = 0.9306, echo = 0.6752
- top response tokens -> grounding context tokens:
    - `2` (resp t=3, w=1.5) -> `Dis` (ctx j=1, g=0.767)
    - `Ġsignal` (resp t=4, w=1.0) -> `3` (ctx j=50, g=0.848)
    - `deficient` (resp t=6, w=1.0) -> `2` (ctx j=22, g=0.828)
    - `Ġlymph` (resp t=11, w=1.0) -> `in` (ctx j=36, g=0.776)
    - `Ġdisplay` (resp t=10, w=1.0) -> `Ġof` (ctx j=34, g=0.768)
- notes: First clause is grounded; second clause about clinical trials is invented.

### A4 - ambiguous / partial - HP:5ab2ac3d
- Q: Be Quick '28 football club is located in what province?
- R: Be Quick '28 is a football club from Zwolle in the province of Overijssel, and it currently plays in the Dutch Eredivisie top flight.
- G_tri = **0.7496**, G_naive_QC = 0.9307, G_rc = 0.9262, echo = 0.6936
- top response tokens -> grounding context tokens:
    - `28` (resp t=4, w=1.5) -> `000` (ctx j=34, g=0.846)
    - `Ġfootball` (resp t=7, w=1.0) -> `ĠQuick` (ctx j=37, g=0.876)
    - `ol` (resp t=12, w=1.0) -> `Ġprovince` (ctx j=18, g=0.788)
    - `Ġplays` (resp t=26, w=1.0) -> `Ġprovince` (ctx j=18, g=0.788)
    - `Ġclub` (resp t=8, w=1.0) -> `Ġprovince` (ctx j=18, g=0.788)
- notes: Province (Overijssel) and city (Zwolle) are grounded; Eredivisie is wrong (context says Eerste Klasse).

### A5 - ambiguous / parametric - SF:238
- Q: Cells undergoing methionine restriction may activate miRNAs.
- R: Methionine restriction extends lifespan in mice by activating SIRT1 deacetylase and downregulating mTOR signaling, an established longevity pathway.
- G_tri = **0.7484**, G_naive_QC = 0.8235, G_rc = 0.8160, echo = 0.6556
- top response tokens -> grounding context tokens:
    - `1` (resp t=13, w=1.5) -> `ĠRNA` (ctx j=77, g=0.646)
    - `Ġsignaling` (resp t=22, w=1.0) -> `micro` (ctx j=1, g=0.793)
    - `ĠS` (resp t=11, w=1.0) -> `micro` (ctx j=1, g=0.793)
    - `reg` (resp t=19, w=1.0) -> `micro` (ctx j=1, g=0.793)
    - `ionine` (resp t=3, w=1.0) -> `micro` (ctx j=1, g=0.793)
- notes: True world knowledge about methionine restriction, but context only mentions stress-activated miRNAs in general; SIRT1/mTOR not present.

### A6 - ambiguous / parametric - HP:5abbbaca
- Q: Honeymoon Academy starred Robert Hays, which award did his co-star win?
- R: Kim Cattrall played Samantha Jones in Sex and the City and was born in Liverpool, England. She trained at the American Academy of Dramatic Arts in New York.
- G_tri = **0.7208**, G_naive_QC = 0.9164, G_rc = 0.9164, echo = 0.5925
- top response tokens -> grounding context tokens:
    - `Kim` (resp t=1, w=1.0) -> `[CLS]` (ctx j=0, g=0.801)
    - `ĠAmerican` (resp t=26, w=1.0) -> `[CLS]` (ctx j=0, g=0.801)
    - `ĠLiverpool` (resp t=18, w=1.0) -> `[CLS]` (ctx j=0, g=0.801)
    - `attr` (resp t=3, w=1.0) -> `[CLS]` (ctx j=0, g=0.801)
    - `Ġtrained` (resp t=23, w=1.0) -> `[CLS]` (ctx j=0, g=0.791)
- notes: Liverpool birthplace and AADA training are true biographical facts not present in context; the context only states English-Canadian and birth date.

### A7 - ambiguous / negation_flip - SF:171
- Q: Basophils counteract disease development in patients with systemic lupus erythematosus (SLE).
- R: Basophils protect against the development of lupus nephritis in SLE by suppressing the production of self-reactive antibodies.
- G_tri = **0.8057**, G_naive_QC = 0.9636, G_rc = 0.9636, echo = 0.7534
- top response tokens -> grounding context tokens:
    - `Ġlupus` (resp t=8, w=1.0) -> `BAS` (ctx j=1, g=0.822)
    - `Ġproduction` (resp t=16, w=1.0) -> `BAS` (ctx j=1, g=0.822)
    - `Ġsuppressing` (resp t=14, w=1.0) -> `BAS` (ctx j=1, g=0.822)
    - `Ġself` (resp t=18, w=1.0) -> `BAS` (ctx j=1, g=0.822)
    - `ritis` (resp t=10, w=1.0) -> `BAS` (ctx j=1, g=0.822)
- notes: Direct negation: context says basophils PROMOTE lupus nephritis; response says they PROTECT.

### A8 - ambiguous / negation_flip - HP:5ab8eabb
- Q: Frank Blake's longtime protégé was also the chairman and CEO of what company in addition to The Home Depot?
- R: Robert Nardelli, Frank Blake's longtime protégé, was never the chairman of Chrysler; he only ever led The Home Depot.
- G_tri = **0.7853**, G_naive_QC = 0.9483, G_rc = 0.9483, echo = 0.7215
- top response tokens -> grounding context tokens:
    - `Ġlongtime` (resp t=9, w=1.0) -> `Ġ2007` (ctx j=52, g=0.845)
    - `Ġnever` (resp t=15, w=1.0) -> `[CLS]` (ctx j=0, g=0.835)
    - `Ġprot` (resp t=10, w=1.0) -> `[CLS]` (ctx j=0, g=0.835)
    - `Robert` (resp t=1, w=1.0) -> `[CLS]` (ctx j=0, g=0.835)
    - `ĠChrysler` (resp t=19, w=1.0) -> `[CLS]` (ctx j=0, g=0.835)
- notes: Direct negation of an explicit context fact (he WAS chairman of Chrysler).

### A9 - ambiguous / entity_swap - SF:327
- Q: Deletion of αvβ8 does not result in a spontaneous inflammatory phenotype.
- R: Treg-cell-specific deletion of integrin αvβ6 did not result in a spontaneous inflammatory phenotype in the studied mice.
- G_tri = **0.7781**, G_naive_QC = 0.9258, G_rc = 0.9258, echo = 0.7019
- top response tokens -> grounding context tokens:
    - `Î²` (resp t=12, w=1.5) -> `-` (ctx j=7, g=0.813)
    - `6` (resp t=13, w=1.5) -> `Ġin` (ctx j=87, g=0.724)
    - `v` (resp t=11, w=1.0) -> `-` (ctx j=7, g=0.813)
    - `ĠÎ±` (resp t=10, w=1.0) -> `-` (ctx j=7, g=0.813)
    - `Ġspontaneous` (resp t=19, w=1.0) -> `Î²` (ctx j=82, g=0.804)
- notes: Single-character entity swap (β6 vs β8); context never discusses β6.

### A10 - ambiguous / entity_swap - HP:5ab3ddbc
- Q: The 2017 European Short Course Swimming Championships will be held in an Arena which began construction when?
- R: Construction on the Royal Arena in Copenhagen, where the 2017 European Short Course Swimming Championships will be held, broke ground on 26 July 2013.
- G_tri = **0.7819**, G_naive_QC = 0.9627, G_rc = 0.9627, echo = 0.7062
- top response tokens -> grounding context tokens:
    - `Ġ2017` (resp t=11, w=1.5) -> `Royal` (ctx j=1, g=0.813)
    - `Ġ26` (resp t=25, w=1.5) -> `[CLS]` (ctx j=0, g=0.777)
    - `Ġ2013` (resp t=27, w=1.5) -> `Ġfor` (ctx j=34, g=0.754)
    - `ĠShort` (resp t=13, w=1.0) -> `ĠFebruary` (ctx j=45, g=0.838)
    - `ĠChampionships` (resp t=17, w=1.0) -> `ĠShort` (ctx j=50, g=0.816)
- notes: Date swap: 26 June -> 26 July (one-letter difference in the month).
