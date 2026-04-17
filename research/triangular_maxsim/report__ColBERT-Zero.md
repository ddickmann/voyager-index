# Triangular MaxSim Falsification Report

Run of `research/triangular_maxsim/experiment.py` on the 20 hand-crafted SciFact + HotpotQA cases. Each (Q, C, R) triple is embedded with `lightonai/ColBERT-Zero` (use_prompts=False) and scored with the Triton kernel in `voyager_index/_internal/kernels/triton_triangular_maxsim.py`.

## Verdict

> HYPOTHESIS FALSIFIED, but naive Reverse MaxSim suffices. Naive R->(Q union C) and/or R->C MaxSim cleanly separate grounded from ungrounded responses (AUROC >= 0.90), so embedding-only Reverse MaxSim CAN score groundedness on this anchor set. However, the Triangular `min(s_RC, a_j)` gating does NOT outperform the naive baseline here -- in fact it compresses the dynamic range and reduces separation. The triangular structure is not earning its keep on this dataset/encoder.

## Anchor metrics (grounded=10 cases? actually 5+5)

| metric | value |
|---|---|
| `AUROC_G_tri` | 0.8400 |
| `AUROC_G_naive_QC` | 1.0000 |
| `AUROC_G_rc` | 1.0000 |
| `AUROC_best` | 1.0000 |
| `mean_G_tri_grounded` | 0.8760 |
| `mean_G_tri_ungrounded` | 0.8374 |
| `G_tri_separation_margin` | 0.0386 |
| `best_thr_G_tri` | 0.8758 |
| `best_acc_G_tri` | 0.9000 |
| `best_thr_G_naive_QC` | 0.9576 |
| `best_acc_G_naive_QC` | 1.0000 |
| `check1_global_separability_pass` | True |
| `check2_triangular_beats_naive_pass` | False |

## Per-case scores

| id | label | sub | G_tri | G_naive_QC | G_rc | echo | GC | kernel-vs-ref err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| G1 | grounded |  | 0.8821 | 0.9641 | 0.9479 | 0.8750 | 0.9182 | 5.96e-08 |
| G2 | grounded |  | 0.8934 | 0.9781 | 0.9767 | 0.8653 | 0.9140 | 1.19e-07 |
| G3 | grounded |  | 0.8758 | 0.9794 | 0.9786 | 0.8618 | 0.9016 | 1.19e-07 |
| G4 | grounded |  | 0.8911 | 0.9812 | 0.9782 | 0.8668 | 0.9567 | 1.19e-07 |
| G5 | grounded |  | 0.8374 | 0.9576 | 0.9555 | 0.8093 | 0.8077 | 1.19e-07 |
| U1 | ungrounded |  | 0.8558 | 0.8850 | 0.8640 | 0.8574 | 0.9105 | 1.19e-07 |
| U2 | ungrounded |  | 0.8495 | 0.8881 | 0.8825 | 0.8356 | 0.8599 | 1.19e-07 |
| U3 | ungrounded |  | 0.8631 | 0.8905 | 0.8825 | 0.8561 | 0.9219 | 1.19e-07 |
| U4 | ungrounded |  | 0.7677 | 0.8483 | 0.8334 | 0.7773 | 0.8807 | 1.19e-07 |
| U5 | ungrounded |  | 0.8508 | 0.8818 | 0.8662 | 0.8644 | 0.8856 | 1.19e-07 |
| A1 | ambiguous | prompt_echo | 0.8293 | 0.9997 | 0.8300 | 0.9997 | 0.8505 | 1.19e-07 |
| A2 | ambiguous | prompt_echo | 0.9289 | 0.9634 | 0.9392 | 0.9634 | 0.9333 | 1.19e-07 |
| A3 | ambiguous | partial | 0.8843 | 0.9266 | 0.9253 | 0.8618 | 0.8967 | 1.79e-07 |
| A4 | ambiguous | partial | 0.7958 | 0.9136 | 0.9129 | 0.7426 | 0.9178 | 1.19e-07 |
| A5 | ambiguous | parametric | 0.7805 | 0.8355 | 0.7817 | 0.8188 | 0.8298 | 1.19e-07 |
| A6 | ambiguous | parametric | 0.7971 | 0.9131 | 0.9061 | 0.7796 | 0.7832 | 1.19e-07 |
| A7 | ambiguous | negation_flip | 0.9303 | 0.9736 | 0.9600 | 0.9305 | 0.9194 | 1.19e-07 |
| A8 | ambiguous | negation_flip | 0.8928 | 0.9687 | 0.9652 | 0.9087 | 0.9383 | 1.19e-07 |
| A9 | ambiguous | entity_swap | 0.9121 | 0.9369 | 0.9358 | 0.9056 | 0.9528 | 1.19e-07 |
| A10 | ambiguous | entity_swap | 0.9114 | 0.9655 | 0.9621 | 0.9058 | 0.9471 | 5.96e-08 |

## Anchor cases - sorted by G_tri

| rank | id | label | G_tri | G_naive_QC | G_rc |
|---:|---|---|---:|---:|---:|
| 1 | G2 `OK` | grounded | 0.8934 | 0.9781 | 0.9767 |
| 2 | G4 `OK` | grounded | 0.8911 | 0.9812 | 0.9782 |
| 3 | G1 `OK` | grounded | 0.8821 | 0.9641 | 0.9479 |
| 4 | G3 `OK` | grounded | 0.8758 | 0.9794 | 0.9786 |
| 5 | U3 `X ` | ungrounded | 0.8631 | 0.8905 | 0.8825 |
| 6 | U1 `X ` | ungrounded | 0.8558 | 0.8850 | 0.8640 |
| 7 | U5 `X ` | ungrounded | 0.8508 | 0.8818 | 0.8662 |
| 8 | U2 `X ` | ungrounded | 0.8495 | 0.8881 | 0.8825 |
| 9 | G5 `OK` | grounded | 0.8374 | 0.9576 | 0.9555 |
| 10 | U4 `X ` | ungrounded | 0.7677 | 0.8483 | 0.8334 |

## Ambiguous cases - per-subcategory diagnostics

### prompt_echo

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A1 | 0.8293 | 0.9997 | 0.9997 | 0.8300 | `5`->`Rs` (0.95); `Ġcentral`->`ĠR` (0.95); `ĠPR`->`Ġvirus` (0.95) |
| A2 | 0.9289 | 0.9634 | 0.9634 | 0.9392 | `ty`->`ty` (0.95); `Ġfather`->`,` (0.95); `re`->`ĠMc` (0.94) |

### partial

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A3 | 0.8843 | 0.8618 | 0.9266 | 0.9253 | `2`->`ĠSH` (0.95); `ritis`->`aly` (0.97); `Ġmice`->`s` (0.96) |
| A4 | 0.7958 | 0.7426 | 0.9136 | 0.9129 | `28`->`000` (0.92); `Ġclub`->`ĠQuick` (0.96); `Ġfootball`->`ĠZ` (0.94) |

### parametric

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A5 | 0.7805 | 0.8188 | 0.8355 | 0.7817 | `1`->`Ġaddition` (0.67); `Ġactivating`->`Ġactiv` (0.96); `ĠmTOR`->`Arg` (0.91) |
| A6 | 0.7971 | 0.7796 | 0.9131 | 0.9061 | `ĠYork`->`),` (0.95); `Ġborn`->`Ġis` (0.92); `ĠSex`->`antha` (0.90) |

### negation_flip

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A7 | 0.9303 | 0.9305 | 0.9736 | 0.9600 | `Ġlupus`->`ĠDEV` (0.97); `ĠSLE`->`Ġ(` (0.97); `Ġdevelopment`->`ĠDEV` (0.97) |
| A8 | 0.8928 | 0.9087 | 0.9687 | 0.9652 | `Ġled`->`Ġsimilar` (0.97); `ĠHome`->`Ġat` (0.96); `ĠChrysler`->`Ġ1949` (0.96) |

### entity_swap

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A9 | 0.9121 | 0.9056 | 0.9369 | 0.9358 | `6`->`-` (0.95); `Î²`->`Ġnot` (0.73); `Ġresult`->`Ġof` (0.98) |
| A10 | 0.9114 | 0.9058 | 0.9655 | 0.9621 | `Ġ2017`->`ĠCourse` (0.97); `Ġ2013`->`ĠRoyal` (0.96); `Ġ26`->`Ġwas` (0.76) |

## Case detail and evidence pointers

### G1 - grounded - SF:839
- Q: Nanoparticles can be targeted against specific cell types by incorporating aptamers into lipid nanoparticles.
- R: Aptamer-functionalized lipid nanoparticles can target specific cell types such as osteoblasts, as demonstrated with the CH6 aptamer-functionalized LNP system.
- G_tri = **0.8821**, G_naive_QC = 0.9641, G_rc = 0.9479, echo = 0.8750
- top response tokens -> grounding context tokens:
    - `6` (resp t=24, w=1.5) -> `.` (ctx j=64, g=0.956)
    - `NP` (resp t=31, w=1.0) -> `rop` (ctx j=163, g=0.970)
    - `Ġlipid` (resp t=7, w=1.0) -> `Ġwe` (ctx j=90, g=0.968)
    - `Ġnanoparticles` (resp t=8, w=1.0) -> `Ġdeveloped` (ctx j=91, g=0.964)
    - `Ġdemonstrated` (resp t=20, w=1.0) -> `Ġfamily` (ctx j=117, g=0.956)
- notes: Close paraphrase of opening sentences of the context.

### G2 - grounded - SF:536
- Q: Hypocretin neurones induce panic-prone state in rats.
- R: Orexin (hypocretin) neurons in the dorsomedial-perifornical hypothalamus play a key role in panic anxiety in a rat model of panic disorder.
- G_tri = **0.8934**, G_naive_QC = 0.9781, G_rc = 0.9767, echo = 0.8653
- top response tokens -> grounding context tokens:
    - `Ġdisorder` (resp t=34, w=1.0) -> `Ġrole` (ctx j=113, g=0.979)
    - `Ġanxiety` (resp t=27, w=1.0) -> `Ġproduces` (ctx j=60, g=0.975)
    - `Ġmodel` (resp t=31, w=1.0) -> `Ġanxiety` (ctx j=61, g=0.969)
    - `Ġrole` (resp t=24, w=1.0) -> `Ġanxiety` (ctx j=61, g=0.969)
    - `Ġplay` (resp t=21, w=1.0) -> `Ġproduces` (ctx j=60, g=0.968)
- notes: Synthesis of two adjacent context sentences; entities preserved.

### G3 - grounded - HP:5ae48aff
- Q: When was George Harrison's second single from his album, that was made with rising conflicts with Warner Bros., released in the United States?
- R: Teardrops, the second single from George Harrison's album Somewhere in England, was released in the United States on 20 July 1981.
- G_tri = **0.8758**, G_naive_QC = 0.9794, G_rc = 0.9786, echo = 0.8618
- top response tokens -> grounding context tokens:
    - `Ġ1981` (resp t=27, w=1.5) -> `Ġone` (ctx j=111, g=0.964)
    - `Ġ20` (resp t=25, w=1.5) -> `ĠStates` (ctx j=29, g=0.814)
    - `ĠGeorge` (resp t=9, w=1.0) -> `Ġis` (ctx j=76, g=0.966)
    - `ĠHarrison` (resp t=10, w=1.0) -> `Ġin` (ctx j=86, g=0.964)
    - `Ġsingle` (resp t=7, w=1.0) -> `Ġthe` (ctx j=39, g=0.956)
- notes: Direct factual answer; every claim has a context source token.

### G4 - grounded - HP:5ab2a54b
- Q: What is the size of the hydroelectric dam that will inundate part of Bui National Park?
- R: The Bui Dam, a 400 MW hydroelectric project on the Black Volta in Ghana, will inundate part of Bui National Park with its reservoir.
- G_tri = **0.8911**, G_naive_QC = 0.9812, G_rc = 0.9782, echo = 0.8668
- top response tokens -> grounding context tokens:
    - `Ġ400` (resp t=7, w=1.5) -> `ĠMW` (ctx j=12, g=0.732)
    - `ui` (resp t=27, w=1.0) -> `ĠPark` (ctx j=58, g=0.967)
    - `ĠNational` (resp t=28, w=1.0) -> `.` (ctx j=54, g=0.966)
    - `ui` (resp t=3, w=1.0) -> `Ġin` (ctx j=51, g=0.966)
    - `Ġhydro` (resp t=9, w=1.0) -> `Ġhydro` (ctx j=13, g=0.965)
- notes: Two-hop answer (dam size + which dam) with both hops in context.

### G5 - grounded - HP:5abce87e
- Q: What role in the military did the discoverer of the elliptical galaxy NGC 3610 have before reaching Britain?
- R: Before migrating to Great Britain in 1757, William Herschel served in the Military Band of Hanover, following his father into that role.
- G_tri = **0.8374**, G_naive_QC = 0.9576, G_rc = 0.9555, echo = 0.8093
- top response tokens -> grounding context tokens:
    - `Ġ17` (resp t=7, w=1.5) -> `ĠGreat` (ctx j=61, g=0.909)
    - `57` (resp t=8, w=1.5) -> `Ġbefore` (ctx j=58, g=0.755)
    - `Ġrole` (resp t=27, w=1.0) -> `Ġ8` (ctx j=95, g=0.950)
    - `Ġfollowing` (resp t=22, w=1.0) -> `ĠGreat` (ctx j=61, g=0.933)
    - `ĠMilitary` (resp t=16, w=1.0) -> `Ġhis` (ctx j=48, g=0.915)
- notes: Multi-hop (NGC 3610 -> Herschel -> military role) all anchored in context.

### U1 - ungrounded - SF:957
- Q: Podocytes are motile and migrate in the presence of injury.
- R: Podocytes are entirely stationary cells that never migrate, even after injury, and serial imaging confirmed they remain locked to the glomerular basement membrane.
- G_tri = **0.8558**, G_naive_QC = 0.8850, G_rc = 0.8640, echo = 0.8574
- top response tokens -> grounding context tokens:
    - `Ġconfirmed` (resp t=18, w=1.0) -> `Ġare` (ctx j=28, g=0.956)
    - `Ġentirely` (resp t=4, w=1.0) -> `Ġare` (ctx j=28, g=0.956)
    - `Ġmembrane` (resp t=26, w=1.0) -> `Ġmigrated` (ctx j=117, g=0.955)
    - `Ġnever` (resp t=8, w=1.0) -> `Ġmultic` (ctx j=108, g=0.945)
    - `ocytes` (resp t=2, w=1.0) -> `ĠPod` (ctx j=26, g=0.921)
- notes: Direct contradiction; the context explicitly shows podocyte motility and migration.

### U2 - ungrounded - SF:362
- Q: During the primary early antibody response activated B cells migrate toward the inner-and outer paracortical areas where oxysterol accumulation is generated by stromal cells.
- R: During the early antibody response, activated B cells are confined to the bone marrow and never enter the lymph node follicles, because CXCL13 actively repels them.
- G_tri = **0.8495**, G_naive_QC = 0.8881, G_rc = 0.8825, echo = 0.8356
- top response tokens -> grounding context tokens:
    - `13` (resp t=26, w=1.5) -> `Ġand` (ctx j=66, g=0.794)
    - `Ġresponse` (resp t=5, w=1.0) -> `-` (ctx j=16, g=0.980)
    - `Ġearly` (resp t=3, w=1.0) -> `Ġplasma` (ctx j=47, g=0.977)
    - `Ġcells` (resp t=9, w=1.0) -> `Ġdepend` (ctx j=25, g=0.974)
    - `ĠB` (resp t=8, w=1.0) -> `Ġresponses` (ctx j=24, g=0.971)
- notes: Inverts the role of CXCL13 and contradicts the entire premise of B cell migration.

### U3 - ungrounded - SF:1216
- Q: The extracellular domain of TMEM27 is cleaved in human beta cells.
- R: TMEM27 is found exclusively inside the nucleus of beta cells, is never glycosylated, and its extracellular domain remains uncleaved throughout the cell's lifetime.
- G_tri = **0.8631**, G_naive_QC = 0.8905, G_rc = 0.8825, echo = 0.8561
- top response tokens -> grounding context tokens:
    - `27` (resp t=3, w=1.5) -> `27` (ctx j=3, g=0.768)
    - `Ġlifetime` (resp t=30, w=1.0) -> `ĠT` (ctx j=92, g=0.976)
    - `Ġextracellular` (resp t=21, w=1.0) -> `Ġdim` (ctx j=48, g=0.964)
    - `Ġcells` (resp t=12, w=1.0) -> `Ġshed` (ctx j=60, g=0.963)
    - `Ġcell` (resp t=28, w=1.0) -> `ĠThis` (ctx j=69, g=0.959)
- notes: Contradicts three separate explicit facts about TMEM27 in the context.

### U4 - ungrounded - HP:5ab2da13
- Q: Who is the brother of the American transgender actor known for his work on MTV's Faking It?
- R: Elliot Fletcher's brother Marcus Fletcher is also a transgender actor, best known for his lead role in the HBO drama Euphoria.
- G_tri = **0.7677**, G_naive_QC = 0.8483, G_rc = 0.8334, echo = 0.7773
- top response tokens -> grounding context tokens:
    - `Ġactor` (resp t=12, w=1.0) -> `Ġan` (ctx j=15, g=0.967)
    - `Ġknown` (resp t=15, w=1.0) -> `Ġtransgender` (ctx j=17, g=0.962)
    - `Ġtransgender` (resp t=11, w=1.0) -> `Ġan` (ctx j=15, g=0.916)
    - `Ġrole` (resp t=19, w=1.0) -> `Ġknown` (ctx j=19, g=0.906)
    - `Ġbrother` (resp t=5, w=1.0) -> `Ġhave` (ctx j=70, g=0.896)
- notes: Marcus, lead role, Euphoria all invented; context says brother is Connor.

### U5 - ungrounded - HP:5a797bd9
- Q: What age was Georgia Middleman when she started singing in the seventh-most populated city in the United States?
- R: Georgia Middleman started singing at age fifteen at the honky-tonks of downtown Houston, Texas, and recorded her first album at sixteen.
- G_tri = **0.8508**, G_naive_QC = 0.8818, G_rc = 0.8662, echo = 0.8644
- top response tokens -> grounding context tokens:
    - `man` (resp t=4, w=1.0) -> `)` (ctx j=43, g=0.941)
    - `ĠTexas` (resp t=21, w=1.0) -> `Ġthe` (ctx j=57, g=0.935)
    - `Ġhon` (resp t=12, w=1.0) -> `Ġthe` (ctx j=57, g=0.933)
    - `Ġstarted` (resp t=5, w=1.0) -> `Ġis` (ctx j=44, g=0.926)
    - `Ġfifteen` (resp t=9, w=1.0) -> `Ġcountry` (ctx j=47, g=0.922)
- notes: Wrong age (15 vs 10), wrong venue, wrong city (Houston vs San Antonio), invented album.

### A1 - ambiguous / prompt_echo - SF:1179
- Q: The PRR MDA5 has a central DExD/H RNA helicase domain.
- R: The PRR MDA5 has a central DExD/H RNA helicase domain.
- G_tri = **0.8293**, G_naive_QC = 0.9997, G_rc = 0.8300, echo = 0.9997
- top response tokens -> grounding context tokens:
    - `5` (resp t=5, w=1.5) -> `Rs` (ctx j=23, g=0.946)
    - `Ġcentral` (resp t=8, w=1.0) -> `ĠR` (ctx j=79, g=0.952)
    - `ĠPR` (resp t=2, w=1.0) -> `Ġvirus` (ctx j=46, g=0.949)
    - `ĠRNA` (resp t=14, w=1.0) -> `LR` (ctx j=57, g=0.925)
    - `ĠMDA` (resp t=4, w=1.0) -> `ĠAber` (ctx j=77, g=0.851)
- notes: Verbatim restatement of the query; the context discusses MDA5 generally but does not mention the DExD/H domain at all.

### A2 - ambiguous / prompt_echo - HP:5ae66b05
- Q: What kind of comedian was Michael McIntyre's father?
- R: Michael McIntyre's father was a comedian.
- G_tri = **0.9289**, G_naive_QC = 0.9634, G_rc = 0.9392, echo = 0.9634
- top response tokens -> grounding context tokens:
    - `ty` (resp t=4, w=1.0) -> `ty` (ctx j=15, g=0.949)
    - `Ġfather` (resp t=7, w=1.0) -> `,` (ctx j=30, g=0.946)
    - `re` (resp t=5, w=1.0) -> `ĠMc` (ctx j=38, g=0.942)
    - `'s` (resp t=6, w=1.0) -> `In` (ctx j=39, g=0.929)
    - `ĠMc` (resp t=2, w=1.0) -> `ĠBritish` (ctx j=35, g=0.913)
- notes: Echoes query verbatim; technically the word 'comedian' is in context but the answer omits the discriminating fact ('Canadian').

### A3 - ambiguous / partial - SF:729
- Q: Lymphadenopathy is observed in knockin mouse lacking the SHP-2 MAPK pathway.
- R: SHP2 signal-deficient knockin mice display lymphadenopathy and splenomegaly, and were the first transgenic model used in human clinical trials for lupus nephritis.
- G_tri = **0.8843**, G_naive_QC = 0.9266, G_rc = 0.9253, echo = 0.8618
- top response tokens -> grounding context tokens:
    - `2` (resp t=3, w=1.5) -> `ĠSH` (ctx j=20, g=0.955)
    - `ritis` (resp t=34, w=1.0) -> `aly` (ctx j=101, g=0.972)
    - `Ġmice` (resp t=9, w=1.0) -> `s` (ctx j=74, g=0.964)
    - `Ġused` (resp t=26, w=1.0) -> `.` (ctx j=29, g=0.963)
    - `Ġknock` (resp t=7, w=1.0) -> `Ġa` (ctx j=32, g=0.962)
- notes: First clause is grounded; second clause about clinical trials is invented.

### A4 - ambiguous / partial - HP:5ab2ac3d
- Q: Be Quick '28 football club is located in what province?
- R: Be Quick '28 is a football club from Zwolle in the province of Overijssel, and it currently plays in the Dutch Eredivisie top flight.
- G_tri = **0.7958**, G_naive_QC = 0.9136, G_rc = 0.9129, echo = 0.7426
- top response tokens -> grounding context tokens:
    - `28` (resp t=4, w=1.5) -> `000` (ctx j=34, g=0.924)
    - `Ġclub` (resp t=8, w=1.0) -> `ĠQuick` (ctx j=42, g=0.959)
    - `Ġfootball` (resp t=7, w=1.0) -> `ĠZ` (ctx j=50, g=0.944)
    - `Ġprovince` (resp t=16, w=1.0) -> `[CLS]` (ctx j=0, g=0.910)
    - `ĠZ` (resp t=10, w=1.0) -> `Ġof` (ctx j=16, g=0.910)
- notes: Province (Overijssel) and city (Zwolle) are grounded; Eredivisie is wrong (context says Eerste Klasse).

### A5 - ambiguous / parametric - SF:238
- Q: Cells undergoing methionine restriction may activate miRNAs.
- R: Methionine restriction extends lifespan in mice by activating SIRT1 deacetylase and downregulating mTOR signaling, an established longevity pathway.
- G_tri = **0.7805**, G_naive_QC = 0.8355, G_rc = 0.7817, echo = 0.8188
- top response tokens -> grounding context tokens:
    - `1` (resp t=13, w=1.5) -> `Ġaddition` (ctx j=31, g=0.673)
    - `Ġactivating` (resp t=10, w=1.0) -> `Ġactiv` (ctx j=54, g=0.962)
    - `ĠmTOR` (resp t=21, w=1.0) -> `Arg` (ctx j=72, g=0.915)
    - `Ġmice` (resp t=8, w=1.0) -> `Ġof` (ctx j=56, g=0.914)
    - `ĠS` (resp t=11, w=1.0) -> `Ġtarget` (ctx j=49, g=0.890)
- notes: True world knowledge about methionine restriction, but context only mentions stress-activated miRNAs in general; SIRT1/mTOR not present.

### A6 - ambiguous / parametric - HP:5abbbaca
- Q: Honeymoon Academy starred Robert Hays, which award did his co-star win?
- R: Kim Cattrall played Samantha Jones in Sex and the City and was born in Liverpool, England. She trained at the American Academy of Dramatic Arts in New York.
- G_tri = **0.7971**, G_naive_QC = 0.9131, G_rc = 0.9061, echo = 0.7796
- top response tokens -> grounding context tokens:
    - `ĠYork` (resp t=35, w=1.0) -> `),` (ctx j=102, g=0.946)
    - `Ġborn` (resp t=16, w=1.0) -> `Ġis` (ctx j=68, g=0.919)
    - `ĠSex` (resp t=10, w=1.0) -> `antha` (ctx j=83, g=0.899)
    - `ĠArts` (resp t=32, w=1.0) -> `ĠGene` (ctx j=43, g=0.898)
    - `ĠAcademy` (resp t=27, w=1.0) -> `ĠGene` (ctx j=43, g=0.880)
- notes: Liverpool birthplace and AADA training are true biographical facts not present in context; the context only states English-Canadian and birth date.

### A7 - ambiguous / negation_flip - SF:171
- Q: Basophils counteract disease development in patients with systemic lupus erythematosus (SLE).
- R: Basophils protect against the development of lupus nephritis in SLE by suppressing the production of self-reactive antibodies.
- G_tri = **0.9303**, G_naive_QC = 0.9736, G_rc = 0.9600, echo = 0.9305
- top response tokens -> grounding context tokens:
    - `Ġlupus` (resp t=8, w=1.0) -> `ĠDEV` (ctx j=22, g=0.971)
    - `ĠSLE` (resp t=12, w=1.0) -> `Ġ(` (ctx j=42, g=0.969)
    - `Ġdevelopment` (resp t=6, w=1.0) -> `ĠDEV` (ctx j=22, g=0.966)
    - `Ġsuppressing` (resp t=14, w=1.0) -> `H` (ctx j=96, g=0.963)
    - `Ġneph` (resp t=9, w=1.0) -> `Ġlupus` (ctx j=38, g=0.962)
- notes: Direct negation: context says basophils PROMOTE lupus nephritis; response says they PROTECT.

### A8 - ambiguous / negation_flip - HP:5ab8eabb
- Q: Frank Blake's longtime protégé was also the chairman and CEO of what company in addition to The Home Depot?
- R: Robert Nardelli, Frank Blake's longtime protégé, was never the chairman of Chrysler; he only ever led The Home Depot.
- G_tri = **0.8928**, G_naive_QC = 0.9687, G_rc = 0.9652, echo = 0.9087
- top response tokens -> grounding context tokens:
    - `Ġled` (resp t=24, w=1.0) -> `Ġsimilar` (ctx j=40, g=0.967)
    - `ĠHome` (resp t=26, w=1.0) -> `Ġat` (ctx j=42, g=0.962)
    - `ĠChrysler` (resp t=19, w=1.0) -> `Ġ1949` (ctx j=66, g=0.960)
    - `Ġprot` (resp t=10, w=1.0) -> `Ġ2007` (ctx j=85, g=0.958)
    - `Ã©g` (resp t=11, w=1.0) -> `Ġto` (ctx j=86, g=0.954)
- notes: Direct negation of an explicit context fact (he WAS chairman of Chrysler).

### A9 - ambiguous / entity_swap - SF:327
- Q: Deletion of αvβ8 does not result in a spontaneous inflammatory phenotype.
- R: Treg-cell-specific deletion of integrin αvβ6 did not result in a spontaneous inflammatory phenotype in the studied mice.
- G_tri = **0.9121**, G_naive_QC = 0.9369, G_rc = 0.9358, echo = 0.9056
- top response tokens -> grounding context tokens:
    - `6` (resp t=13, w=1.5) -> `-` (ctx j=75, g=0.952)
    - `Î²` (resp t=12, w=1.5) -> `Ġnot` (ctx j=85, g=0.728)
    - `Ġresult` (resp t=16, w=1.0) -> `Ġof` (ctx j=78, g=0.981)
    - `Ġspontaneous` (resp t=19, w=1.0) -> `v` (ctx j=81, g=0.973)
    - `Ġmice` (resp t=25, w=1.0) -> `Ġnot` (ctx j=98, g=0.972)
- notes: Single-character entity swap (β6 vs β8); context never discusses β6.

### A10 - ambiguous / entity_swap - HP:5ab3ddbc
- Q: The 2017 European Short Course Swimming Championships will be held in an Arena which began construction when?
- R: Construction on the Royal Arena in Copenhagen, where the 2017 European Short Course Swimming Championships will be held, broke ground on 26 July 2013.
- G_tri = **0.9114**, G_naive_QC = 0.9655, G_rc = 0.9621, echo = 0.9058
- top response tokens -> grounding context tokens:
    - `Ġ2017` (resp t=11, w=1.5) -> `ĠCourse` (ctx j=51, g=0.973)
    - `Ġ2013` (resp t=27, w=1.5) -> `ĠRoyal` (ctx j=79, g=0.964)
    - `Ġ26` (resp t=25, w=1.5) -> `Ġwas` (ctx j=32, g=0.762)
    - `ĠCourse` (resp t=14, w=1.0) -> `ĠChampionships` (ctx j=54, g=0.981)
    - `imming` (resp t=16, w=1.0) -> `ĠThe` (ctx j=56, g=0.980)
- notes: Date swap: 26 June -> 26 July (one-letter difference in the month).
