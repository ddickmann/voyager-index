# Triangular MaxSim Falsification Report

Run of `research/triangular_maxsim/experiment.py` on the 20 hand-crafted SciFact + HotpotQA cases. Each (Q, C, R) triple is embedded with `lightonai/GTE-ModernColBERT-v1` (use_prompts=False) and scored with the Triton kernel in `voyager_index/_internal/kernels/triton_triangular_maxsim.py`.

## Verdict

> HYPOTHESIS FALSIFIED, but naive Reverse MaxSim suffices. Naive R->(Q union C) and/or R->C MaxSim cleanly separate grounded from ungrounded responses (AUROC >= 0.90), so embedding-only Reverse MaxSim CAN score groundedness on this anchor set. However, the Triangular `min(s_RC, a_j)` gating does NOT outperform the naive baseline here -- in fact it compresses the dynamic range and reduces separation. The triangular structure is not earning its keep on this dataset/encoder.

## Anchor metrics (grounded=10 cases? actually 5+5)

| metric | value |
|---|---|
| `AUROC_G_tri` | 0.7600 |
| `AUROC_G_naive_QC` | 1.0000 |
| `AUROC_G_rc` | 1.0000 |
| `AUROC_best` | 1.0000 |
| `mean_G_tri_grounded` | 0.9562 |
| `mean_G_tri_ungrounded` | 0.9503 |
| `G_tri_separation_margin` | 0.0059 |
| `best_thr_G_tri` | 0.9520 |
| `best_acc_G_tri` | 0.7000 |
| `best_thr_G_naive_QC` | 0.9874 |
| `best_acc_G_naive_QC` | 1.0000 |
| `check1_global_separability_pass` | True |
| `check2_triangular_beats_naive_pass` | False |

## Per-case scores

| id | label | sub | G_tri | G_naive_QC | G_rc | echo | GC | kernel-vs-ref err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| G1 | grounded |  | 0.9520 | 0.9878 | 0.9865 | 0.9478 | 0.9547 | 1.19e-07 |
| G2 | grounded |  | 0.9588 | 0.9915 | 0.9915 | 0.9407 | 0.9324 | 1.19e-07 |
| G3 | grounded |  | 0.9618 | 0.9947 | 0.9947 | 0.9541 | 0.9613 | 1.19e-07 |
| G4 | grounded |  | 0.9538 | 0.9958 | 0.9958 | 0.9430 | 0.9619 | 1.19e-07 |
| G5 | grounded |  | 0.9547 | 0.9874 | 0.9874 | 0.9417 | 0.9028 | 1.19e-07 |
| U1 | ungrounded |  | 0.9544 | 0.9613 | 0.9611 | 0.9499 | 0.9569 | 5.96e-08 |
| U2 | ungrounded |  | 0.9494 | 0.9654 | 0.9637 | 0.9482 | 0.9384 | 1.19e-07 |
| U3 | ungrounded |  | 0.9526 | 0.9649 | 0.9649 | 0.9452 | 0.9572 | 1.19e-07 |
| U4 | ungrounded |  | 0.9386 | 0.9593 | 0.9591 | 0.9305 | 0.9590 | 1.19e-07 |
| U5 | ungrounded |  | 0.9568 | 0.9635 | 0.9627 | 0.9484 | 0.9613 | 1.19e-07 |
| A1 | ambiguous | prompt_echo | 0.9387 | 0.9772 | 0.9450 | 0.9742 | 0.9462 | 5.96e-08 |
| A2 | ambiguous | prompt_echo | 0.9668 | 0.9781 | 0.9781 | 0.9594 | 0.9647 | 1.19e-07 |
| A3 | ambiguous | partial | 0.9518 | 0.9735 | 0.9734 | 0.9411 | 0.9368 | 1.19e-07 |
| A4 | ambiguous | partial | 0.9470 | 0.9804 | 0.9804 | 0.9257 | 0.9647 | 1.19e-07 |
| A5 | ambiguous | parametric | 0.9366 | 0.9421 | 0.9372 | 0.9325 | 0.9057 | 1.19e-07 |
| A6 | ambiguous | parametric | 0.9548 | 0.9781 | 0.9781 | 0.9359 | 0.9124 | 1.19e-07 |
| A7 | ambiguous | negation_flip | 0.9647 | 0.9876 | 0.9876 | 0.9569 | 0.9325 | 1.19e-07 |
| A8 | ambiguous | negation_flip | 0.9638 | 0.9912 | 0.9912 | 0.9604 | 0.9681 | 1.19e-07 |
| A9 | ambiguous | entity_swap | 0.9615 | 0.9783 | 0.9783 | 0.9499 | 0.9422 | 1.19e-07 |
| A10 | ambiguous | entity_swap | 0.9587 | 0.9873 | 0.9873 | 0.9509 | 0.9622 | 1.19e-07 |

## Anchor cases - sorted by G_tri

| rank | id | label | G_tri | G_naive_QC | G_rc |
|---:|---|---|---:|---:|---:|
| 1 | G3 `OK` | grounded | 0.9618 | 0.9947 | 0.9947 |
| 2 | G2 `OK` | grounded | 0.9588 | 0.9915 | 0.9915 |
| 3 | U5 `X ` | ungrounded | 0.9568 | 0.9635 | 0.9627 |
| 4 | G5 `OK` | grounded | 0.9547 | 0.9874 | 0.9874 |
| 5 | U1 `X ` | ungrounded | 0.9544 | 0.9613 | 0.9611 |
| 6 | G4 `OK` | grounded | 0.9538 | 0.9958 | 0.9958 |
| 7 | U3 `X ` | ungrounded | 0.9526 | 0.9649 | 0.9649 |
| 8 | G1 `OK` | grounded | 0.9520 | 0.9878 | 0.9865 |
| 9 | U2 `X ` | ungrounded | 0.9494 | 0.9654 | 0.9637 |
| 10 | U4 `X ` | ungrounded | 0.9386 | 0.9593 | 0.9591 |

## Ambiguous cases - per-subcategory diagnostics

### prompt_echo

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A1 | 0.9387 | 0.9742 | 0.9772 | 0.9450 | `5`->`Rs` (0.95); `ĠPR`->`ĠMDA` (0.99); `Ġcentral`->`ĠMDA` (0.99) |
| A2 | 0.9668 | 0.9594 | 0.9781 | 0.9781 | `Ġcomedian`->`Ġcomedian` (0.98); `Ġfather`->`Ġcomedian` (0.98); `ty`->`ĠMichael` (0.97) |

### partial

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A3 | 0.9518 | 0.9411 | 0.9735 | 0.9734 | `2`->`Ġand` (0.96); `Ġused`->`.` (0.99); `ritis`->`Ġnormal` (0.99) |
| A4 | 0.9470 | 0.9257 | 0.9804 | 0.9804 | `28`->`ĠBe` (0.98); `Ġfootball`->`Ġis` (0.99); `ĠZ`->`Ġis` (0.99) |

### parametric

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A5 | 0.9366 | 0.9325 | 0.9421 | 0.9372 | `1`->`Ġinstrumental` (0.89); `Ġmice`->`Ġin` (0.99); `Ġactivating`->`Ġin` (0.99) |
| A6 | 0.9548 | 0.9359 | 0.9781 | 0.9781 | `ĠYork`->`,` (0.99); `Ġborn`->`,` (0.99); `ĠSex`->`Sex` (0.99) |

### negation_flip

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A7 | 0.9647 | 0.9569 | 0.9876 | 0.9876 | `Ġsuppressing`->`ritis` (0.99); `Ġdevelopment`->`ritis` (0.99); `ĠSLE`->`ritis` (0.99) |
| A8 | 0.9638 | 0.9604 | 0.9912 | 0.9912 | `ĠDep`->`Ġfrom` (0.99); `Ġnever`->`Ġfrom` (0.99); `ĠChrysler`->`Ġfrom` (0.99) |

### entity_swap

| id | G_tri | echo | G_naive_QC | G_rc | top-evidence (r_tok -> c_tok, g) |
|---|---:|---:|---:|---:|---|
| A9 | 0.9615 | 0.9499 | 0.9783 | 0.9783 | `6`->`ĠCells` (0.99); `Î²`->`-` (0.93); `Ġresult`->`ĠCells` (0.99) |
| A10 | 0.9587 | 0.9509 | 0.9873 | 0.9873 | `Ġ2013`->`Ġin` (0.98); `Ġ2017`->`ĠCourse` (0.96); `Ġ26`->`Ġwas` (0.95) |

## Case detail and evidence pointers

### G1 - grounded - SF:839
- Q: Nanoparticles can be targeted against specific cell types by incorporating aptamers into lipid nanoparticles.
- R: Aptamer-functionalized lipid nanoparticles can target specific cell types such as osteoblasts, as demonstrated with the CH6 aptamer-functionalized LNP system.
- G_tri = **0.9520**, G_naive_QC = 0.9878, G_rc = 0.9865, echo = 0.9478
- top response tokens -> grounding context tokens:
    - `6` (resp t=24, w=1.5) -> `.` (ctx j=64, g=0.964)
    - `Ġdemonstrated` (resp t=20, w=1.0) -> `Ġtargeting` (ctx j=80, g=0.991)
    - `NP` (resp t=31, w=1.0) -> `-` (ctx j=150, g=0.988)
    - `pt` (resp t=2, w=1.0) -> `RNAs` (ctx j=63, g=0.971)
    - `Ġnanoparticles` (resp t=8, w=1.0) -> `Ġdeveloped` (ctx j=91, g=0.971)
- notes: Close paraphrase of opening sentences of the context.

### G2 - grounded - SF:536
- Q: Hypocretin neurones induce panic-prone state in rats.
- R: Orexin (hypocretin) neurons in the dorsomedial-perifornical hypothalamus play a key role in panic anxiety in a rat model of panic disorder.
- G_tri = **0.9588**, G_naive_QC = 0.9915, G_rc = 0.9915, echo = 0.9407
- top response tokens -> grounding context tokens:
    - `Ġrole` (resp t=24, w=1.0) -> `X` (ctx j=99, g=0.981)
    - `hyp` (resp t=5, w=1.0) -> `X` (ctx j=99, g=0.981)
    - `Ġanxiety` (resp t=27, w=1.0) -> `X` (ctx j=99, g=0.981)
    - `Ġdisorder` (resp t=34, w=1.0) -> `X` (ctx j=99, g=0.981)
    - `Ġplay` (resp t=21, w=1.0) -> `X` (ctx j=99, g=0.981)
- notes: Synthesis of two adjacent context sentences; entities preserved.

### G3 - grounded - HP:5ae48aff
- Q: When was George Harrison's second single from his album, that was made with rising conflicts with Warner Bros., released in the United States?
- R: Teardrops, the second single from George Harrison's album Somewhere in England, was released in the United States on 20 July 1981.
- G_tri = **0.9618**, G_naive_QC = 0.9947, G_rc = 0.9947, echo = 0.9541
- top response tokens -> grounding context tokens:
    - `Ġ1981` (resp t=27, w=1.5) -> `ĠStates` (ctx j=29, g=0.988)
    - `Ġ20` (resp t=25, w=1.5) -> `Ġreleased` (ctx j=21, g=0.970)
    - `Ġreleased` (resp t=19, w=1.0) -> `ĠStates` (ctx j=29, g=0.989)
    - `'s` (resp t=11, w=1.0) -> `ĠStates` (ctx j=29, g=0.989)
    - `ĠStates` (resp t=23, w=1.0) -> `Ġ1981` (ctx j=25, g=0.985)
- notes: Direct factual answer; every claim has a context source token.

### G4 - grounded - HP:5ab2a54b
- Q: What is the size of the hydroelectric dam that will inundate part of Bui National Park?
- R: The Bui Dam, a 400 MW hydroelectric project on the Black Volta in Ghana, will inundate part of Bui National Park with its reservoir.
- G_tri = **0.9538**, G_naive_QC = 0.9958, G_rc = 0.9958, echo = 0.9430
- top response tokens -> grounding context tokens:
    - `Ġ400` (resp t=7, w=1.5) -> `Ġa` (ctx j=10, g=0.919)
    - `Ġreservoir` (resp t=32, w=1.0) -> `Ġis` (ctx j=9, g=0.986)
    - `Ġpart` (resp t=24, w=1.0) -> `Ġis` (ctx j=9, g=0.986)
    - `ĠPark` (resp t=29, w=1.0) -> `Ġis` (ctx j=9, g=0.986)
    - `ĠB` (resp t=2, w=1.0) -> `Ġis` (ctx j=9, g=0.986)
- notes: Two-hop answer (dam size + which dam) with both hops in context.

### G5 - grounded - HP:5abce87e
- Q: What role in the military did the discoverer of the elliptical galaxy NGC 3610 have before reaching Britain?
- R: Before migrating to Great Britain in 1757, William Herschel served in the Military Band of Hanover, following his father into that role.
- G_tri = **0.9547**, G_naive_QC = 0.9874, G_rc = 0.9874, echo = 0.9417
- top response tokens -> grounding context tokens:
    - `Ġ17` (resp t=7, w=1.5) -> `Ġ17` (ctx j=64, g=0.980)
    - `57` (resp t=8, w=1.5) -> `Ġbefore` (ctx j=58, g=0.966)
    - `Ġrole` (resp t=27, w=1.0) -> `)` (ctx j=19, g=0.986)
    - `Ġfather` (resp t=24, w=1.0) -> `)` (ctx j=19, g=0.986)
    - `Ġfollowing` (resp t=22, w=1.0) -> `omer` (ctx j=25, g=0.986)
- notes: Multi-hop (NGC 3610 -> Herschel -> military role) all anchored in context.

### U1 - ungrounded - SF:957
- Q: Podocytes are motile and migrate in the presence of injury.
- R: Podocytes are entirely stationary cells that never migrate, even after injury, and serial imaging confirmed they remain locked to the glomerular basement membrane.
- G_tri = **0.9544**, G_naive_QC = 0.9613, G_rc = 0.9611, echo = 0.9499
- top response tokens -> grounding context tokens:
    - `Ġnever` (resp t=8, w=1.0) -> `Ġmultic` (ctx j=108, g=0.988)
    - `Ġentirely` (resp t=4, w=1.0) -> `Ġmultic` (ctx j=108, g=0.988)
    - `Ġmembrane` (resp t=26, w=1.0) -> `Ġmultic` (ctx j=108, g=0.983)
    - `Ġconfirmed` (resp t=18, w=1.0) -> `Ġare` (ctx j=28, g=0.983)
    - `Ġlocked` (resp t=21, w=1.0) -> `Ġfate` (ctx j=4, g=0.981)
- notes: Direct contradiction; the context explicitly shows podocyte motility and migration.

### U2 - ungrounded - SF:362
- Q: During the primary early antibody response activated B cells migrate toward the inner-and outer paracortical areas where oxysterol accumulation is generated by stromal cells.
- R: During the early antibody response, activated B cells are confined to the bone marrow and never enter the lymph node follicles, because CXCL13 actively repels them.
- G_tri = **0.9494**, G_naive_QC = 0.9654, G_rc = 0.9637, echo = 0.9482
- top response tokens -> grounding context tokens:
    - `13` (resp t=26, w=1.5) -> `Ġmemory` (ctx j=67, g=0.932)
    - `Ġearly` (resp t=3, w=1.0) -> `Ġphases` (ctx j=13, g=0.995)
    - `ĠB` (resp t=8, w=1.0) -> `Ġresponses` (ctx j=24, g=0.992)
    - `Ġcells` (resp t=9, w=1.0) -> `Ġdepend` (ctx j=25, g=0.992)
    - `Ġresponse` (resp t=5, w=1.0) -> `-` (ctx j=16, g=0.987)
- notes: Inverts the role of CXCL13 and contradicts the entire premise of B cell migration.

### U3 - ungrounded - SF:1216
- Q: The extracellular domain of TMEM27 is cleaved in human beta cells.
- R: TMEM27 is found exclusively inside the nucleus of beta cells, is never glycosylated, and its extracellular domain remains uncleaved throughout the cell's lifetime.
- G_tri = **0.9526**, G_naive_QC = 0.9649, G_rc = 0.9649, echo = 0.9452
- top response tokens -> grounding context tokens:
    - `27` (resp t=3, w=1.5) -> `27` (ctx j=3, g=0.949)
    - `Ġfound` (resp t=5, w=1.0) -> `ĠThis` (ctx j=69, g=0.990)
    - `Ġcell` (resp t=28, w=1.0) -> `ĠThis` (ctx j=69, g=0.990)
    - `Ġnucleus` (resp t=9, w=1.0) -> `Ġof` (ctx j=65, g=0.989)
    - `Ġlifetime` (resp t=30, w=1.0) -> `ers` (ctx j=49, g=0.987)
- notes: Contradicts three separate explicit facts about TMEM27 in the context.

### U4 - ungrounded - HP:5ab2da13
- Q: Who is the brother of the American transgender actor known for his work on MTV's Faking It?
- R: Elliot Fletcher's brother Marcus Fletcher is also a transgender actor, best known for his lead role in the HBO drama Euphoria.
- G_tri = **0.9386**, G_naive_QC = 0.9593, G_rc = 0.9591, echo = 0.9305
- top response tokens -> grounding context tokens:
    - `Ġbrother` (resp t=5, w=1.0) -> `Ġactor` (ctx j=18, g=0.991)
    - `Ġtransgender` (resp t=11, w=1.0) -> `Ġactor` (ctx j=18, g=0.991)
    - `Ġrole` (resp t=19, w=1.0) -> `Ġhis` (ctx j=21, g=0.983)
    - `Ġknown` (resp t=15, w=1.0) -> `Ġtransgender` (ctx j=17, g=0.974)
    - `Ġbest` (resp t=14, w=1.0) -> `Ġactor` (ctx j=18, g=0.973)
- notes: Marcus, lead role, Euphoria all invented; context says brother is Connor.

### U5 - ungrounded - HP:5a797bd9
- Q: What age was Georgia Middleman when she started singing in the seventh-most populated city in the United States?
- R: Georgia Middleman started singing at age fifteen at the honky-tonks of downtown Houston, Texas, and recorded her first album at sixteen.
- G_tri = **0.9568**, G_naive_QC = 0.9635, G_rc = 0.9627, echo = 0.9484
- top response tokens -> grounding context tokens:
    - `ĠTexas` (resp t=21, w=1.0) -> `ĠStates` (ctx j=17, g=0.991)
    - `Ġhon` (resp t=12, w=1.0) -> `Ġthe` (ctx j=57, g=0.990)
    - `Ġage` (resp t=8, w=1.0) -> `.` (ctx j=49, g=0.983)
    - `Ġfirst` (resp t=26, w=1.0) -> `.` (ctx j=49, g=0.976)
    - `Ġstarted` (resp t=5, w=1.0) -> `.` (ctx j=28, g=0.976)
- notes: Wrong age (15 vs 10), wrong venue, wrong city (Houston vs San Antonio), invented album.

### A1 - ambiguous / prompt_echo - SF:1179
- Q: The PRR MDA5 has a central DExD/H RNA helicase domain.
- R: The PRR MDA5 has a central DExD/H RNA helicase domain.
- G_tri = **0.9387**, G_naive_QC = 0.9772, G_rc = 0.9450, echo = 0.9742
- top response tokens -> grounding context tokens:
    - `5` (resp t=5, w=1.5) -> `Rs` (ctx j=23, g=0.951)
    - `ĠPR` (resp t=2, w=1.0) -> `ĠMDA` (ctx j=30, g=0.993)
    - `Ġcentral` (resp t=8, w=1.0) -> `ĠMDA` (ctx j=30, g=0.992)
    - `ĠRNA` (resp t=14, w=1.0) -> `LR` (ctx j=57, g=0.960)
    - `ase` (resp t=16, w=1.0) -> `2` (ctx j=36, g=0.943)
- notes: Verbatim restatement of the query; the context discusses MDA5 generally but does not mention the DExD/H domain at all.

### A2 - ambiguous / prompt_echo - HP:5ae66b05
- Q: What kind of comedian was Michael McIntyre's father?
- R: Michael McIntyre's father was a comedian.
- G_tri = **0.9668**, G_naive_QC = 0.9781, G_rc = 0.9781, echo = 0.9594
- top response tokens -> grounding context tokens:
    - `Ġcomedian` (resp t=10, w=1.0) -> `Ġcomedian` (ctx j=29, g=0.985)
    - `Ġfather` (resp t=7, w=1.0) -> `Ġcomedian` (ctx j=29, g=0.985)
    - `ty` (resp t=4, w=1.0) -> `ĠMichael` (ctx j=37, g=0.968)
    - `'s` (resp t=6, w=1.0) -> `In` (ctx j=39, g=0.965)
    - `re` (resp t=5, w=1.0) -> `ĠMc` (ctx j=38, g=0.964)
- notes: Echoes query verbatim; technically the word 'comedian' is in context but the answer omits the discriminating fact ('Canadian').

### A3 - ambiguous / partial - SF:729
- Q: Lymphadenopathy is observed in knockin mouse lacking the SHP-2 MAPK pathway.
- R: SHP2 signal-deficient knockin mice display lymphadenopathy and splenomegaly, and were the first transgenic model used in human clinical trials for lupus nephritis.
- G_tri = **0.9518**, G_naive_QC = 0.9735, G_rc = 0.9734, echo = 0.9411
- top response tokens -> grounding context tokens:
    - `2` (resp t=3, w=1.5) -> `Ġand` (ctx j=19, g=0.960)
    - `Ġused` (resp t=26, w=1.0) -> `.` (ctx j=29, g=0.992)
    - `ritis` (resp t=34, w=1.0) -> `Ġnormal` (ctx j=95, g=0.990)
    - `Ġtrials` (resp t=30, w=1.0) -> `Ġvivo` (ctx j=11, g=0.984)
    - `Ġmodel` (resp t=25, w=1.0) -> `Ġvivo` (ctx j=11, g=0.964)
- notes: First clause is grounded; second clause about clinical trials is invented.

### A4 - ambiguous / partial - HP:5ab2ac3d
- Q: Be Quick '28 football club is located in what province?
- R: Be Quick '28 is a football club from Zwolle in the province of Overijssel, and it currently plays in the Dutch Eredivisie top flight.
- G_tri = **0.9470**, G_naive_QC = 0.9804, G_rc = 0.9804, echo = 0.9257
- top response tokens -> grounding context tokens:
    - `28` (resp t=4, w=1.5) -> `ĠBe` (ctx j=41, g=0.982)
    - `Ġfootball` (resp t=7, w=1.0) -> `Ġis` (ctx j=10, g=0.985)
    - `ĠZ` (resp t=10, w=1.0) -> `Ġis` (ctx j=10, g=0.985)
    - `Ġprovince` (resp t=16, w=1.0) -> `ĠIt` (ctx j=27, g=0.979)
    - `Ġplays` (resp t=26, w=1.0) -> `Ġwhich` (ctx j=57, g=0.975)
- notes: Province (Overijssel) and city (Zwolle) are grounded; Eredivisie is wrong (context says Eerste Klasse).

### A5 - ambiguous / parametric - SF:238
- Q: Cells undergoing methionine restriction may activate miRNAs.
- R: Methionine restriction extends lifespan in mice by activating SIRT1 deacetylase and downregulating mTOR signaling, an established longevity pathway.
- G_tri = **0.9366**, G_naive_QC = 0.9421, G_rc = 0.9372, echo = 0.9325
- top response tokens -> grounding context tokens:
    - `1` (resp t=13, w=1.5) -> `Ġinstrumental` (ctx j=23, g=0.892)
    - `Ġmice` (resp t=8, w=1.0) -> `Ġin` (ctx j=30, g=0.990)
    - `Ġactivating` (resp t=10, w=1.0) -> `Ġin` (ctx j=30, g=0.990)
    - `ĠmTOR` (resp t=21, w=1.0) -> `Ġrecognized` (ctx j=35, g=0.985)
    - `eth` (resp t=2, w=1.0) -> `Ġare` (ctx j=22, g=0.975)
- notes: True world knowledge about methionine restriction, but context only mentions stress-activated miRNAs in general; SIRT1/mTOR not present.

### A6 - ambiguous / parametric - HP:5abbbaca
- Q: Honeymoon Academy starred Robert Hays, which award did his co-star win?
- R: Kim Cattrall played Samantha Jones in Sex and the City and was born in Liverpool, England. She trained at the American Academy of Dramatic Arts in New York.
- G_tri = **0.9548**, G_naive_QC = 0.9781, G_rc = 0.9781, echo = 0.9359
- top response tokens -> grounding context tokens:
    - `ĠYork` (resp t=35, w=1.0) -> `,` (ctx j=91, g=0.989)
    - `Ġborn` (resp t=16, w=1.0) -> `,` (ctx j=91, g=0.989)
    - `ĠSex` (resp t=10, w=1.0) -> `Sex` (ctx j=93, g=0.989)
    - `ĠCity` (resp t=13, w=1.0) -> `Sex` (ctx j=93, g=0.989)
    - `ĠAcademy` (resp t=27, w=1.0) -> `Ġfor` (ctx j=78, g=0.987)
- notes: Liverpool birthplace and AADA training are true biographical facts not present in context; the context only states English-Canadian and birth date.

### A7 - ambiguous / negation_flip - SF:171
- Q: Basophils counteract disease development in patients with systemic lupus erythematosus (SLE).
- R: Basophils protect against the development of lupus nephritis in SLE by suppressing the production of self-reactive antibodies.
- G_tri = **0.9647**, G_naive_QC = 0.9876, G_rc = 0.9876, echo = 0.9569
- top response tokens -> grounding context tokens:
    - `Ġsuppressing` (resp t=14, w=1.0) -> `ritis` (ctx j=59, g=0.986)
    - `Ġdevelopment` (resp t=6, w=1.0) -> `ritis` (ctx j=59, g=0.986)
    - `ĠSLE` (resp t=12, w=1.0) -> `ritis` (ctx j=59, g=0.986)
    - `Ġlupus` (resp t=8, w=1.0) -> `ritis` (ctx j=59, g=0.986)
    - `Ġself` (resp t=18, w=1.0) -> `ritis` (ctx j=59, g=0.986)
- notes: Direct negation: context says basophils PROMOTE lupus nephritis; response says they PROTECT.

### A8 - ambiguous / negation_flip - HP:5ab8eabb
- Q: Frank Blake's longtime protégé was also the chairman and CEO of what company in addition to The Home Depot?
- R: Robert Nardelli, Frank Blake's longtime protégé, was never the chairman of Chrysler; he only ever led The Home Depot.
- G_tri = **0.9638**, G_naive_QC = 0.9912, G_rc = 0.9912, echo = 0.9604
- top response tokens -> grounding context tokens:
    - `ĠDep` (resp t=27, w=1.0) -> `Ġfrom` (ctx j=83, g=0.990)
    - `Ġnever` (resp t=15, w=1.0) -> `Ġfrom` (ctx j=83, g=0.990)
    - `ĠChrysler` (resp t=19, w=1.0) -> `Ġfrom` (ctx j=83, g=0.990)
    - `'s` (resp t=8, w=1.0) -> `Ġfrom` (ctx j=83, g=0.990)
    - `Ġever` (resp t=23, w=1.0) -> `Ġ1949` (ctx j=66, g=0.989)
- notes: Direct negation of an explicit context fact (he WAS chairman of Chrysler).

### A9 - ambiguous / entity_swap - SF:327
- Q: Deletion of αvβ8 does not result in a spontaneous inflammatory phenotype.
- R: Treg-cell-specific deletion of integrin αvβ6 did not result in a spontaneous inflammatory phenotype in the studied mice.
- G_tri = **0.9615**, G_naive_QC = 0.9783, G_rc = 0.9783, echo = 0.9499
- top response tokens -> grounding context tokens:
    - `6` (resp t=13, w=1.5) -> `ĠCells` (ctx j=20, g=0.990)
    - `Î²` (resp t=12, w=1.5) -> `-` (ctx j=69, g=0.934)
    - `Ġresult` (resp t=16, w=1.0) -> `ĠCells` (ctx j=20, g=0.990)
    - `Ġphenotype` (resp t=21, w=1.0) -> `ĠCells` (ctx j=20, g=0.990)
    - `Ġmice` (resp t=25, w=1.0) -> `Ġdid` (ctx j=84, g=0.988)
- notes: Single-character entity swap (β6 vs β8); context never discusses β6.

### A10 - ambiguous / entity_swap - HP:5ab3ddbc
- Q: The 2017 European Short Course Swimming Championships will be held in an Arena which began construction when?
- R: Construction on the Royal Arena in Copenhagen, where the 2017 European Short Course Swimming Championships will be held, broke ground on 26 July 2013.
- G_tri = **0.9587**, G_naive_QC = 0.9873, G_rc = 0.9873, echo = 0.9509
- top response tokens -> grounding context tokens:
    - `Ġ2013` (resp t=27, w=1.5) -> `Ġin` (ctx j=67, g=0.984)
    - `Ġ2017` (resp t=11, w=1.5) -> `ĠCourse` (ctx j=51, g=0.957)
    - `Ġ26` (resp t=25, w=1.5) -> `Ġwas` (ctx j=32, g=0.950)
    - `ĠCopenhagen` (resp t=7, w=1.0) -> `Ġin` (ctx j=67, g=0.984)
    - `ĠRoyal` (resp t=4, w=1.0) -> `Ġin` (ctx j=67, g=0.984)
- notes: Date swap: 26 June -> 26 July (one-letter difference in the month).
