"""Twenty hand-crafted (query, context, response) triples for the Triangular
MaxSim falsification experiment.

Distribution:
    grounded:     5  (response is a close paraphrase of the context)
    ungrounded:   5  (response is plausible but contradicts/invents)
    ambiguous:   10  (5 sub-categories x 2 cases)
        prompt_echo:    response repeats the query, low context content
        partial:        response has one supported clause + one unsupported
        parametric:     response is true world-knowledge but not in C
        negation_flip:  response negates a fact stated in C
        entity_swap:    response swaps an entity/number with a near miss

Sources (provenance for reproducibility):
    SF = SciFact (BEIR), with `qid` matching qrels/test.jsonl.
    HP = HotpotQA distractor split, with `id` matching the upstream record.
Contexts are quoted verbatim from the upstream datasets so the experiment is
fully reproducible without re-downloading anything.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional

LABEL = Literal["grounded", "ungrounded", "ambiguous"]
SUBCAT = Optional[Literal["prompt_echo", "partial", "parametric",
                          "negation_flip", "entity_swap", None]]


@dataclass
class Case:
    id: str
    source: str          # "SF:<qid>" or "HP:<id>"
    query: str
    context: str
    response: str
    label: str
    subcategory: Optional[str]   # None for grounded/ungrounded
    notes: str


CASES: List[Case] = [
    # ----------------------------------------------------------------
    # GROUNDED (5)
    # ----------------------------------------------------------------
    Case(
        id="G1",
        source="SF:839",
        query="Nanoparticles can be targeted against specific cell types by incorporating aptamers into lipid nanoparticles.",
        context=(
            "Aptamer-functionalized lipid nanoparticles targeting osteoblasts as a novel RNA "
            "interference-based bone anabolic strategy. Currently, major concerns about the safety and "
            "efficacy of RNA interference (RNAi)-based bone anabolic strategies still exist because of "
            "the lack of direct osteoblast-specific delivery systems for osteogenic siRNAs. Here we "
            "screened the aptamer CH6 by cell-SELEX, specifically targeting both rat and human "
            "osteoblasts, and then we developed CH6 aptamer-functionalized lipid nanoparticles (LNPs) "
            "encapsulating osteogenic pleckstrin homology domain-containing family O member 1 (Plekho1) "
            "siRNA (CH6-LNPs-siRNA). Our results showed that CH6 facilitated in vitro "
            "osteoblast-selective uptake of Plekho1 siRNA, mainly via macropinocytosis, and boosted in "
            "vivo osteoblast-specific Plekho1 gene silencing."
        ),
        response=(
            "Aptamer-functionalized lipid nanoparticles can target specific cell types such as "
            "osteoblasts, as demonstrated with the CH6 aptamer-functionalized LNP system."
        ),
        label="grounded",
        subcategory=None,
        notes="Close paraphrase of opening sentences of the context.",
    ),
    Case(
        id="G2",
        source="SF:536",
        query="Hypocretin neurones induce panic-prone state in rats.",
        context=(
            "A KEY ROLE FOR OREXIN IN PANIC ANXIETY. Panic disorder is a severe anxiety disorder with "
            "recurrent, debilitating panic attacks. In a rat model of panic disorder, chronic "
            "inhibition of GABA synthesis in the dorsomedial-perifornical hypothalamus of rats produces "
            "anxiety-like states and a similar vulnerability to sodium lactate-induced cardioexcitatory "
            "responses. The dorsomedial-perifornical hypothalamus is enriched in neurons containing "
            "orexin (ORX, also known as hypocretin), which have a crucial role in arousal, vigilance "
            "and central autonomic mobilization."
        ),
        response=(
            "Orexin (hypocretin) neurons in the dorsomedial-perifornical hypothalamus play a key role in "
            "panic anxiety in a rat model of panic disorder."
        ),
        label="grounded",
        subcategory=None,
        notes="Synthesis of two adjacent context sentences; entities preserved.",
    ),
    Case(
        id="G3",
        source="HP:5ae48aff",
        query="When was George Harrison's second single from his album, that was made with rising conflicts with Warner Bros., released in the United States?",
        context=(
            "Teardrops (George Harrison song). \"Teardrops\" is a single by George Harrison, released "
            "on 20 July 1981 in the United States and 31 July in the UK. It was the second single from "
            "the album \"Somewhere in England\" and reached number 51 on the US \"Billboard\" Top "
            "Tracks chart. Somewhere in England. Somewhere in England is the ninth studio album by "
            "George Harrison, released in 1981. The album was recorded as Harrison was becoming "
            "increasingly frustrated with the music industry. The album's making was a long one, "
            "during which conflicts with Warner Bros. Records arose."
        ),
        response=(
            "Teardrops, the second single from George Harrison's album Somewhere in England, was "
            "released in the United States on 20 July 1981."
        ),
        label="grounded",
        subcategory=None,
        notes="Direct factual answer; every claim has a context source token.",
    ),
    Case(
        id="G4",
        source="HP:5ab2a54b",
        query="What is the size of the hydroelectric dam that will inundate part of Bui National Park?",
        context=(
            "Bui Dam. The Bui Dam is a 400 MW hydroelectric project in Ghana. It is built on the Black "
            "Volta river at the Bui Gorge, at the southern end of Bui National Park. Construction on "
            "the main dam began in December 2009. Bui National Park. The Bui National Park is found in "
            "Ghana. Part of the park will be inundated by the reservoir of the Bui Dam, which has been "
            "under construction since 2009."
        ),
        response=(
            "The Bui Dam, a 400 MW hydroelectric project on the Black Volta in Ghana, will inundate "
            "part of Bui National Park with its reservoir."
        ),
        label="grounded",
        subcategory=None,
        notes="Two-hop answer (dam size + which dam) with both hops in context.",
    ),
    Case(
        id="G5",
        source="HP:5abce87e",
        query="What role in the military did the discoverer of the elliptical galaxy NGC 3610 have before reaching Britain?",
        context=(
            "William Herschel. Frederick William Herschel (15 November 1738 - 25 August 1822) was a "
            "British astronomer and composer of German and Czech-Jewish origin. Born in the Electorate "
            "of Hanover, Herschel followed his father into the Military Band of Hanover, before "
            "migrating to Great Britain in 1757 at the age of nineteen. NGC 3610. NGC 3610 is an "
            "elliptical galaxy in the constellation Ursa Major. It was discovered on 8 April 1793 by "
            "William Herschel."
        ),
        response=(
            "Before migrating to Great Britain in 1757, William Herschel served in the Military Band "
            "of Hanover, following his father into that role."
        ),
        label="grounded",
        subcategory=None,
        notes="Multi-hop (NGC 3610 -> Herschel -> military role) all anchored in context.",
    ),

    # ----------------------------------------------------------------
    # UNGROUNDED (5)
    # ----------------------------------------------------------------
    Case(
        id="U1",
        source="SF:957",
        query="Podocytes are motile and migrate in the presence of injury.",
        context=(
            "Tracking the fate of glomerular epithelial cells in vivo using serial multiphoton "
            "imaging in novel mouse models with fluorescent lineage tags. Podocytes are critical in "
            "the maintenance of a healthy glomerular filter; however, they have been difficult to "
            "study in the intact kidney because of technical limitations. Here we report the "
            "development of serial multiphoton microscopy (MPM) of the same glomeruli over several "
            "days to visualize the motility of podocytes and parietal epithelial cells (PECs) in vivo. "
            "In podocin-GFP mice, podocytes formed sporadic multicellular clusters after unilateral "
            "ureteral ligation and migrated into the parietal Bowman's capsule."
        ),
        response=(
            "Podocytes are entirely stationary cells that never migrate, even after injury, and serial "
            "imaging confirmed they remain locked to the glomerular basement membrane."
        ),
        label="ungrounded",
        subcategory=None,
        notes="Direct contradiction; the context explicitly shows podocyte motility and migration.",
    ),
    Case(
        id="U2",
        source="SF:362",
        query="During the primary early antibody response activated B cells migrate toward the inner-and outer paracortical areas where oxysterol accumulation is generated by stromal cells.",
        context=(
            "Finding the right niche: B-cell migration in the early phases of T-dependent antibody "
            "responses. Humoral immune responses depend on B cells encountering antigen, interacting "
            "with helper T cells, proliferating and differentiating into low-affinity plasma cells or, "
            "after organizing into a germinal center (GC), high-affinity plasma cells and memory B "
            "cells. The chemokine, CXCL13, plays a central role in guiding B cells to follicles "
            "whereas T-zone chemokines guide activated B cells to the T zone. Sphingosine-1-phosphate "
            "(S1P) promotes cell egress from lymphoid tissue."
        ),
        response=(
            "During the early antibody response, activated B cells are confined to the bone marrow and "
            "never enter the lymph node follicles, because CXCL13 actively repels them."
        ),
        label="ungrounded",
        subcategory=None,
        notes="Inverts the role of CXCL13 and contradicts the entire premise of B cell migration.",
    ),
    Case(
        id="U3",
        source="SF:1216",
        query="The extracellular domain of TMEM27 is cleaved in human beta cells.",
        context=(
            "Tmem27: a cleaved and shed plasma membrane protein that stimulates pancreatic beta cell "
            "proliferation. Here, we report the identification and characterization of transmembrane "
            "protein 27 (Tmem27, collectrin) in pancreatic beta cells. Tmem27 forms dimers and its "
            "extracellular domain is glycosylated, cleaved and shed from the plasma membrane of beta "
            "cells. This cleavage process is beta cell specific and does not occur in other cell "
            "types. Overexpression of full-length Tmem27 leads to increased beta cell proliferation."
        ),
        response=(
            "TMEM27 is found exclusively inside the nucleus of beta cells, is never glycosylated, and "
            "its extracellular domain remains uncleaved throughout the cell's lifetime."
        ),
        label="ungrounded",
        subcategory=None,
        notes="Contradicts three separate explicit facts about TMEM27 in the context.",
    ),
    Case(
        id="U4",
        source="HP:5ab2da13",
        query="Who is the brother of the American transgender actor known for his work on MTV's Faking It?",
        context=(
            "Elliot Fletcher. Elliot Fletcher (born June 30, 1996) is an American transgender actor "
            "known for his work on the MTV comedy series \"Faking It\" and Freeform's \"The Fosters\". "
            "Julia Fletcher. Julia Fletcher (born 1958) is an American voice actress. She is married to "
            "voice actor John DeMita. They have two sons, Connor (born 1995) and Elliot Fletcher "
            "(born 1996)."
        ),
        response=(
            "Elliot Fletcher's brother Marcus Fletcher is also a transgender actor, best known for his "
            "lead role in the HBO drama Euphoria."
        ),
        label="ungrounded",
        subcategory=None,
        notes="Marcus, lead role, Euphoria all invented; context says brother is Connor.",
    ),
    Case(
        id="U5",
        source="HP:5a797bd9",
        query="What age was Georgia Middleman when she started singing in the seventh-most populated city in the United States?",
        context=(
            "San Antonio. San Antonio is the seventh-most populous city in the United States and the "
            "second-most populous city in Texas. Georgia Middleman. Georgia Leigh Middleman (born "
            "December 27, 1967) is an American country singer. Middleman sang from age ten at the "
            "Texas Star Inn in San Antonio, and began writing songs shortly thereafter."
        ),
        response=(
            "Georgia Middleman started singing at age fifteen at the honky-tonks of downtown Houston, "
            "Texas, and recorded her first album at sixteen."
        ),
        label="ungrounded",
        subcategory=None,
        notes="Wrong age (15 vs 10), wrong venue, wrong city (Houston vs San Antonio), invented album.",
    ),

    # ----------------------------------------------------------------
    # AMBIGUOUS (10) - 2 per sub-category
    # ----------------------------------------------------------------

    # ---- prompt_echo (2) ----
    Case(
        id="A1",
        source="SF:1179",
        query="The PRR MDA5 has a central DExD/H RNA helicase domain.",
        context=(
            "Immune signaling by RIG-I-like receptors. The RIG-I-like receptors (RLRs) RIG-I, MDA5, "
            "and LGP2 play a major role in pathogen sensing of RNA virus infection to initiate and "
            "modulate antiviral immunity. The RLRs detect viral RNA ligands or processed self RNA in "
            "the cytoplasm to trigger innate immunity and inflammation. Aberrant RLR signaling or "
            "dysregulation of RLR expression is implicated in autoimmune disease."
        ),
        response=(
            "The PRR MDA5 has a central DExD/H RNA helicase domain."
        ),
        label="ambiguous",
        subcategory="prompt_echo",
        notes="Verbatim restatement of the query; the context discusses MDA5 generally but does not mention the DExD/H domain at all.",
    ),
    Case(
        id="A2",
        source="HP:5ae66b05",
        query="What kind of comedian was Michael McIntyre's father?",
        context=(
            "Ray Cameron (comedian). Ray Cameron (born Thomas Cameron McIntyre; 19 April 1936 - "
            "27 December 1993) was a Canadian comedian, and the father of British comedian Michael "
            "McIntyre. Michael McIntyre. Michael Hazen James McIntyre (born 21 February 1976) is an "
            "English comedian, actor and presenter."
        ),
        response=(
            "Michael McIntyre's father was a comedian."
        ),
        label="ambiguous",
        subcategory="prompt_echo",
        notes="Echoes query verbatim; technically the word 'comedian' is in context but the answer omits the discriminating fact ('Canadian').",
    ),

    # ---- partial (2) ----
    Case(
        id="A3",
        source="SF:729",
        query="Lymphadenopathy is observed in knockin mouse lacking the SHP-2 MAPK pathway.",
        context=(
            "Dissection of signaling cascades through gp130 in vivo: reciprocal roles for STAT3- and "
            "SHP2-mediated signals in immune responses. We generated a series of knockin mouse lines, "
            "in which the cytokine receptor gp130-dependent STAT3 and/or SHP2 signals were disrupted, "
            "by replacing the mouse gp130 gene with human gp130 mutant cDNAs. The SHP2 signal-deficient "
            "mice (gp130F759/F759) were born normal but displayed splenomegaly and lymphadenopathy and "
            "an enhanced acute phase reaction."
        ),
        response=(
            "SHP2 signal-deficient knockin mice display lymphadenopathy and splenomegaly, and were the "
            "first transgenic model used in human clinical trials for lupus nephritis."
        ),
        label="ambiguous",
        subcategory="partial",
        notes="First clause is grounded; second clause about clinical trials is invented.",
    ),
    Case(
        id="A4",
        source="HP:5ab2ac3d",
        query="Be Quick '28 football club is located in what province?",
        context=(
            "Zwolle. Zwolle is the capital city and municipality of the province of Overijssel, "
            "Netherlands. It has a population around 125,000. Be Quick '28. Be Quick '28 is a football "
            "club from Zwolle, Netherlands, which was founded on 22 November 1928. They currently play "
            "in the Eerste Klasse."
        ),
        response=(
            "Be Quick '28 is a football club from Zwolle in the province of Overijssel, and it "
            "currently plays in the Dutch Eredivisie top flight."
        ),
        label="ambiguous",
        subcategory="partial",
        notes="Province (Overijssel) and city (Zwolle) are grounded; Eredivisie is wrong (context says Eerste Klasse).",
    ),

    # ---- parametric (2) ----
    Case(
        id="A5",
        source="SF:238",
        query="Cells undergoing methionine restriction may activate miRNAs.",
        context=(
            "microRNAs: A Safeguard against Turmoil? Emerging data suggest that microRNAs (miRNAs) "
            "are instrumental in a variety of stress responses in addition to their more recognized "
            "role in development. Surprisingly, miRNAs, which normally suppress expression of target "
            "transcripts, may become activators of expression during stress. This might be partially "
            "explained by new interactions of miRNA/Argonaute complexes with RNA-binding proteins "
            "that relocate from different subcellular compartments during stress."
        ),
        response=(
            "Methionine restriction extends lifespan in mice by activating SIRT1 deacetylase and "
            "downregulating mTOR signaling, an established longevity pathway."
        ),
        label="ambiguous",
        subcategory="parametric",
        notes="True world knowledge about methionine restriction, but context only mentions stress-activated miRNAs in general; SIRT1/mTOR not present.",
    ),
    Case(
        id="A6",
        source="HP:5abbbaca",
        query="Honeymoon Academy starred Robert Hays, which award did his co-star win?",
        context=(
            "Honeymoon Academy. Honeymoon Academy (also titled For Better or For Worse) is a 1990 "
            "American comedy-drama film starring Robert Hays and Kim Cattrall. It was directed by "
            "Gene Quintano and was filmed in Spain. Kim Cattrall. Kim Victoria Cattrall (born 21 "
            "August 1956) is an English-Canadian actress. She is known for her role as Samantha Jones "
            "in the HBO romantic comedy series, \"Sex and the City\" (1998-2004), winning the 2002 "
            "Golden Globe for Best Supporting Actress."
        ),
        response=(
            "Kim Cattrall played Samantha Jones in Sex and the City and was born in Liverpool, "
            "England. She trained at the American Academy of Dramatic Arts in New York."
        ),
        label="ambiguous",
        subcategory="parametric",
        notes="Liverpool birthplace and AADA training are true biographical facts not present in context; the context only states English-Canadian and birth date.",
    ),

    # ---- negation_flip (2) ----
    Case(
        id="A7",
        source="SF:171",
        query="Basophils counteract disease development in patients with systemic lupus erythematosus (SLE).",
        context=(
            "BASOPHILS AND THE T HELPER 2 ENVIRONMENT CAN PROMOTE THE DEVELOPMENT OF LUPUS NEPHRITIS. "
            "In systemic lupus erythematosus (SLE), self-reactive antibodies can target the kidney "
            "(lupus nephritis), leading to functional failure and possible mortality. We report that "
            "activation of basophils by autoreactive IgE causes their homing to lymph nodes, promoting "
            "T helper type 2 (T(H)2) cell differentiation and enhancing the production of "
            "self-reactive antibodies that cause lupus-like nephritis in mice. Individuals with SLE "
            "also have elevated serum IgE, self-reactive IgEs and activated basophils."
        ),
        response=(
            "Basophils protect against the development of lupus nephritis in SLE by suppressing the "
            "production of self-reactive antibodies."
        ),
        label="ambiguous",
        subcategory="negation_flip",
        notes="Direct negation: context says basophils PROMOTE lupus nephritis; response says they PROTECT.",
    ),
    Case(
        id="A8",
        source="HP:5ab8eabb",
        query="Frank Blake's longtime protégé was also the chairman and CEO of what company in addition to The Home Depot?",
        context=(
            "Robert Nardelli. Robert Louis Nardelli (born May 17, 1948) is an American businessman. "
            "He was the chairman and chief executive officer of Chrysler. He had earlier served in a "
            "similar capacity at The Home Depot from December 2000 to January 2007. Frank Blake. "
            "Francis Stanton Frank Blake (born July 30, 1949) is an American businessman who was the "
            "chairman and CEO of The Home Depot from January 2007 to May 2014. He was a longtime "
            "protégé of Robert Nardelli."
        ),
        response=(
            "Robert Nardelli, Frank Blake's longtime protégé, was never the chairman of Chrysler; he "
            "only ever led The Home Depot."
        ),
        label="ambiguous",
        subcategory="negation_flip",
        notes="Direct negation of an explicit context fact (he WAS chairman of Chrysler).",
    ),

    # ---- entity_swap (2) ----
    Case(
        id="A9",
        source="SF:327",
        query="Deletion of αvβ8 does not result in a spontaneous inflammatory phenotype.",
        context=(
            "Integrin αvβ8-Mediated TGF-β Activation by Effector Regulatory T Cells Is Essential for "
            "Suppression of T-Cell-Mediated Inflammation. Here we show that effector Treg cells "
            "express high amounts of the integrin αvβ8, which enables them to activate latent "
            "transforming growth factor-β (TGF-β). Treg-cell-specific deletion of integrin αvβ8 did "
            "not result in a spontaneous inflammatory phenotype, suggesting that this pathway is not "
            "important in Treg-cell-mediated maintenance of immune homeostasis."
        ),
        response=(
            "Treg-cell-specific deletion of integrin αvβ6 did not result in a spontaneous inflammatory "
            "phenotype in the studied mice."
        ),
        label="ambiguous",
        subcategory="entity_swap",
        notes="Single-character entity swap (β6 vs β8); context never discusses β6.",
    ),
    Case(
        id="A10",
        source="HP:5ab3ddbc",
        query="The 2017 European Short Course Swimming Championships will be held in an Arena which began construction when?",
        context=(
            "Royal Arena. Royal Arena is a multi-use indoor arena in Ørestad Syd, a new development "
            "in the city of Copenhagen, Denmark. The ground was broken for construction on 26 June "
            "2013 and the arena opened in February 2017. 2017 European Short Course Swimming "
            "Championships. The 2017 European Short Course Swimming Championships will take place in "
            "Copenhagen, Denmark. The meet will be held in the Royal Arena, which was finished in "
            "early 2017."
        ),
        response=(
            "Construction on the Royal Arena in Copenhagen, where the 2017 European Short Course "
            "Swimming Championships will be held, broke ground on 26 July 2013."
        ),
        label="ambiguous",
        subcategory="entity_swap",
        notes="Date swap: 26 June -> 26 July (one-letter difference in the month).",
    ),
]


def case_dicts() -> List[dict]:
    """Return the case list as plain dicts for JSON-friendly downstream use."""
    return [asdict(c) for c in CASES]


# Sanity assertions executed on import - cheap and helpful
def _validate():
    by_label = {"grounded": 0, "ungrounded": 0, "ambiguous": 0}
    by_sub = {"prompt_echo": 0, "partial": 0, "parametric": 0,
              "negation_flip": 0, "entity_swap": 0}
    for c in CASES:
        assert c.label in by_label, c
        by_label[c.label] += 1
        if c.label == "ambiguous":
            assert c.subcategory in by_sub, c
            by_sub[c.subcategory] += 1
        else:
            assert c.subcategory is None, c
    assert by_label == {"grounded": 5, "ungrounded": 5, "ambiguous": 10}, by_label
    assert all(v == 2 for v in by_sub.values()), by_sub


_validate()
