"""Long-context groundedness fixtures built from the benchmark-aligned case bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from research.triangular_maxsim.cases import CASES


@dataclass
class LongGroundednessCase:
    id: str
    label: str
    subcategory: Optional[str]
    query: str
    response: str
    core_case_ids: List[str]
    distractor_case_ids: List[str]
    notes: str
    token_budget: int = 7800


CASE_BY_ID = {case.id: case for case in CASES}
ALL_CASE_IDS = [case.id for case in CASES]


def _distractors(*core_case_ids: str) -> List[str]:
    core = set(core_case_ids)
    return [case_id for case_id in ALL_CASE_IDS if case_id not in core]


LONG_AMBIGUOUS_CASES: List[LongGroundednessCase] = [
    LongGroundednessCase(
        id="LG1",
        label="grounded",
        subcategory=None,
        query=CASE_BY_ID["G3"].query,
        response=CASE_BY_ID["G3"].response,
        core_case_ids=["G3"],
        distractor_case_ids=_distractors("G3"),
        notes="Long distractor-heavy context with the George Harrison evidence paragraph buried near the middle.",
    ),
    LongGroundednessCase(
        id="LG2",
        label="grounded",
        subcategory=None,
        query=CASE_BY_ID["G1"].query,
        response=CASE_BY_ID["G1"].response,
        core_case_ids=["G1"],
        distractor_case_ids=_distractors("G1"),
        notes="Long biomedical context where the relevant aptamer paragraph must survive thousands of unrelated tokens.",
    ),
    LongGroundednessCase(
        id="LG3",
        label="ungrounded",
        subcategory="entity_swap",
        query=CASE_BY_ID["A10"].query,
        response=CASE_BY_ID["A10"].response,
        core_case_ids=["A10"],
        distractor_case_ids=_distractors("A10"),
        notes="Date-swap near miss inside an ultra-long context; hard for embeddings and easy for a human to miss.",
    ),
    LongGroundednessCase(
        id="LG4",
        label="ungrounded",
        subcategory="entity_swap",
        query=CASE_BY_ID["A9"].query,
        response=CASE_BY_ID["A9"].response,
        core_case_ids=["A9"],
        distractor_case_ids=_distractors("A9"),
        notes="Single-character entity swap inside a long scientific context block.",
    ),
    LongGroundednessCase(
        id="LG5",
        label="ambiguous",
        subcategory="partial",
        query=CASE_BY_ID["A3"].query,
        response=CASE_BY_ID["A3"].response,
        core_case_ids=["A3"],
        distractor_case_ids=_distractors("A3"),
        notes="First clause is supported, second clause is a plausible biomedical extrapolation hidden inside a long context.",
    ),
    LongGroundednessCase(
        id="LG6",
        label="ambiguous",
        subcategory="partial",
        query=CASE_BY_ID["A4"].query,
        response=CASE_BY_ID["A4"].response,
        core_case_ids=["A4"],
        distractor_case_ids=_distractors("A4"),
        notes="Supported province fact mixed with an unsupported league claim under heavy distractor load.",
    ),
]
