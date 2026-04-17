"""Deterministic minimal-pair fixtures for Phase E groundedness evaluation.

Each pair contains a shared context plus two responses: one ``positive``
(grounded) and one ``negative`` (intentionally broken along the named
``stratum``). Pairs are generated via small text templates so the suite stays
fully reproducible without external downloads.

Strata covered (>= 200 pairs total):

- ``entity_swap``    : a named entity is swapped for a near-miss
- ``date_swap``      : a date is shifted by a small delta
- ``number_swap``    : a numeric value is altered
- ``unit_swap``      : the unit is replaced with a different unit
- ``negation``       : a positive assertion is negated
- ``role_swap``      : two role-bearing entities are exchanged
- ``partial``        : the positive answer is split, one clause is dropped

The generator is deterministic for a given ``seed`` and emits stable IDs of
the form ``MP-{stratum}-{index:04d}``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Sequence


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MinimalPair:
    """Single (context, positive, negative) triple in a labelled stratum."""

    pair_id: str
    stratum: str
    context: str
    positive: str
    negative: str
    notes: str

    def signature(self) -> str:
        """Stable content signature used for de-dup and snapshot tests."""

        payload = "|".join((self.stratum, self.context, self.positive, self.negative))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------


_ENTITY_TEMPLATES = [
    (
        "{name} graduated from {university} in {year} and joined {company} the following spring.",
        "{name} graduated from {university} in {year} and then joined {company}.",
        "{name} graduated from {fake_university} in {year} and then joined {company}.",
        "Renames the university to a near-miss institution.",
    ),
    (
        "The {city} branch of {company} was opened by {name} in the autumn of {year}.",
        "{name} opened the {city} branch of {company} in {year}.",
        "{fake_name} opened the {city} branch of {company} in {year}.",
        "Renames the founder to a near-miss person.",
    ),
    (
        "Dr. {name} led the {project} expedition through {country} in {year}.",
        "Dr. {name} led the {project} expedition through {country} in {year}.",
        "Dr. {name} led the {project} expedition through {fake_country} in {year}.",
        "Renames the destination country to a near-miss country.",
    ),
]

_DATE_TEMPLATES = [
    (
        "The treaty between {country_a} and {country_b} was signed on {month} {day}, {year}.",
        "The treaty between {country_a} and {country_b} was signed on {month} {day}, {year}.",
        "The treaty between {country_a} and {country_b} was signed on {month} {alt_day}, {year}.",
        "Day-of-month shift by a small offset.",
    ),
    (
        "Construction on the {bridge} began on {month} {day}, {year} and finished four years later.",
        "Construction on the {bridge} began on {month} {day}, {year}.",
        "Construction on the {bridge} began on {month} {day}, {alt_year}.",
        "Year off by one.",
    ),
    (
        "The {satellite} satellite reached orbit on {month} {day}, {year}.",
        "The {satellite} satellite reached orbit on {month} {day}, {year}.",
        "The {satellite} satellite reached orbit on {alt_month} {day}, {year}.",
        "Month swapped to an adjacent month.",
    ),
]

_NUMBER_TEMPLATES = [
    (
        "The {project} reactor produced {value} megawatts in its first year of operation.",
        "The {project} reactor produced {value} megawatts in its first year.",
        "The {project} reactor produced {alt_value} megawatts in its first year.",
        "Magnitude bumped by ~10%.",
    ),
    (
        "The {company} fleet expanded to {value} vessels by the end of the {year} season.",
        "The {company} fleet had {value} vessels by the end of the {year} season.",
        "The {company} fleet had {alt_value} vessels by the end of the {year} season.",
        "Counted entities altered.",
    ),
    (
        "{name} completed the marathon in a time of {value} minutes flat.",
        "{name} finished the marathon in {value} minutes.",
        "{name} finished the marathon in {alt_value} minutes.",
        "Numeric measurement perturbed.",
    ),
]

_UNIT_TEMPLATES = [
    (
        "The {city} water main delivers {value} liters per second to the central district.",
        "The {city} water main delivers {value} liters per second.",
        "The {city} water main delivers {value} gallons per second.",
        "Unit swap (liters -> gallons).",
    ),
    (
        "Average temperature on the {valley} plateau peaked at {value} degrees Celsius last August.",
        "Average temperature on the {valley} plateau peaked at {value} degrees Celsius.",
        "Average temperature on the {valley} plateau peaked at {value} degrees Fahrenheit.",
        "Unit swap (Celsius -> Fahrenheit).",
    ),
    (
        "The {project} pipeline runs for {value} kilometers across the desert basin.",
        "The {project} pipeline runs for {value} kilometers across the basin.",
        "The {project} pipeline runs for {value} miles across the basin.",
        "Unit swap (kilometers -> miles).",
    ),
]

_NEGATION_TEMPLATES = [
    (
        "The {study} clinical trial confirmed that {drug} reduces hospitalisations in elderly patients.",
        "The {study} trial confirmed that {drug} reduces hospitalisations in elderly patients.",
        "The {study} trial confirmed that {drug} does not reduce hospitalisations in elderly patients.",
        "Adds an explicit negation to the claim.",
    ),
    (
        "The {report} review concluded that {policy} is effective for coastal flood mitigation.",
        "The {report} review concluded that {policy} is effective for coastal flood mitigation.",
        "The {report} review concluded that {policy} is not effective for coastal flood mitigation.",
        "Inserts negation in the verb phrase.",
    ),
    (
        "Field tests showed that the {model} engine starts reliably below freezing.",
        "The {model} engine starts reliably below freezing in field tests.",
        "The {model} engine never starts reliably below freezing in field tests.",
        "Replaces positive adverb with hard negation.",
    ),
]

_ROLE_TEMPLATES = [
    (
        "{a} mentored {b} during the {project} program; {b} later led the {city} office.",
        "{a} mentored {b} during the {project} program.",
        "{b} mentored {a} during the {project} program.",
        "Role reversal between two named people.",
    ),
    (
        "In the dispute, {a} sued {b} over the patent for the {device} sensor.",
        "{a} sued {b} over the patent for the {device} sensor.",
        "{b} sued {a} over the patent for the {device} sensor.",
        "Plaintiff and defendant swapped.",
    ),
    (
        "The {team} coach hired {a} as captain after benching {b} for two seasons.",
        "The {team} coach hired {a} as captain after benching {b}.",
        "The {team} coach hired {b} as captain after benching {a}.",
        "Promoted vs benched roles swapped.",
    ),
]

_PARTIAL_TEMPLATES = [
    (
        "{name} served as both the {role_a} and the {role_b} of the {institution} between {year_a} and {year_b}.",
        "{name} served as both the {role_a} and the {role_b} of the {institution}.",
        "{name} served as the {role_a} of the {institution} and then became {fake_role}.",
        "Half the answer is grounded; the other half is replaced with a fabrication.",
    ),
    (
        "The {project} initiative covered {region_a} in its first year and {region_b} in its second year.",
        "The {project} covered {region_a} in its first year and {region_b} in its second year.",
        "The {project} covered {region_a} in its first year and {fake_region} in its second year.",
        "Second clause replaced with an unsupported region.",
    ),
    (
        "The {publisher} catalogue lists {book} in {year_a} and {sequel} in {year_b}.",
        "The {publisher} catalogue lists {book} in {year_a} and {sequel} in {year_b}.",
        "The {publisher} catalogue lists {book} in {year_a} and {fake_sequel} in {year_b}.",
        "Sequel title swapped while the rest stays grounded.",
    ),
]


_NAMES = [
    "Alexandra Morton", "Benedikt Hauser", "Camille Okafor", "Dilan Petrov",
    "Eira Nakamura", "Felipe Villaverde", "Gisele Toussaint", "Hiro Tanaka",
    "Iris Belmondo", "Jens Kowalski", "Karima Reuter", "Lakshmi Aiyar",
    "Marius Sandberg", "Nora Achterberg", "Owen Whitcombe", "Priya Ranjan",
]
_FAKE_NAMES = [
    "Alexandr Morten", "Benedikta Hauserin", "Camillo Okafa", "Diana Petrov",
    "Eiren Nakamura", "Felipa Villaverde", "Gisele Toussant", "Hira Tanaka",
    "Iras Belmondo", "Jen Kowalsky", "Karim Reuter", "Lakshman Aiyar",
    "Marie Sandberg", "Norah Achterberg", "Olwen Whitcombe", "Pria Ranjan",
]
_UNIVERSITIES = [
    "Uppsala University", "Universidad de Chile", "Heidelberg University",
    "Kyoto University", "ETH Zurich", "TU Delft", "McGill University",
    "Tsinghua University", "Bocconi University", "Karolinska Institute",
]
_FAKE_UNIVERSITIES = [
    "Uppsala State College", "Universidad Central de Chile", "Hesselberg University",
    "Kyoto Polytechnic", "ETH Geneva", "TU Eindhoven", "McGill Polytechnic",
    "Tsinghua Normal University", "Bocconi College", "Karolinska College",
]
_COMPANIES = [
    "Lumen Robotics", "Northstar Energy", "Pacific Logistics", "Serica Biotech",
    "Aurora Composites", "Helios Press", "Vespertine Foods", "Quantica Maritime",
]
_CITIES = [
    "Reykjavik", "Adelaide", "Lyon", "Belem", "Tashkent", "Edinburgh",
    "Daejeon", "Cluj-Napoca", "Bilbao", "Kanazawa",
]
_COUNTRIES = [
    "Mongolia", "Uruguay", "Estonia", "Cambodia", "Slovenia", "Botswana",
    "Iceland", "Brunei", "Suriname", "Latvia",
]
_FAKE_COUNTRIES = [
    "Mongolia East", "Uruguay Central", "Estonia North", "Cambodia West",
    "Slovenia South", "Botswana East", "Iceland West", "Brunei North",
    "Suriname Central", "Latvia South",
]
_PROJECTS = [
    "Halcyon", "Meridian", "Calypso", "Atlas", "Verdant", "Luminary",
    "Chronos", "Aetherwave", "Daystar", "Riftway",
]
_BRIDGES = [
    "Strathmore Bridge", "Ladenburg Span", "Calder Crossing",
    "North Mole Bridge", "Pelham Viaduct",
]
_SATELLITES = [
    "Tycho-7", "Hyperion-3", "Auriga-12", "Carina-5", "Borealis-1",
]
_VALLEYS = [
    "Caracara", "Tigris-East", "Cygnet", "Bauplan", "Quennell",
]
_DRUGS = [
    "norventine", "calixate-B", "ritolimab", "phelerin", "dynocort",
]
_STUDIES = [
    "PROVEN-7", "ATLAS-9", "MERIDIAN-3", "ARROW-5", "CRESCENT-2",
]
_REPORTS = [
    "Hartmann-Liu", "Okonkwo-Reyes", "Sandberg-Yamamoto", "Bertrand-Patel",
]
_POLICIES = [
    "elevated levee construction", "managed coastal retreat",
    "tidal-gate retrofitting", "saltmarsh restoration",
]
_MODELS = [
    "Vector-9", "Tundra Mark VI", "Borealis Plus", "Helios LX", "Halcyon-X",
]
_TEAMS = [
    "Riverton United", "North Mole Athletic", "Capeford Rovers", "Strathmore Lions",
]
_INSTITUTIONS = [
    "Verdant Foundation", "North Mole Society", "Halcyon Council",
    "Atlas Trust", "Meridian Institute",
]
_REGIONS = [
    "the Strathmore lowlands", "the Caracara highlands", "the Cygnet basin",
    "the North Mole flats", "the Quennell coast",
]
_FAKE_REGIONS = [
    "the Strathmore uplands", "the Caracara basin", "the Cygnet plateau",
    "the North Mole shoals", "the Quennell hinterland",
]
_BOOKS = [
    "Halcyon Days", "Meridian Drift", "Calypso Wake", "Atlas in Bloom",
    "Verdant Hours",
]
_SEQUELS = [
    "Halcyon Nights", "Meridian Tide", "Calypso Reckoning", "Atlas in Twilight",
    "Verdant Decades",
]
_FAKE_SEQUELS = [
    "Halcyon Mornings", "Meridian Foam", "Calypso Echo", "Atlas in Stone",
    "Verdant Currents",
]
_PUBLISHERS = [
    "Lumen Press", "Northstar Books", "Pacific House", "Serica Editions",
]
_DEVICES = [
    "stratospheric pressure", "halide-doped infrared", "axial-flux torque",
    "magneto-resistive flow",
]
_ROLES = [
    "chair", "treasurer", "executive director", "head curator",
    "managing trustee",
]
_FAKE_ROLES = [
    "honorary patron", "founding navigator", "deputy archivist",
    "interim envoy",
]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------


def _alt_month(month: str) -> str:
    idx = _MONTHS.index(month)
    return _MONTHS[(idx + 1) % len(_MONTHS)]


def _slot_factory(rng: random.Random) -> dict:
    name = rng.choice(_NAMES)
    fake_name = rng.choice([n for n in _FAKE_NAMES if n != name])
    a = rng.choice(_NAMES)
    b = rng.choice([n for n in _NAMES if n != a])
    year = rng.randint(1965, 2018)
    alt_year = year + rng.choice([-2, -1, 1, 2])
    day = rng.randint(2, 27)
    alt_day = day + rng.choice([-1, 1, 2])
    month = rng.choice(_MONTHS)
    value = rng.choice([12, 18, 24, 36, 50, 75, 120, 240, 360, 480, 950])
    alt_value = value + max(1, value // 8)
    return {
        "name": name,
        "fake_name": fake_name,
        "a": a,
        "b": b,
        "university": rng.choice(_UNIVERSITIES),
        "fake_university": rng.choice(_FAKE_UNIVERSITIES),
        "company": rng.choice(_COMPANIES),
        "city": rng.choice(_CITIES),
        "country": rng.choice(_COUNTRIES),
        "country_a": rng.choice(_COUNTRIES),
        "country_b": rng.choice(_COUNTRIES),
        "fake_country": rng.choice(_FAKE_COUNTRIES),
        "year": str(year),
        "alt_year": str(alt_year),
        "year_a": str(year),
        "year_b": str(year + rng.randint(2, 6)),
        "month": month,
        "alt_month": _alt_month(month),
        "day": str(day),
        "alt_day": str(alt_day),
        "value": str(value),
        "alt_value": str(alt_value),
        "project": rng.choice(_PROJECTS),
        "bridge": rng.choice(_BRIDGES),
        "satellite": rng.choice(_SATELLITES),
        "valley": rng.choice(_VALLEYS),
        "drug": rng.choice(_DRUGS),
        "study": rng.choice(_STUDIES),
        "report": rng.choice(_REPORTS),
        "policy": rng.choice(_POLICIES),
        "model": rng.choice(_MODELS),
        "team": rng.choice(_TEAMS),
        "device": rng.choice(_DEVICES),
        "institution": rng.choice(_INSTITUTIONS),
        "role_a": rng.choice(_ROLES),
        "role_b": rng.choice([r for r in _ROLES if r != "chair"]),
        "fake_role": rng.choice(_FAKE_ROLES),
        "region_a": rng.choice(_REGIONS),
        "region_b": rng.choice([r for r in _REGIONS if r != "the Cygnet basin"]),
        "fake_region": rng.choice(_FAKE_REGIONS),
        "book": rng.choice(_BOOKS),
        "sequel": rng.choice(_SEQUELS),
        "fake_sequel": rng.choice(_FAKE_SEQUELS),
        "publisher": rng.choice(_PUBLISHERS),
    }


_STRATA = [
    ("entity_swap", _ENTITY_TEMPLATES),
    ("date_swap", _DATE_TEMPLATES),
    ("number_swap", _NUMBER_TEMPLATES),
    ("unit_swap", _UNIT_TEMPLATES),
    ("negation", _NEGATION_TEMPLATES),
    ("role_swap", _ROLE_TEMPLATES),
    ("partial", _PARTIAL_TEMPLATES),
]


def build_minimal_pairs(
    *,
    pairs_per_stratum: int = 30,
    seed: int = 17,
) -> List[MinimalPair]:
    """Build the deterministic minimal-pair fixture.

    The default ``pairs_per_stratum=30`` yields ``7 * 30 = 210`` pairs which
    exceeds the Phase E lower bound of 200 minimal pairs.
    """

    rng = random.Random(seed)
    pairs: List[MinimalPair] = []
    seen: set = set()
    for stratum, templates in _STRATA:
        emitted = 0
        attempts = 0
        while emitted < pairs_per_stratum:
            attempts += 1
            if attempts > pairs_per_stratum * 64:
                break
            template = templates[(emitted + attempts) % len(templates)]
            slots = _slot_factory(rng)
            try:
                context = template[0].format(**slots)
                positive = template[1].format(**slots)
                negative = template[2].format(**slots)
            except KeyError:
                continue
            if positive == negative:
                continue
            sig = (stratum, positive, negative)
            if sig in seen:
                continue
            seen.add(sig)
            pair = MinimalPair(
                pair_id="MP-{stratum}-{idx:04d}".format(stratum=stratum, idx=emitted),
                stratum=stratum,
                context=context,
                positive=positive,
                negative=negative,
                notes=template[3],
            )
            pairs.append(pair)
            emitted += 1
    return pairs


def stratum_summary(pairs: Sequence[MinimalPair]) -> dict:
    """Per-stratum count summary; useful for tests and reports."""

    counts: dict = {}
    for pair in pairs:
        counts[pair.stratum] = counts.get(pair.stratum, 0) + 1
    return counts


__all__ = [
    "MinimalPair",
    "build_minimal_pairs",
    "stratum_summary",
]
