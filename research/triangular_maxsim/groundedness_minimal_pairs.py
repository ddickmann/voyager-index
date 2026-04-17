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


# ----------------------------------------------------------------------
# Hard family: long contexts with distractor sentences, tightly paraphrased
# positives and negatives that share most surface tokens. These targets are
# specifically designed to be hard for dense MaxSim and require either
# literal guardrails or NLI/claim verification to separate the positive from
# the negative. Each note is suffixed with ``[HARD]`` for the harness test.
# ----------------------------------------------------------------------


_HARD_ENTITY_TEMPLATES = [
    (
        "Background notes from the {project} historical archive list several alumni from the program. "
        "Among them, {name} graduated from {university} in {year} and joined {company} the following spring. "
        "Other alumni went on to {fake_university}, but {name} did not. "
        "The archive also records earlier cohorts from the same decade.",
        "{name} graduated from {university} in {year} and joined {company}.",
        "{name} graduated from {fake_university} in {year} and joined {company}.",
        "Distractor sentence mentions the fake university; only {name}'s line states the true institution. [HARD]",
    ),
    (
        "The {company} corporate history documents three branch openings. "
        "The {city} office was opened by {name} in autumn of {year}. "
        "A neighbouring branch was opened the prior year by {fake_name} in a different city. "
        "Several smaller offices opened later in the decade.",
        "{name} opened the {city} branch of {company} in {year}.",
        "{fake_name} opened the {city} branch of {company} in {year}.",
        "{fake_name} appears in the context but with a different city. [HARD]",
    ),
]

_HARD_DATE_TEMPLATES = [
    (
        "Diplomatic records list every treaty between {country_a} and {country_b} during the {year}s. "
        "The principal accord was signed on {month} {day}, {year}. "
        "An earlier draft was initialled on {month} {alt_day}, {year} but was never ratified. "
        "Subsequent amendments followed in later years.",
        "The treaty between {country_a} and {country_b} was signed on {month} {day}, {year}.",
        "The treaty between {country_a} and {country_b} was signed on {month} {alt_day}, {year}.",
        "{alt_day} appears as a distractor (initialled draft date) inside the context. [HARD]",
    ),
    (
        "Construction notes for {bridge} include several milestone dates. "
        "Ground was broken on {month} {day}, {year} and the deck was poured four years later. "
        "Planning had begun in {alt_year} but funding was delayed. "
        "The opening ceremony was held a year after the deck pour.",
        "Construction on {bridge} began on {month} {day}, {year}.",
        "Construction on {bridge} began on {month} {day}, {alt_year}.",
        "{alt_year} appears as a distractor (planning year) inside the context. [HARD]",
    ),
]

_HARD_NUMBER_TEMPLATES = [
    (
        "Annual reports for the {project} reactor list output figures for the first decade. "
        "Year one delivered {value} megawatts, year two delivered {alt_value} megawatts after a turbine upgrade, "
        "and later years stabilised in between. Maintenance windows reduced output every fifth year.",
        "The {project} reactor produced {value} megawatts in its first year.",
        "The {project} reactor produced {alt_value} megawatts in its first year.",
        "Both values appear in the context but only {value} is attributed to year one. [HARD]",
    ),
    (
        "The {company} fleet log shows year-end vessel counts. "
        "By the end of the {year} season the count stood at {value} vessels, after a winter refit programme. "
        "By the following season the count had risen to {alt_value} vessels.",
        "The {company} fleet had {value} vessels by the end of the {year} season.",
        "The {company} fleet had {alt_value} vessels by the end of the {year} season.",
        "Both values appear, attached to different seasons. [HARD]",
    ),
]

_HARD_UNIT_TEMPLATES = [
    (
        "Engineering specifications for the {city} water main describe both metric and imperial measurements in different sections. "
        "The metric section gives the rated flow as {value} liters per second. "
        "The imperial conversion table separately notes the equivalent in gallons per second.",
        "The {city} water main delivers {value} liters per second.",
        "The {city} water main delivers {value} gallons per second.",
        "Both units appear in the context; only liters/second is the rated metric flow. [HARD]",
    ),
    (
        "Climate observations from the {valley} plateau in two different scales were tabulated for the August summary. "
        "The Celsius column peaked at {value} degrees. The Fahrenheit conversion column listed the same data in degrees Fahrenheit. "
        "Either scale can be quoted depending on the audience.",
        "Average temperature on the {valley} plateau peaked at {value} degrees Celsius.",
        "Average temperature on the {valley} plateau peaked at {value} degrees Fahrenheit.",
        "Both Celsius and Fahrenheit appear; only Celsius matches the {value} number. [HARD]",
    ),
]

_HARD_NEGATION_TEMPLATES = [
    (
        "The {study} clinical trial included three arms with different dosing protocols. "
        "In the primary endpoint analysis, {drug} reduced hospitalisations in elderly patients by a clinically meaningful margin. "
        "In a secondary subgroup of younger patients, the same drug showed no statistically significant effect. "
        "Adverse events were comparable across arms.",
        "{drug} reduced hospitalisations in elderly patients in the {study} trial.",
        "{drug} did not reduce hospitalisations in elderly patients in the {study} trial.",
        "Negative claim contradicts the primary endpoint while echoing the surface form. [HARD]",
    ),
    (
        "The {report} review evaluated {policy} across coastal and riverine deployments. "
        "On coastal flood mitigation, the policy is effective and the panel recommends scaling it up. "
        "On riverine flood mitigation, the panel found the policy is not effective in isolation. "
        "Cost-benefit analyses are reported separately.",
        "{policy} is effective for coastal flood mitigation in the {report} review.",
        "{policy} is not effective for coastal flood mitigation in the {report} review.",
        "Negation flips a polarity that the context places in a different deployment. [HARD]",
    ),
]

_HARD_ROLE_TEMPLATES = [
    (
        "The {project} program biography records two principal figures. "
        "After two seasons of close mentorship, {a} guided {b} through the program. "
        "{b} subsequently took over the {city} office while {a} returned to research. "
        "Records list both names in many places, often in passive voice.",
        "{b} was mentored by {a} during the {project} program.",
        "{a} was mentored by {b} during the {project} program.",
        "Passive voice keeps tokens identical; only argument order distinguishes positive vs negative. [HARD]",
    ),
    (
        "The patent dispute file lists the parties to the {device} sensor case. "
        "{a} filed suit against {b} over alleged infringement. "
        "{b} counterclaimed in a separate filing later that year, which was dismissed.",
        "{a} sued {b} over the patent for the {device} sensor.",
        "{b} sued {a} over the patent for the {device} sensor.",
        "Both names appear in adjacent sentences; only the suit direction matters. [HARD]",
    ),
]

_HARD_PARTIAL_TEMPLATES = [
    (
        "The {publisher} catalogue shows {book} appearing in {year_a} as a hardback first edition. "
        "The follow-up volume, {sequel}, came out in {year_b}, with the box-set reissue arriving five years later. "
        "A separately licensed paperback of {book} was distributed in {year_b} as well.",
        "The {publisher} catalogue shows {book} in {year_a} and {sequel} in {year_b}.",
        "The {publisher} catalogue shows {book} in {year_a} and {fake_sequel} in {year_b}.",
        "Distractor mentions an additional title in {year_b}, the negative substitutes the sequel name. [HARD]",
    ),
    (
        "{name} held two roles at the {institution}. "
        "Between {year_a} and {year_b} they served as both the {role_a} and the {role_b}. "
        "Earlier in their career they had also been {fake_role} at a different organisation.",
        "{name} served as the {role_a} and the {role_b} of the {institution}.",
        "{name} served as the {role_a} of the {institution} and then became {fake_role}.",
        "{fake_role} appears in the context but at a different organisation. [HARD]",
    ),
]


_ENTITY_TEMPLATES = _ENTITY_TEMPLATES + _HARD_ENTITY_TEMPLATES
_DATE_TEMPLATES = _DATE_TEMPLATES + _HARD_DATE_TEMPLATES
_NUMBER_TEMPLATES = _NUMBER_TEMPLATES + _HARD_NUMBER_TEMPLATES
_UNIT_TEMPLATES = _UNIT_TEMPLATES + _HARD_UNIT_TEMPLATES
_NEGATION_TEMPLATES = _NEGATION_TEMPLATES + _HARD_NEGATION_TEMPLATES
_ROLE_TEMPLATES = _ROLE_TEMPLATES + _HARD_ROLE_TEMPLATES
_PARTIAL_TEMPLATES = _PARTIAL_TEMPLATES + _HARD_PARTIAL_TEMPLATES


# ----------------------------------------------------------------------
# Phase J: additional hard families targeting compound facts, structured
# sources (JSON / markdown table), and distributed dialogue evidence.
# ----------------------------------------------------------------------

_HARD_COMPOUND_FACT_TEMPLATES = [
    (
        "Research log entry #{value} lists three facts about the {project} expedition. "
        "The team departed from {city} on {month} {day}, {year} under the command of Dr. {name}. "
        "Samples were collected across {country} before returning to port. "
        "The mission report filed on return is archived under log entry #{value}.",
        "The {project} expedition departed from {city} on {month} {day}, {year} under Dr. {name} and collected samples in {country}.",
        "The {project} expedition departed from {city} on {month} {day}, {year} under Dr. {name} and collected samples in {fake_country}.",
        "Three facts are conjoined; only the country is flipped. All individual surface tokens occur in the context. [HARD]",
    ),
    (
        "Shipping manifest {value} lists the {company} convoy logistics. "
        "{value} vessels departed {city} in {month} {year} carrying {alt_value} tonnes of cargo. "
        "The manifest was countersigned by {name}. "
        "Cross-references to manifests from adjacent seasons appear in the appendix.",
        "The {company} convoy had {value} vessels departing {city} in {month} {year} with {alt_value} tonnes of cargo, signed by {name}.",
        "The {company} convoy had {value} vessels departing {city} in {month} {year} with {value} tonnes of cargo, signed by {name}.",
        "Four facts conjoined; the numeric tonnage is flipped to the vessel count distractor. [HARD]",
    ),
]

_HARD_STRUCTURED_TEMPLATES = [
    (
        '{{"item": "{project}", "region": "{country}", "year": {year}, "value": {value}, "unit": "megawatts"}}',
        "{project} produced {value} megawatts in {country} in {year}.",
        "{project} produced {alt_value} megawatts in {country} in {year}.",
        "Structured JSON source: numeric value is flipped. Requires the structured-source channel. [HARD]",
    ),
    (
        "| Project | Region | Year | Value |\n"
        "|---------|--------|------|-------|\n"
        "| {project} | {country} | {year} | {value} |\n"
        "| {fake_project} | {fake_country} | {alt_year} | {alt_value} |",
        "The {project} project value in {country} for {year} was {value}.",
        "The {project} project value in {country} for {year} was {alt_value}.",
        "Markdown table: row/column lookup required; negative steals the alternative row's value. [HARD]",
    ),
    (
        '{{"name": "{name}", "institution": "{institution}", "role": "{role_a}", "year_from": {year_a}, "year_to": {year_b}}}',
        "{name} served as the {role_a} of the {institution} from {year_a} to {year_b}.",
        "{name} served as the {fake_role} of the {institution} from {year_a} to {year_b}.",
        "Structured JSON source: role is flipped to an unrelated role. [HARD]",
    ),
]

_HARD_DIALOGUE_DISTRIBUTED_TEMPLATES = [
    (
        "Transcript excerpt:\n"
        "Moderator: Let us start with the timeline.\n"
        "Dr. {name}: The {study} trial began recruiting in {year}.\n"
        "Moderator: And the primary endpoint?\n"
        "Dr. {name}: Hospitalisations were reduced by a clinically meaningful margin.\n"
        "Moderator: Any secondary signals?\n"
        "Dr. {name}: In younger subgroups we saw no significant effect, but the primary held.",
        "The {study} trial reduced hospitalisations; it began recruiting in {year}.",
        "The {study} trial reduced hospitalisations; it began recruiting in {alt_year}.",
        "Evidence is split across two speaker turns; the negative flips the start year. [HARD]",
    ),
    (
        "Transcript excerpt:\n"
        "Moderator: Tell me about the {project} team.\n"
        "{a}: I served as lead navigator through the {country} leg.\n"
        "Moderator: And the mentorship chain?\n"
        "{b}: {a} mentored me during that leg and I took over the {city} office afterwards.\n"
        "Moderator: Thank you both.",
        "{b} was mentored by {a} during the {project} program.",
        "{a} was mentored by {b} during the {project} program.",
        "Mentorship evidence sits in {b}'s turn; positive preserves the role order. [HARD]",
    ),
]


_STRUCTURED_EXTRA_SLOTS_KEYS = ("fake_project",)
_FAKE_PROJECTS = [
    "Halcyon II", "Meridian II", "Calypso II", "Atlas II",
    "Verdant II", "Luminary II", "Chronos II", "Aetherwave II",
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
        "fake_project": rng.choice(_FAKE_PROJECTS),
    }


_STRATA = [
    ("entity_swap", _ENTITY_TEMPLATES),
    ("date_swap", _DATE_TEMPLATES),
    ("number_swap", _NUMBER_TEMPLATES),
    ("unit_swap", _UNIT_TEMPLATES),
    ("negation", _NEGATION_TEMPLATES),
    ("role_swap", _ROLE_TEMPLATES),
    ("partial", _PARTIAL_TEMPLATES),
    ("hard_compound_facts", _HARD_COMPOUND_FACT_TEMPLATES),
    ("hard_structured", _HARD_STRUCTURED_TEMPLATES),
    ("hard_dialogue_distributed", _HARD_DIALOGUE_DISTRIBUTED_TEMPLATES),
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


def hard_family_summary(pairs: Sequence[MinimalPair]) -> dict:
    """Per-stratum count of pairs sourced from the ``[HARD]`` template family."""

    counts: dict = {}
    for pair in pairs:
        if "[HARD]" in pair.notes:
            counts[pair.stratum] = counts.get(pair.stratum, 0) + 1
    return counts


__all__ = [
    "MinimalPair",
    "build_minimal_pairs",
    "stratum_summary",
    "hard_family_summary",
]
