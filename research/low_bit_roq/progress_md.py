"""
PROGRESS.md helper.

The Phase 0 harness (and any other runner) calls ``append_stub`` after each
experiment. The stub contains the parts the harness can fill in
mechanically: config, dataset/seed list, the metrics table, the artifact
links, and a ``[VERDICT-PENDING]`` marker. The engineer fills in Verdict +
Why + Gate-impact within one working day, otherwise the entry stays
flagged in the Current-State header.

Entry format mirrors the template in the plan's "Continuous progress log"
section. We intentionally write plain markdown rather than a structured
templating system so the file remains hand-editable.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROGRESS_PATH_DEFAULT = Path(__file__).resolve().parent / "PROGRESS.md"
REPORTS_DIR_DEFAULT = Path(__file__).resolve().parent / "reports"

_PENDING_MARKER = "[VERDICT-PENDING]"


@dataclass
class MetricRow:
    """One row of the per-experiment table that goes into PROGRESS.md."""

    name: str
    baseline: float | str
    this: float | str
    delta: float | str = ""
    p_value: float | str = ""

    @staticmethod
    def fmt(value: float | str, *, kind: str = "auto") -> str:
        if isinstance(value, str):
            return value
        if value != value:  # NaN
            return "—"
        if kind == "pct":
            return f"{value:+.1%}" if value < 1 and value > -1 else f"{value:+.0%}"
        if kind == "p":
            if value < 1e-3:
                return "<1e-3"
            return f"{value:.2g}"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}"


@dataclass
class StubEntry:
    """Auto-stub data passed by a runner to ``append_stub``."""

    experiment_id: str
    summary: str
    config: Mapping[str, Any]
    datasets: Sequence[str]
    seeds: int
    baseline_name: str
    metrics: Sequence[MetricRow]
    artifacts: Sequence[Path | str] = field(default_factory=list)
    gate: str = "informs nothing"
    timestamp: _dt.datetime | None = None


def _format_config(config: Mapping[str, Any]) -> str:
    parts = []
    for key, value in config.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_metrics_table(rows: Iterable[MetricRow]) -> str:
    header = (
        "| metric             | baseline | this   | delta   | p (paired bootstrap) |\n"
        "|--------------------|---------:|-------:|--------:|---------------------:|\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {name:<18} | {baseline:>8} | {this:>6} | {delta:>7} | {p:>20} |".format(
                name=row.name,
                baseline=MetricRow.fmt(row.baseline),
                this=MetricRow.fmt(row.this),
                delta=MetricRow.fmt(row.delta),
                p=MetricRow.fmt(row.p_value, kind="p") if row.p_value != "" else "",
            )
        )
    return header + "\n".join(body) + "\n"


def render_entry(stub: StubEntry) -> str:
    timestamp = stub.timestamp or _dt.datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
    artifact_links = ", ".join(
        f"[`{Path(p).name}`]({Path(p).as_posix()})" for p in stub.artifacts
    ) or "—"
    return (
        f"## [{timestamp_str}] {stub.experiment_id} — {stub.summary}\n\n"
        f"**Config:** {_format_config(stub.config)}\n"
        f"**Datasets / seeds:** {' | '.join(stub.datasets)}, n={stub.seeds} seeds\n"
        f"**Baseline:** {stub.baseline_name}\n\n"
        f"{_format_metrics_table(stub.metrics)}\n"
        f"**Verdict:** {_PENDING_MARKER} (PROMOTE / KEEP-EXPERIMENTAL / KILL)\n"
        f"**Why:** {_PENDING_MARKER} — fill within 1 working day. "
        f"What physically caused the result, what surprised us, what's next.\n"
        f"**Artifacts:** {artifact_links}\n"
        f"**Gate impact:** {stub.gate}\n"
    )


def _split_at_separator(text: str) -> tuple[str, str]:
    """Split PROGRESS.md into (pinned-headers-block, log-block) at the first
    standalone ``---`` line that follows the headers."""
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError(
            "PROGRESS.md is missing the '---' separator between pinned "
            "headers and the chronological log. Re-initialize with the "
            "template in this file's docstring."
        )
    return parts[0], parts[1]


def append_stub(
    stub: StubEntry,
    *,
    progress_path: Path | str = PROGRESS_PATH_DEFAULT,
    list_in_pending: bool = True,
) -> Path:
    """Append a stub entry. Newest-first ordering (immediately after the
    separator). Optionally adds the entry to the
    ``Open [VERDICT-PENDING] entries`` pinned list.
    """
    path = Path(progress_path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not text:
        raise FileNotFoundError(
            f"{path} not found. Initialize it with the template before calling append_stub()."
        )
    headers, log = _split_at_separator(text)

    if list_in_pending:
        headers = _add_to_pending_list(headers, stub.experiment_id)

    rendered = render_entry(stub)
    new_log = "\n" + rendered.rstrip() + "\n\n" + log.lstrip("\n")
    path.write_text(headers + "\n---\n" + new_log, encoding="utf-8")
    return path


def _add_to_pending_list(headers: str, experiment_id: str) -> str:
    bullet = f"- `{experiment_id}` — Verdict pending"
    section_re = re.compile(
        r"(## Open `\[VERDICT-PENDING\]` entries\s*\n)((?:.|\n)*?)(?=\n##|\Z)",
        re.MULTILINE,
    )
    match = section_re.search(headers)
    if not match:
        return headers
    body = match.group(2).strip()
    if "_(empty" in body:
        body = bullet
    elif bullet in body:
        return headers
    else:
        body = body + "\n" + bullet
    return section_re.sub(match.group(1) + body + "\n\n", headers)


def mark_complete(
    experiment_id: str,
    verdict: str,
    why: str,
    gate_impact: str | None = None,
    *,
    progress_path: Path | str = PROGRESS_PATH_DEFAULT,
) -> Path:
    """Replace the ``[VERDICT-PENDING]`` markers in the entry for
    ``experiment_id`` with the engineer-supplied Verdict + Why, and remove
    the entry from the pending list.
    """
    if verdict not in {"PROMOTE", "KEEP-EXPERIMENTAL", "KILL"}:
        raise ValueError("verdict must be one of PROMOTE / KEEP-EXPERIMENTAL / KILL")
    path = Path(progress_path)
    text = path.read_text(encoding="utf-8")
    headers, log = _split_at_separator(text)
    headers = _remove_from_pending_list(headers, experiment_id)

    entry_re = re.compile(
        rf"(## \[[^\]]+\] {re.escape(experiment_id)} —[\s\S]*?)(?=\n## \[|\Z)",
        re.MULTILINE,
    )
    match = entry_re.search(log)
    if not match:
        raise KeyError(f"No entry found for experiment_id={experiment_id}")
    entry = match.group(1)
    entry = re.sub(
        rf"\*\*Verdict:\*\* {re.escape(_PENDING_MARKER)}.*$",
        f"**Verdict:** {verdict}",
        entry,
        flags=re.MULTILINE,
    )
    entry = re.sub(
        rf"\*\*Why:\*\* {re.escape(_PENDING_MARKER)}.*$",
        f"**Why:** {why}",
        entry,
        flags=re.MULTILINE,
    )
    if gate_impact:
        entry = re.sub(
            r"\*\*Gate impact:\*\*.*$",
            f"**Gate impact:** {gate_impact}",
            entry,
            flags=re.MULTILINE,
        )
    log = log[: match.start()] + entry + log[match.end():]
    path.write_text(headers + "\n---\n" + log, encoding="utf-8")
    return path


def _remove_from_pending_list(headers: str, experiment_id: str) -> str:
    section_re = re.compile(
        r"(## Open `\[VERDICT-PENDING\]` entries\s*\n)((?:.|\n)*?)(?=\n##|\Z)",
        re.MULTILINE,
    )
    match = section_re.search(headers)
    if not match:
        return headers
    body = match.group(2)
    new_body_lines = [
        line for line in body.splitlines() if f"`{experiment_id}`" not in line
    ]
    new_body = "\n".join(new_body_lines).strip()
    if not new_body:
        new_body = "_(empty — auto-populated when the harness emits stub entries that have not\nyet been completed by an engineer)_"
    return section_re.sub(match.group(1) + new_body + "\n\n", headers)


def stub_from_report_json(report_path: Path | str) -> StubEntry:
    """Build a StubEntry from a structured JSON report.

    The harness is expected to dump JSON in the schema documented in
    ``research/low_bit_roq/reports/SCHEMA.md`` (top-level keys: ``id``,
    ``summary``, ``config``, ``datasets``, ``seeds``, ``baseline``,
    ``metrics``, ``artifacts``, ``gate``).
    """
    path = Path(report_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = [MetricRow(**row) for row in data["metrics"]]
    artifacts = list(data.get("artifacts", []))
    if path.as_posix() not in {Path(a).as_posix() for a in artifacts}:
        artifacts.insert(0, path)
    return StubEntry(
        experiment_id=data["id"],
        summary=data["summary"],
        config=data["config"],
        datasets=data["datasets"],
        seeds=int(data["seeds"]),
        baseline_name=data["baseline"],
        metrics=metrics,
        artifacts=artifacts,
        gate=data.get("gate", "informs nothing"),
    )


def append_stub_from_report(
    report_path: Path | str,
    *,
    progress_path: Path | str = PROGRESS_PATH_DEFAULT,
) -> Path:
    return append_stub(
        stub_from_report_json(report_path), progress_path=progress_path
    )


def update_promoted(
    bullet: str, *, progress_path: Path | str = PROGRESS_PATH_DEFAULT
) -> Path:
    return _append_to_pinned(progress_path, "## Promoted", bullet)


def update_killed(
    bullet: str, *, progress_path: Path | str = PROGRESS_PATH_DEFAULT
) -> Path:
    return _append_to_pinned(progress_path, "## Killed", bullet)


def _append_to_pinned(progress_path: Path | str, section: str, bullet: str) -> Path:
    path = Path(progress_path)
    text = path.read_text(encoding="utf-8")
    section_re = re.compile(
        rf"({re.escape(section)}\s*\n)((?:.|\n)*?)(?=\n##|\Z)",
        re.MULTILINE,
    )
    match = section_re.search(text)
    if not match:
        raise KeyError(f"Section {section!r} not found in {path}")
    body = match.group(2).strip()
    bullet_line = f"- {bullet}" if not bullet.startswith("- ") else bullet
    if "_(empty" in body:
        body = bullet_line
    elif bullet_line in body:
        return path
    else:
        body = body + "\n" + bullet_line
    text = section_re.sub(match.group(1) + body + "\n\n", text)
    path.write_text(text, encoding="utf-8")
    return path


def update_current_state(
    *,
    phase: str | None = None,
    most_recent_gate: str | None = None,
    a_best: Sequence[str] | None = None,
    b_best: str | None = None,
    production_candidate: str | None = None,
    open_questions: Sequence[str] | None = None,
    blocked_on: str | None = None,
    progress_path: Path | str = PROGRESS_PATH_DEFAULT,
) -> Path:
    """Overwrite the Current State pinned block. Pass only the fields you
    want to update; ``None`` leaves the existing value untouched."""
    path = Path(progress_path)
    text = path.read_text(encoding="utf-8")
    section_re = re.compile(
        r"(## Current State\s*\n)((?:.|\n)*?)(?=\n##|\Z)", re.MULTILINE
    )
    match = section_re.search(text)
    if not match:
        raise KeyError("Current State section not found")
    existing = match.group(2)

    def _replace(field: str, new: str | None) -> None:
        nonlocal existing
        if new is None:
            return
        line_re = re.compile(rf"(- \*\*{re.escape(field)}:\*\* ).*$", re.MULTILINE)
        if line_re.search(existing):
            existing = line_re.sub(rf"\g<1>{new}", existing)

    _replace("Phase", phase)
    _replace("Most recent gate", most_recent_gate)
    _replace("B-best candidate", b_best)
    _replace("Production candidate", production_candidate)
    _replace("Blocked-on", blocked_on)

    if a_best is not None:
        a_text = ", ".join(a_best) if a_best else "none yet"
        existing = re.sub(
            r"(- \*\*A-best candidates:\*\* ).*$",
            rf"\g<1>{a_text}",
            existing,
            flags=re.MULTILINE,
        )
    if open_questions is not None:
        bullets = "\n".join(f"  - {q}" for q in open_questions) if open_questions else "  - none"
        existing = re.sub(
            r"(- \*\*Open questions:\*\*\s*\n)(?:  - .*\n)+",
            rf"\g<1>{bullets}\n",
            existing,
        )
    text = section_re.sub(match.group(1) + existing.rstrip() + "\n\n", text)
    path.write_text(text, encoding="utf-8")
    return path
