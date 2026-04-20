"""Tests for the PROGRESS.md helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.low_bit_roq import progress_md


SEED_PROGRESS_TEMPLATE = """\
# PROGRESS — test

## Current State

- **Phase:** 0 (harness build)
- **Most recent gate:** none
- **A-best candidates:** none yet
- **B-best candidate:** none yet
- **Production candidate:** none yet
- **Open questions:**
  - none
- **Blocked-on:** nothing

## Promoted

_(empty)_

## Killed

_(empty)_

## Open `[VERDICT-PENDING]` entries

_(empty — auto-populated when the harness emits stub entries that have not
yet been completed by an engineer)_

---
"""


def _seed(tmp_path: Path) -> Path:
    p = tmp_path / "PROGRESS.md"
    p.write_text(SEED_PROGRESS_TEMPLATE, encoding="utf-8")
    return p


def test_append_stub_inserts_entry_and_pending_bullet(tmp_path: Path):
    progress = _seed(tmp_path)
    stub = progress_md.StubEntry(
        experiment_id="t-001",
        summary="dummy summary",
        config={"alpha": 0.1, "beta": 2},
        datasets=["ArguAna"],
        seeds=3,
        baseline_name="fp16",
        metrics=[
            progress_md.MetricRow(
                name="Recall@10", baseline=0.50, this=0.51, delta=0.01, p_value=0.02
            )
        ],
    )
    progress_md.append_stub(stub, progress_path=progress)
    text = progress.read_text(encoding="utf-8")
    assert "## [" in text and "t-001" in text
    assert "[VERDICT-PENDING]" in text
    assert "`t-001` — Verdict pending" in text


def test_mark_complete_clears_pending(tmp_path: Path):
    progress = _seed(tmp_path)
    stub = progress_md.StubEntry(
        experiment_id="t-002",
        summary="another",
        config={},
        datasets=["FiQA"],
        seeds=1,
        baseline_name="roq4",
        metrics=[],
    )
    progress_md.append_stub(stub, progress_path=progress)
    progress_md.mark_complete(
        "t-002",
        verdict="PROMOTE",
        why="works on small toy",
        progress_path=progress,
    )
    text = progress.read_text(encoding="utf-8")
    assert "**Verdict:** PROMOTE" in text
    assert "**Why:** works on small toy" in text
    assert "`t-002` — Verdict pending" not in text


def test_update_promoted_and_killed_lists(tmp_path: Path):
    progress = _seed(tmp_path)
    progress_md.update_promoted("`a1-cell-007` — passes A6 gate", progress_path=progress)
    progress_md.update_killed(
        "`a1-cell-008` — fails A6 gate (recall_delta=-0.03)", progress_path=progress
    )
    text = progress.read_text(encoding="utf-8")
    assert "`a1-cell-007` — passes A6 gate" in text
    assert "`a1-cell-008` — fails A6 gate" in text


def test_update_current_state_replaces_phase(tmp_path: Path):
    progress = _seed(tmp_path)
    progress_md.update_current_state(
        phase="A (mechanics — A1 sweep running)",
        most_recent_gate="A6 — PASS",
        progress_path=progress,
    )
    text = progress.read_text(encoding="utf-8")
    assert "- **Phase:** A (mechanics — A1 sweep running)" in text
    assert "- **Most recent gate:** A6 — PASS" in text


def test_invalid_verdict_raises(tmp_path: Path):
    progress = _seed(tmp_path)
    with pytest.raises(ValueError):
        progress_md.mark_complete(
            "missing", verdict="MAYBE", why="x", progress_path=progress
        )
