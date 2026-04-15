from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    import tomli as tomllib


FALLBACK_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dataset_di_fixture.json"
ONTOLOGY_FIXTURE_NAME = "dataset_di_825cbaae40335bc4265a3726.json"


def resolve_fixture(repo_root: Path) -> tuple[Path, str]:
    configured = os.environ.get("VOYAGER_ONTOLOGY_FIXTURE")
    if configured:
        return Path(configured), "env"
    candidates = (
        repo_root / "tmp_data" / ONTOLOGY_FIXTURE_NAME,
        repo_root.parent / "tmp_data" / ONTOLOGY_FIXTURE_NAME,
        Path.cwd() / "tmp_data" / ONTOLOGY_FIXTURE_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate, "tmp_data"
    return FALLBACK_FIXTURE_PATH, "fallback"


def describe_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return path.name


def load_project_version(repo_root: Path) -> str:
    pyproject = repo_root / "pyproject.toml"
    with pyproject.open("rb") as handle:
        payload = tomllib.load(handle)
    return str(payload["project"]["version"])


def copy_artifacts(output_dir: Path, artifacts: list[Path]) -> list[str]:
    copied: list[str] = []
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        if not artifact.exists():
            continue
        destination = artifacts_dir / artifact.name
        shutil.copy2(artifact, destination)
        copied.append(str(destination.relative_to(output_dir)))
    return copied


def summarize_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "top_level_keys": sorted(payload.keys()),
    }
    nested_status = {
        key: value["status"]
        for key, value in payload.items()
        if isinstance(value, dict) and "status" in value
    }
    if nested_status:
        summary["nested_status"] = nested_status
    nested_keys = {
        key: sorted(value.keys())
        for key, value in payload.items()
        if isinstance(value, dict)
    }
    if nested_keys:
        summary["nested_keys"] = nested_keys
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a release validation report bundle.")
    parser.add_argument("--output-dir", required=True, help="Directory for the report bundle")
    parser.add_argument("--job-name", required=True, help="Logical validation job name")
    parser.add_argument("--benchmark-json", help="Optional benchmark JSON file to copy and summarize")
    parser.add_argument("--artifact", action="append", default=[], help="Optional artifact file to copy into the bundle")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture_path, fixture_source = resolve_fixture(repo_root)
    version = load_project_version(repo_root)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_name": args.job_name,
        "version": version,
        "fixture": {
            "path": describe_path(fixture_path, repo_root),
            "source": fixture_source,
            "exists": fixture_path.exists(),
        },
        "artifacts": [],
    }

    benchmark_path = None
    if args.benchmark_json:
        candidate = Path(args.benchmark_json).resolve()
        if candidate.exists():
            destination = output_dir / "benchmark.json"
            shutil.copy2(candidate, destination)
            benchmark_path = destination
            with candidate.open("r", encoding="utf-8") as handle:
                benchmark_payload = json.load(handle)
            report["benchmark"] = {
                "path": str(destination.relative_to(output_dir)),
                **summarize_mapping(benchmark_payload),
            }

    artifacts = [Path(item).resolve() for item in args.artifact]
    if benchmark_path is not None:
        artifacts.append(benchmark_path)
    report["artifacts"] = copy_artifacts(output_dir, artifacts)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
