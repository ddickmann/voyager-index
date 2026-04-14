from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from voyager_index._internal.inference.index_core.feature_bridge import FeatureBridge


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_validation_report",
    REPO_ROOT / "scripts" / "release_validation_report.py",
)
assert SPEC is not None and SPEC.loader is not None
rvr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rvr)


def test_release_report_redacts_external_paths() -> None:
    repo_root = Path("/tmp/example-repo")
    assert rvr.describe_path(Path("/tmp/private/dataset.json"), repo_root) == "dataset.json"


def test_no_committed_validation_reports_bundle_remains() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "validation-reports"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == ""


def test_no_tracked_native_target_or_packaging_artifacts_remain() -> None:
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "research/gem_index/target",
            "src/kernels/shard_engine/target",
            "dist",
            "voyager_index.egg-info",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == ""


def test_feature_bridge_error_is_portable() -> None:
    with pytest.raises(ImportError) as exc:
        FeatureBridge()

    assert "/workspace/" not in str(exc.value)
    assert "private Voyager extension repo" in str(exc.value)


def test_root_pyproject_no_longer_packages_from_compat_src() -> None:
    payload = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "compat/src" not in payload
    assert "\"src\"" not in payload
    assert "\"src.server\"" not in payload
    assert "\"src.inference\"" not in payload
    assert "\"src.kernels\"" not in payload
    assert "voyager_index._internal.inference.index_gpu" not in payload
    assert "voyager_index._internal.inference.gym" not in payload
    assert "voyager_index._internal.inference.control" not in payload
    assert "voyager_index._internal.inference.distributed" not in payload


def test_reference_api_dockerfile_uses_root_src_tree() -> None:
    payload = (REPO_ROOT / "deploy" / "reference-api" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY src /app/src" in payload
    assert "COPY compat" not in payload
    assert "./src/kernels/knapsack_solver" in payload
    assert "latence_solver-*.whl" in payload


def test_install_docs_agree_on_pypi_distribution() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install \"voyager-index" in readme or "pip install voyager-index" in readme

    for doc_name in ["reference_api_tutorial.md", "full_feature_cookbook.md"]:
        doc = REPO_ROOT / "docs" / doc_name
        if doc.exists():
            payload = doc.read_text(encoding="utf-8")
            assert "no PyPI package is required" not in payload


def test_no_stale_github_org_urls_in_docs() -> None:
    doc_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "RELEASING.md",
    ]
    for extra in ["docs/README.md", "docs/reference_api_tutorial.md",
                   "docs/full_feature_cookbook.md", "examples/README.md"]:
        p = REPO_ROOT / extra
        if p.exists():
            doc_files.append(p)
    for path in doc_files:
        if path.exists():
            payload = path.read_text(encoding="utf-8")
            assert "latenceai/voyager-index" not in payload, f"stale org URL in {path.name}"


def test_no_test_or_benchmark_files_in_published_package() -> None:
    package_root = REPO_ROOT / "voyager_index"
    for path in package_root.rglob("test_*.py"):
        pytest.fail(f"test file in published package: {path.relative_to(REPO_ROOT)}")
    for path in package_root.rglob("benchmark_*.py"):
        pytest.fail(f"benchmark file in published package: {path.relative_to(REPO_ROOT)}")


def test_native_crate_licenses_are_consistent() -> None:
    import tomllib

    for crate_dir in ["knapsack_solver"]:
        pyproject_path = REPO_ROOT / "src" / "kernels" / crate_dir / "pyproject.toml"
        if pyproject_path.exists():
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            license_text = data.get("project", {}).get("license", {})
            if isinstance(license_text, dict):
                license_text = license_text.get("text", "")
            assert "MIT" not in str(license_text), (
                f"{crate_dir}/pyproject.toml license should be Apache-2.0, got {license_text}"
            )


def test_changelog_covers_current_version() -> None:
    import tomllib

    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    current_version = pyproject["project"]["version"]

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert current_version in changelog, (
        f"CHANGELOG.md missing entry for current version {current_version}"
    )
