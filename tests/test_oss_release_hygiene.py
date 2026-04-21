from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from colsearch._internal.inference.index_core.feature_bridge import FeatureBridge

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    import tomli as tomllib

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
            "colsearch.egg-info",
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
    assert "private extension repo" in str(exc.value)


def test_root_pyproject_no_longer_packages_from_compat_src() -> None:
    payload = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "compat/src" not in payload
    assert '"src"' not in payload
    assert '"src.server"' not in payload
    assert '"src.inference"' not in payload
    assert '"src.kernels"' not in payload
    assert "colsearch._internal.inference.index_gpu" not in payload
    assert "colsearch._internal.inference.gym" not in payload
    assert "colsearch._internal.inference.control" not in payload
    assert "colsearch._internal.inference.distributed" not in payload


def test_release_workflow_only_builds_supported_native_wheels() -> None:
    payload = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "knapsack_solver" in payload
    assert "shard_engine" in payload
    assert "publish-native" in payload
    assert "publish-root" in payload
    assert "packages-dir: dist-native/" in payload
    assert "packages-dir: dist-root/" in payload
    assert "skip-existing" not in payload
    assert "crate: hnsw_indexer" not in payload
    assert "crate: gem_router" not in payload
    assert "crate: gem_index" not in payload


def test_reference_api_dockerfile_uses_root_src_tree() -> None:
    payload = (REPO_ROOT / "deploy" / "reference-api" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY src /app/src" in payload
    assert "COPY compat" not in payload
    assert "./src/kernels/shard_engine" in payload
    assert "./src/kernels/knapsack_solver" in payload
    assert "latence_shard_engine-*.whl" in payload
    assert "latence_solver-*.whl" in payload


def test_install_docs_agree_on_pypi_distribution() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert 'pip install "colsearch' in readme or "pip install colsearch" in readme
    assert "colsearch[full]" in readme

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
    for extra in [
        "docs/README.md",
        "docs/reference_api_tutorial.md",
        "docs/full_feature_cookbook.md",
        "examples/README.md",
    ]:
        p = REPO_ROOT / extra
        if p.exists():
            doc_files.append(p)
    for path in doc_files:
        if path.exists():
            payload = path.read_text(encoding="utf-8")
            assert "latenceai/colsearch" not in payload, f"stale org URL in {path.name}"


def test_repo_governance_files_exist() -> None:
    assert (REPO_ROOT / ".github" / "CODEOWNERS").exists()
    assert (REPO_ROOT / ".github" / "dependabot.yml").exists()
    assert (REPO_ROOT / "CODE_OF_CONDUCT.md").exists()


def test_no_test_or_benchmark_files_in_published_package() -> None:
    package_root = REPO_ROOT / "colsearch"
    for path in package_root.rglob("test_*.py"):
        pytest.fail(f"test file in published package: {path.relative_to(REPO_ROOT)}")
    for path in package_root.rglob("benchmark_*.py"):
        pytest.fail(f"benchmark file in published package: {path.relative_to(REPO_ROOT)}")


def test_native_crate_licenses_are_consistent() -> None:
    for crate_dir in ["knapsack_solver", "shard_engine"]:
        pyproject_path = REPO_ROOT / "src" / "kernels" / crate_dir / "pyproject.toml"
        if pyproject_path.exists():
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            license_text = data.get("project", {}).get("license", {})
            if isinstance(license_text, dict):
                license_text = license_text.get("text", "")
            assert str(license_text) == "CC-BY-NC-4.0", (
                f"{crate_dir}/pyproject.toml license should be CC-BY-NC-4.0, got {license_text}"
            )


def test_pyproject_install_contract_matches_public_release_story() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert pyproject["project"]["version"] == "0.2.0"
    assert "full" in extras
    assert "shard-native" in extras
    assert "solver" in extras
    assert "native" in extras
    assert "latence-graph" in extras
    assert "latence-shard-engine>=0.1.6" in extras["shard-native"]
    assert "latence-solver>=0.1.6" in extras["solver"]
    assert "latence-shard-engine>=0.1.6" in extras["native"]
    assert "latence-solver>=0.1.6" in extras["native"]
    assert "latence>=0.1.1" in extras["latence-graph"]


def test_changelog_covers_current_version() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    current_version = pyproject["project"]["version"]

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert current_version in changelog, f"CHANGELOG.md missing entry for current version {current_version}"
