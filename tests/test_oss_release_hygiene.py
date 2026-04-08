from __future__ import annotations

import importlib.util
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
    assert not (REPO_ROOT / "validation-reports").exists()


def test_feature_bridge_error_is_portable() -> None:
    with pytest.raises(ImportError) as exc:
        FeatureBridge()

    assert "/workspace/" not in str(exc.value)
    assert "private Voyager extension repo" in str(exc.value)


def test_root_pyproject_no_longer_packages_from_compat_src() -> None:
    payload = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "compat/src" not in payload


def test_reference_api_dockerfile_uses_root_src_tree() -> None:
    payload = (REPO_ROOT / "deploy" / "reference-api" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY src /app/src" in payload
    assert "COPY compat" not in payload
    assert "./src/kernels/knapsack_solver" in payload
    assert "latence_solver-*.whl" in payload


def test_install_docs_agree_on_pypi_distribution() -> None:
    oss_foundation = (REPO_ROOT / "OSS_FOUNDATION.md").read_text(encoding="utf-8")
    assert "PyPI publication is deferred" not in oss_foundation
    assert "pip install voyager-index" in oss_foundation

    docs_readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "no PyPI package is required" not in docs_readme

    tutorial = (REPO_ROOT / "docs" / "reference_api_tutorial.md").read_text(encoding="utf-8")
    assert "no PyPI package is required" not in tutorial
    assert "pip install voyager-index" in tutorial

    cookbook = (REPO_ROOT / "docs" / "full_feature_cookbook.md").read_text(encoding="utf-8")
    assert "no PyPI package is required" not in cookbook
    assert "pip install voyager-index" in cookbook


def test_no_stale_github_org_urls_in_docs() -> None:
    doc_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "OSS_FOUNDATION.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "RELEASING.md",
        REPO_ROOT / "docs" / "README.md",
        REPO_ROOT / "docs" / "reference_api_tutorial.md",
        REPO_ROOT / "docs" / "full_feature_cookbook.md",
        REPO_ROOT / "examples" / "README.md",
    ]
    for path in doc_files:
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

    for crate_dir in ["hnsw_indexer", "knapsack_solver", "gem_router"]:
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
