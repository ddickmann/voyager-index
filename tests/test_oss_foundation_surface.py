from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from colsearch import (
    DEFAULT_MULTIMODAL_MODEL,
    DEFAULT_MULTIMODAL_MODEL_SPEC,
    SUPPORTED_MULTIMODAL_MODELS,
    fast_colbert_scores,
)
from colsearch._internal.inference.engines import BM25Engine
from colsearch._internal.inference.engines.bm25s_engine import BM25sEngine
from colsearch._internal.inference.index_core.feature_bridge import MaxSimBridge


def test_public_multimodal_model_matrix_is_explicit() -> None:
    assert set(SUPPORTED_MULTIMODAL_MODELS) == {
        "colqwen3",
        "collfm2",
        "nemotron_colembed",
    }
    assert DEFAULT_MULTIMODAL_MODEL == "collfm2"
    assert DEFAULT_MULTIMODAL_MODEL_SPEC is SUPPORTED_MULTIMODAL_MODELS["collfm2"]
    assert DEFAULT_MULTIMODAL_MODEL_SPEC.pooling_task == "token_embed"
    assert SUPPORTED_MULTIMODAL_MODELS["colqwen3"].pooling_task == "token_embed"
    assert "vllm serve" in SUPPORTED_MULTIMODAL_MODELS["nemotron_colembed"].serve_command


def test_fast_colbert_scores_prefers_the_matching_document() -> None:
    query = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    documents = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    ])

    scores = fast_colbert_scores(query, documents)

    assert tuple(scores.shape) == (1, 2)
    assert float(scores[0, 0]) > float(scores[0, 1])


def test_maxsim_bridge_uses_shared_kernel_path() -> None:
    bridge = MaxSimBridge(device="cpu")
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    candidates = [
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    ]

    with patch.object(MaxSimBridge, "_maxsim_cpu", side_effect=AssertionError("unexpected fallback")):
        scores = bridge.compute_maxsim_scores(query, candidates)

    assert tuple(scores.shape) == (2,)
    assert float(scores[0]) > float(scores[1])


def test_sparse_engine_surface_points_to_bm25s() -> None:
    assert BM25Engine is BM25sEngine


def test_deprecated_src_namespace_is_not_shipped() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "\"src\"" not in pyproject
    assert "\"src.server\"" not in pyproject
    assert "\"src.inference\"" not in pyproject
    assert "\"src.kernels\"" not in pyproject


def test_public_voyager_modules_do_not_import_src_directly() -> None:
    package_root = Path(__file__).resolve().parents[1] / "colsearch"
    for module_name in ("config.py", "search.py", "kernels.py", "server.py"):
        source = (package_root / module_name).read_text(encoding="utf-8")
        assert "from src." not in source
        assert "import src." not in source


def test_repo_no_longer_contains_mixed_runtime_python_trees() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "src" / "__init__.py").exists()
    assert not (repo_root / "src" / "inference" / "__init__.py").exists()
    assert not (repo_root / "src" / "server" / "__init__.py").exists()
    assert not (repo_root / "src" / "kernels" / "__init__.py").exists()
    assert not (repo_root / "src" / "inference" / "index_core").exists()
    assert not (repo_root / "src" / "server" / "main.py").exists()
    assert not (repo_root / "src" / "kernels" / "triton_maxsim.py").exists()
    assert not (repo_root / "src" / "kernels" / "triton_roq.py").exists()
    assert not (repo_root / "colsearch" / "_internal" / "inference" / "solver").exists()
    assert not (repo_root / "colsearch" / "_internal" / "inference" / "index_gpu").exists()
    assert not (repo_root / "colsearch" / "_internal" / "inference" / "gym").exists()
    assert not (repo_root / "colsearch" / "_internal" / "inference" / "control").exists()
    assert not (repo_root / "colsearch" / "_internal" / "inference" / "distributed").exists()
    assert (repo_root / "deploy" / "reference-api" / "Dockerfile").exists()


def test_tmp_data_notebook_tutorial_exists() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "02_tmp_data_full_api_tutorial.ipynb"
    assert notebook_path.exists()
