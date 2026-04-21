from __future__ import annotations

import base64
from pathlib import Path

from colsearch import RENDERABLE_SOURCE_SUFFIXES, enumerate_renderable_documents, render_documents


MINIMAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/w8AAn8B9pVHtVQAAAAASUVORK5CYII="
)


def _write_png(path: Path) -> Path:
    path.write_bytes(base64.b64decode(MINIMAL_PNG_B64))
    return path


def test_public_preprocessing_helpers_render_image_bundle(tmp_path: Path) -> None:
    source = _write_png(tmp_path / "sample.png")

    inventory = enumerate_renderable_documents(tmp_path)
    result = render_documents(inventory["documents"], tmp_path / "rendered", source_root=tmp_path)

    assert ".png" in RENDERABLE_SOURCE_SUFFIXES
    assert inventory["documents"] == [source.resolve()]
    assert result["status"] == "passed"
    assert result["summary"]["documents_rendered"] == 1
    assert result["summary"]["pages_rendered"] == 1
    assert result["bundles"][0]["metadata"]["relative_source_path"] == "sample.png"
    assert Path(result["bundles"][0]["pages"][0]["image_path"]).exists()
