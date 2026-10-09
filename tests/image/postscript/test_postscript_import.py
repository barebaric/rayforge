import io
import shutil
import subprocess
from pathlib import Path
from typing import cast

import cairo
import pytest

from rayforge.core.source_asset import SourceAsset
from rayforge.core.vectorization_spec import PassthroughSpec, TraceSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import import_file, importer_registry
from rayforge.image.pdf.renderer import PDF_RENDERER
from rayforge.image.postscript import ghostscript
from rayforge.image.postscript.importer import AiImporter, EpsImporter

HAS_GS = ghostscript.find_ghostscript() is not None
needs_gs = pytest.mark.skipif(not HAS_GS, reason="Ghostscript not installed")

PS_DRAWING = b"""newpath 10 10 moveto 134 10 lineto 134 62 lineto 10 62 lineto
closepath 1 setlinewidth stroke
showpage
%%EOF
"""

EPS_DATA = (
    b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 144 72\n"
    b"%%EndComments\n" + PS_DRAWING
)

LEGACY_AI_DATA = (
    b"%!PS-Adobe-3.0 EPSF-3.0\n%%Creator: Adobe Illustrator(R) 8.0\n"
    b"%%BoundingBox: 0 0 144 72\n%%EndComments\n" + PS_DRAWING
)


def _pdf_compatible_ai() -> bytes:
    """A PDF stream as written by Illustrator with PDF compatibility."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 144, 72)
    cr = cairo.Context(surface)
    cr.rectangle(10, 10, 124, 52)
    cr.set_line_width(1)
    cr.stroke()
    surface.finish()
    return buf.getvalue()


@pytest.fixture
def no_ghostscript(monkeypatch):
    monkeypatch.setattr(ghostscript, "find_ghostscript", lambda: None)
    ghostscript.clear_cache()
    yield
    ghostscript.clear_cache()


def _source(result) -> SourceAsset:
    assert result is not None
    assert result.payload is not None, result.errors
    return result.payload.source


def _items(result) -> list[WorkPiece]:
    assert result is not None
    assert result.payload is not None, result.errors
    return cast(list[WorkPiece], result.payload.items)


class TestRegistration:
    @pytest.mark.parametrize(
        "extension, importer_cls",
        [
            (".ai", AiImporter),
            (".eps", EpsImporter),
            (".epsf", EpsImporter),
            (".ps", EpsImporter),
        ],
    )
    def test_extension_lookup(self, extension, importer_cls):
        assert importer_registry.get_by_extension(extension) is importer_cls

    @pytest.mark.parametrize(
        "mime_type, importer_cls",
        [
            ("application/illustrator", AiImporter),
            ("image/x-eps", EpsImporter),
            ("application/postscript", EpsImporter),
        ],
    )
    def test_mime_lookup(self, mime_type, importer_cls):
        assert importer_registry.get_by_mime_type(mime_type) is importer_cls


class TestIllustrator:
    def test_pdf_compatible_file_needs_no_ghostscript(self, no_ghostscript):
        data = _pdf_compatible_ai()
        importer = AiImporter(data, Path("logo.ai"))
        assert importer.scan().errors == []

        result = AiImporter(data, Path("logo.ai")).get_doc_items(
            PassthroughSpec()
        )
        items = _items(result)
        assert items
        assert items[0].boundaries is not None
        assert not items[0].boundaries.is_empty()
        source = _source(result)
        assert source.original_data == data
        assert source.renderer is PDF_RENDERER
        assert source.metadata["_importer_class"] == "AiImporter"

    def test_pdf_compatible_file_can_be_traced(self, no_ghostscript):
        result = AiImporter(
            _pdf_compatible_ai(), Path("logo.ai")
        ).get_doc_items(TraceSpec())
        assert _items(result)

    def test_legacy_file_without_ghostscript_explains(self, no_ghostscript):
        importer = AiImporter(LEGACY_AI_DATA, Path("old.ai"))
        errors = importer.scan().errors
        assert len(errors) == 1
        assert "PDF" in errors[0]
        assert "Ghostscript" in errors[0]

        result = AiImporter(LEGACY_AI_DATA, Path("old.ai")).get_doc_items(
            PassthroughSpec()
        )
        assert result is not None
        assert result.payload is None
        assert result.errors

    @needs_gs
    def test_legacy_file_is_converted_with_ghostscript(self):
        result = AiImporter(LEGACY_AI_DATA, Path("old.ai")).get_doc_items(
            PassthroughSpec()
        )
        assert _items(result)
        assert _source(result).original_data.startswith(b"%PDF")

    def test_import_file_picks_illustrator_importer(
        self, tmp_path, no_ghostscript
    ):
        path = tmp_path / "logo.ai"
        path.write_bytes(_pdf_compatible_ai())
        payload = import_file(path, mime_type="application/illustrator")
        assert payload is not None
        assert payload.source.metadata["_importer_class"] == "AiImporter"


class TestEps:
    def test_without_ghostscript_reports_missing_dependency(
        self, no_ghostscript
    ):
        manifest = EpsImporter(EPS_DATA, Path("logo.eps")).scan()
        assert len(manifest.errors) == 1
        assert "Ghostscript" in manifest.errors[0]

        result = EpsImporter(EPS_DATA, Path("logo.eps")).get_doc_items(
            PassthroughSpec()
        )
        assert result is not None
        assert result.payload is None
        assert any("Ghostscript" in e for e in result.errors)

    @needs_gs
    def test_scan_uses_bounding_box(self):
        manifest = EpsImporter(EPS_DATA, Path("logo.eps")).scan()
        assert manifest.errors == []
        assert manifest.natural_size_mm == pytest.approx(
            (144 * 25.4 / 72, 72 * 25.4 / 72), abs=0.5
        )

    @needs_gs
    def test_vectors_are_imported(self):
        result = EpsImporter(EPS_DATA, Path("logo.eps")).get_doc_items(
            PassthroughSpec()
        )
        items = _items(result)
        assert len(items) >= 1
        wp = items[0]
        assert wp.boundaries is not None
        assert not wp.boundaries.is_empty()
        width_mm, height_mm = wp.size
        assert width_mm == pytest.approx(144 * 25.4 / 72, abs=0.5)
        assert height_mm == pytest.approx(72 * 25.4 / 72, abs=0.5)

        source = _source(result)
        assert source.original_data.startswith(b"%PDF")
        assert source.source_file == Path("logo.eps")
        assert source.renderer is PDF_RENDERER
        assert source.metadata["_importer_class"] == "EpsImporter"

    @needs_gs
    def test_source_remembers_the_original_eps_file(self):
        result = EpsImporter(EPS_DATA, Path("logo.eps")).get_doc_items(
            PassthroughSpec()
        )
        source = _source(result)
        assert source.original_data != EPS_DATA
        assert source.matches_source_file(EPS_DATA)
        assert source.source_file_size == len(EPS_DATA)
        assert not source.matches_source_file(source.original_data)

    @needs_gs
    def test_reimport_keeps_the_original_fingerprint(self):
        source = _source(
            EpsImporter(EPS_DATA, Path("logo.eps")).get_doc_items(
                PassthroughSpec()
            )
        )
        importer = EpsImporter(source.original_data, Path("logo.eps"))
        again = importer.get_doc_items_for_reimport(source, PassthroughSpec())
        assert again is not None
        assert source.matches_source_file(EPS_DATA)

    def test_pdf_compatible_ai_fingerprint_is_the_file(self, no_ghostscript):
        data = _pdf_compatible_ai()
        source = _source(
            AiImporter(data, Path("logo.ai")).get_doc_items(PassthroughSpec())
        )
        assert source.matches_source_file(data)

    @needs_gs
    def test_reimport_does_not_need_ghostscript(self, monkeypatch):
        result = EpsImporter(EPS_DATA, Path("logo.eps")).get_doc_items(
            PassthroughSpec()
        )
        source = _source(result)
        monkeypatch.setattr(ghostscript, "find_ghostscript", lambda: None)
        ghostscript.clear_cache()

        importer = EpsImporter(source.original_data, Path("logo.eps"))
        again = importer.get_doc_items_for_reimport(source, PassthroughSpec())
        assert again is not None
        assert again.errors == []

    @needs_gs
    def test_broken_file_reports_conversion_error(self):
        broken = b"%!PS-Adobe-3.0\nthis is not postscript {{{\n"
        manifest = EpsImporter(broken, Path("broken.eps")).scan()
        assert len(manifest.errors) == 1
        assert "broken.eps" in manifest.errors[0]


class TestGhostscript:
    def test_conversion_is_cached(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            out = next(a for a in cmd if a.startswith("-sOutputFile="))
            Path(out.split("=", 1)[1]).write_bytes(b"%PDF-1.4 fake")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(ghostscript, "find_ghostscript", lambda: "gs")
        monkeypatch.setattr(ghostscript.subprocess, "run", fake_run)
        ghostscript.clear_cache()

        first = ghostscript.convert_to_pdf(EPS_DATA)
        second = ghostscript.convert_to_pdf(EPS_DATA)
        assert first == second == b"%PDF-1.4 fake"
        assert len(calls) == 1
        assert "-dSAFER" in calls[0]
        assert "-dNoOutputFonts" in calls[0]
        ghostscript.clear_cache()

    def test_missing_ghostscript_raises(self, no_ghostscript):
        with pytest.raises(ghostscript.GhostscriptNotFound):
            ghostscript.convert_to_pdf(EPS_DATA)

    def test_finds_windows_console_binary(self, monkeypatch):
        found = {"gswin64c": "C:/gs/bin/gswin64c.exe"}
        monkeypatch.setattr(shutil, "which", lambda name: found.get(name))
        assert ghostscript.find_ghostscript() == "C:/gs/bin/gswin64c.exe"
