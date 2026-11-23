from pathlib import Path
from unittest.mock import Mock

import pytest
from raygeo.geo import Matrix

from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import import_file, importer_registry
from rayforge.image.base_importer import ImporterFeature
from rayforge.image.gcode.importer import GcodeImporter
from rayforge.image.gcode.renderer import GCODE_RENDERER

SQUARE_INCH = b"G20 G90\nG0 X0 Y0\nG1 X1.25 Y0\nY1.25\nX0\nY0\nM2\n"

SQUARE_MM = b"G21 G90\nG0 X0 Y0\nG1 X10 Y0\nY10\nX0\nY0\n"


def _workpieces(result) -> list[WorkPiece]:
    assert result is not None and result.payload is not None
    found: list[WorkPiece] = []
    for item in result.payload.items:
        if isinstance(item, WorkPiece):
            found.append(item)
        else:
            found.extend(item.get_descendants(WorkPiece))
    return found


def _world_size(wp: WorkPiece) -> tuple[float, float]:
    parent = Mock()
    parent.get_world_transform.return_value = Matrix.identity()
    wp.parent = parent
    return wp.size


class TestImporter:
    def test_registration(self):
        for ext in (".gcode", ".ngc", ".nc", ".tap", ".gc"):
            assert importer_registry.get_by_extension(ext) is GcodeImporter
        for mime in ("text/x.gcode", "application/x-gcode"):
            assert importer_registry.get_by_mime_type(mime) is GcodeImporter
        assert ImporterFeature.DIRECT_VECTOR in GcodeImporter.features

    def test_scan_reports_size_and_layer(self):
        manifest = GcodeImporter(SQUARE_INCH, Path("square.gcode")).scan()

        assert manifest.errors == []
        assert manifest.warnings == []
        assert manifest.natural_size_mm == pytest.approx((31.75, 31.75))
        assert [(layer.id, layer.name) for layer in manifest.layers] == [
            ("__default__", "square")
        ]
        assert manifest.layers[0].feature_count == 1

    def test_import_creates_workpiece(self):
        result = GcodeImporter(
            SQUARE_INCH, Path("square.gcode")
        ).get_doc_items(PassthroughSpec())
        items = _workpieces(result)

        assert len(items) == 1
        assert _world_size(items[0]) == pytest.approx((31.75, 31.75))
        assert result is not None and result.payload is not None
        source = result.payload.source
        assert source.renderer is GCODE_RENDERER
        assert source.original_data == SQUARE_INCH
        assert source.metadata["_importer_class"] == "GcodeImporter"

    def test_fill_travel_moves_are_not_imported(self):
        data = (
            b"G21 G90\n"
            b"G0 X0 Y0\n"
            b"G91\n"
            b"G1 X10 Y0 S200\n"
            b"G1 Y1 S0\n"
            b"G1 X-10 Y0 S200\n"
            b"G90\n"
            b"M2\n"
        )
        manifest = GcodeImporter(data, Path("fill.gcode")).scan()

        assert manifest.natural_size_mm == pytest.approx((10, 1))
        assert manifest.layers[0].feature_count == 2

    def test_unsupported_instructions_become_a_warning(self):
        data = b"G21 G90\nG0 X0 Y0\nG1 X10 Y0\nG81 X5 Y5\nG41\n"
        manifest = GcodeImporter(data, Path("mill.gcode")).scan()

        assert len(manifest.warnings) == 1
        assert "G41, G81" in manifest.warnings[0]

    def test_file_without_cut_moves_reports_error(self):
        importer = GcodeImporter(b"G21 G90\nG0 X0 Y0\n", Path("empty.gcode"))
        assert importer.scan().errors

        result = GcodeImporter(
            b"G21 G90\nG0 X0 Y0\n", Path("empty.gcode")
        ).get_doc_items(PassthroughSpec())
        assert result is not None
        assert result.payload is None or not result.payload.items

    def test_import_file_by_extension(self, tmp_path):
        path = tmp_path / "square.gcode"
        path.write_bytes(SQUARE_MM)
        payload = import_file(path)

        assert payload is not None
        assert payload.source.metadata["_importer_class"] == "GcodeImporter"
        assert payload.source.width_mm == pytest.approx(10)
        assert payload.source.height_mm == pytest.approx(10)
