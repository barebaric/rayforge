from pathlib import Path
from unittest.mock import Mock

import pytest
from raygeo.geo import Matrix

from rayforge.core.vectorization_spec import LayerImportMode, PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import import_file, importer_registry
from rayforge.image.base_importer import ImporterFeature
from rayforge.image.hpgl.importer import HpglImporter
from rayforge.image.hpgl.parser import parse_hpgl
from rayforge.image.hpgl.renderer import HPGL_RENDERER

SQUARE = b"IN;SP1;PU0,0;PD400,0,400,400,0,400,0,0;PU;"


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


def _rect(geo):
    return tuple(round(v, 4) for v in geo.rect())


def _points(geo):
    return [
        (round(c.end[0], 4), round(c.end[1], 4))
        for c in geo.iter_typed_commands()
    ]


class TestParser:
    def test_square_in_millimetres(self):
        drawing = parse_hpgl(SQUARE)
        assert list(drawing.geometries_by_pen) == [1]
        geo = drawing.geometries_by_pen[1]
        assert _rect(geo) == (0, 0, 10, 10)
        assert geo.is_closed()

    def test_relative_and_absolute_moves(self):
        drawing = parse_hpgl(b"IN;SP1;PA40,40;PD;PR400,0,0,400;PA40,40;")
        assert _points(drawing.geometries_by_pen[1]) == [
            (1, 1),
            (11, 1),
            (11, 11),
            (1, 1),
        ]

    def test_pen_up_moves_do_not_draw(self):
        drawing = parse_hpgl(b"IN;SP1;PU0,0,400,0;PD400,400;")
        assert _points(drawing.geometries_by_pen[1]) == [(10, 0), (10, 10)]

    def test_pens_become_separate_geometries(self):
        data = b"IN;SP1;PU0,0;PD400,0;SP2;PU0,400;PD400,400;SP1;PD800,400;"
        drawing = parse_hpgl(data)
        assert sorted(drawing.geometries_by_pen) == [1, 2]
        assert _rect(drawing.geometries_by_pen[2]) == (0, 10, 10, 10)
        assert _rect(drawing.geometries_by_pen[1]) == (0, 0, 20, 10)

    def test_pen_zero_draws_nothing(self):
        drawing = parse_hpgl(b"IN;SP0;PU0,0;PD400,0;")
        assert drawing.geometries_by_pen == {}

    def test_circle_around_current_position(self):
        drawing = parse_hpgl(b"IN;SP1;PU400,400;CI200;")
        geo = drawing.geometries_by_pen[1]
        assert _rect(geo) == pytest.approx((5, 5, 15, 15), abs=1e-3)
        assert geo.is_closed()

    def test_arc_absolute_counter_clockwise(self):
        drawing = parse_hpgl(b"IN;SP1;PU400,0;PD;AA0,0,90;")
        geo = drawing.geometries_by_pen[1]
        assert _points(geo)[-1] == pytest.approx((0, 10), abs=1e-6)
        assert _rect(geo) == pytest.approx((0, 0, 10, 10), abs=1e-3)

    def test_arc_relative_clockwise(self):
        drawing = parse_hpgl(b"IN;SP1;PU0,400;PD;AR0,-400,-90;")
        geo = drawing.geometries_by_pen[1]
        assert _points(geo)[-1] == pytest.approx((10, 0), abs=1e-6)

    def test_full_circle_arc(self):
        drawing = parse_hpgl(b"IN;SP1;PU400,0;PD;AA0,0,360;")
        geo = drawing.geometries_by_pen[1]
        assert _rect(geo) == pytest.approx((-10, -10, 10, 10), abs=1e-3)

    def test_edge_rectangles(self):
        drawing = parse_hpgl(b"IN;SP1;PU0,0;EA400,200;PU800,0;ER400,200;")
        geo = drawing.geometries_by_pen[1]
        assert len(geo.split_into_contours()) == 2
        assert _rect(geo) == pytest.approx((0, 0, 30, 5))

    def test_free_format_parameters(self):
        drawing = parse_hpgl(b"IN SP1 PU 0 0 PD 400 0\nPD400 -400.5")
        geo = drawing.geometries_by_pen[1]
        assert _points(geo)[-1] == pytest.approx((10, -10.0125))

    def test_pcl_escape_sequences_are_skipped(self):
        drawing = parse_hpgl(b"\x1b%-1BIN;SP1;PU0,0;PD400,0;\x1b%0A")
        assert _rect(drawing.geometries_by_pen[1]) == (0, 0, 10, 0)

    def test_labels_are_skipped_and_reported(self):
        drawing = parse_hpgl(b"IN;SP1;PU0,0;LBHELLO;PD\x03PD400,0;")
        assert _rect(drawing.geometries_by_pen[1]) == (0, 0, 10, 0)
        assert "LB" in drawing.unsupported

    def test_scaling_is_reported(self):
        drawing = parse_hpgl(b"IN;SC0,100,0,100;SP1;PD400,0;")
        assert "SC" in drawing.unsupported

    def test_harmless_instructions_are_not_reported(self):
        drawing = parse_hpgl(b"IN;VS10;FS2;SP1;PD400,0;")
        assert drawing.unsupported == set()

    def test_garbage_has_no_geometry(self):
        assert parse_hpgl(b"\x00\x01binary junk").geometries_by_pen == {}


class TestImporter:
    def test_registration(self):
        for ext in (".plt", ".hpgl", ".hpg", ".hgl"):
            assert importer_registry.get_by_extension(ext) is HpglImporter
        for mime in ("application/vnd.hp-hpgl", "application/vnd.hp-HPGL"):
            assert importer_registry.get_by_mime_type(mime) is HpglImporter
        assert ImporterFeature.DIRECT_VECTOR in HpglImporter.features
        assert ImporterFeature.LAYER_SELECTION in HpglImporter.features

    def test_scan_lists_pens_as_layers(self):
        data = b"IN;SP1;PU0,0;PD400,0;SP3;PU0,0;PD0,800;"
        manifest = HpglImporter(data, Path("cut.plt")).scan()
        assert manifest.errors == []
        assert manifest.natural_size_mm == pytest.approx((10, 20))
        assert [(layer.id, layer.name) for layer in manifest.layers] == [
            ("pen-1", "Pen 1"),
            ("pen-3", "Pen 3"),
        ]

    def test_import_creates_workpiece(self):
        result = HpglImporter(SQUARE, Path("square.plt")).get_doc_items(
            PassthroughSpec()
        )
        items = _workpieces(result)
        assert len(items) == 1
        assert _world_size(items[0]) == pytest.approx((10, 10))
        assert result is not None and result.payload is not None
        source = result.payload.source
        assert source.renderer is HPGL_RENDERER
        assert source.original_data == SQUARE
        assert source.metadata["_importer_class"] == "HpglImporter"

    def test_import_split_by_pen(self):
        data = b"IN;SP1;PU0,0;PD400,0,400,400;SP2;PU800,0;PD1200,0,1200,400;"
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.NEW_LAYERS)
        result = HpglImporter(data, Path("two.plt")).get_doc_items(spec)
        assert len(_workpieces(result)) == 2

    def test_import_only_selected_pen(self):
        data = b"IN;SP1;PU0,0;PD400,0,400,400;SP2;PU800,0;PD1200,0,1200,400;"
        spec = PassthroughSpec(active_layer_ids=["pen-2"])
        result = HpglImporter(data, Path("two.plt")).get_doc_items(spec)
        workpieces = _workpieces(result)
        assert len(workpieces) == 1
        assert _world_size(workpieces[0]) == pytest.approx((10, 10))

    def test_unsupported_instructions_become_a_warning(self):
        data = b"IN;SP1;PU0,0;PD400,0,400,400;LBX\x03;"
        manifest = HpglImporter(data, Path("text.plt")).scan()
        assert len(manifest.warnings) == 1
        assert "LB" in manifest.warnings[0]

    def test_file_without_drawing_reports_error(self):
        importer = HpglImporter(b"not a plot file", Path("bad.plt"))
        assert importer.scan().errors
        result = HpglImporter(b"not a plot file", Path("bad.plt"))
        doc = result.get_doc_items(PassthroughSpec())
        assert doc is not None
        assert doc.payload is None or not doc.payload.items

    def test_import_file_by_extension(self, tmp_path):
        path = tmp_path / "square.plt"
        path.write_bytes(SQUARE)
        payload = import_file(path)
        assert payload is not None
        assert payload.source.metadata["_importer_class"] == "HpglImporter"
