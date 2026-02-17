import pytest
from pathlib import Path
from rayforge.core.geo import Geometry
from rayforge.image.svg.exporter import GeometrySvgExporter
from rayforge.image.svg.importer import SvgImporter
from rayforge.core.vectorization_spec import PassthroughSpec


@pytest.fixture
def line_geometry() -> Geometry:
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(100, 50)
    return geo


@pytest.fixture
def rectangle_geometry() -> Geometry:
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(100, 0)
    geo.line_to(100, 50)
    geo.line_to(0, 50)
    geo.close_path()
    return geo


@pytest.fixture
def arc_geometry() -> Geometry:
    geo = Geometry()
    geo.move_to(100, 50)
    geo.arc_to(50, 100, i=-50, j=0, clockwise=True)
    return geo


@pytest.fixture
def bezier_geometry() -> Geometry:
    geo = Geometry()
    geo.move_to(0, 0)
    geo.bezier_to(100, 0, 25, 50, 75, 50)
    return geo


class TestGeometrySvgExporterInit:
    def test_init_with_geometry(self, line_geometry: Geometry):
        exporter = GeometrySvgExporter(line_geometry)
        assert exporter.geometry is line_geometry

    def test_class_attributes(self):
        assert GeometrySvgExporter.extensions == (".svg",)
        assert GeometrySvgExporter.mime_types == ("image/svg+xml",)


class TestGeometrySvgExporterExport:
    def test_export_line_geometry(self, line_geometry: Geometry):
        exporter = GeometrySvgExporter(line_geometry)
        svg_bytes = exporter.export()

        assert isinstance(svg_bytes, bytes)
        svg_str = svg_bytes.decode("utf-8")
        assert "<svg" in svg_str
        assert "</svg>" in svg_str
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg_str
        assert "<path" in svg_str

    def test_export_rectangle_geometry(self, rectangle_geometry: Geometry):
        exporter = GeometrySvgExporter(rectangle_geometry)
        svg_bytes = exporter.export()

        svg_str = svg_bytes.decode("utf-8")
        assert "M " in svg_str
        assert "L " in svg_str

    def test_export_arc_geometry(self, arc_geometry: Geometry):
        exporter = GeometrySvgExporter(arc_geometry)
        svg_bytes = exporter.export()

        svg_str = svg_bytes.decode("utf-8")
        assert "A " in svg_str

    def test_export_bezier_geometry(self, bezier_geometry: Geometry):
        exporter = GeometrySvgExporter(bezier_geometry)
        svg_bytes = exporter.export()

        svg_str = svg_bytes.decode("utf-8")
        assert "C " in svg_str

    def test_export_empty_geometry_raises(self):
        geo = Geometry()
        exporter = GeometrySvgExporter(geo)
        with pytest.raises(ValueError, match="geometry is empty"):
            exporter.export()

    def test_export_has_correct_viewbox(self, rectangle_geometry: Geometry):
        exporter = GeometrySvgExporter(rectangle_geometry)
        svg_bytes = exporter.export()

        svg_str = svg_bytes.decode("utf-8")
        assert 'viewBox="' in svg_str
        assert 'mm"' in svg_str

    def test_export_uses_mm_units(self, line_geometry: Geometry):
        exporter = GeometrySvgExporter(line_geometry)
        svg_bytes = exporter.export()

        svg_str = svg_bytes.decode("utf-8")
        assert 'width="102.000mm"' in svg_str
        assert 'height="52.000mm"' in svg_str


class TestGeometrySvgExporterRealWorld:
    def test_export_mouse_sketch(self):
        import json
        from rayforge.core.sketcher import Sketch

        mouse_file = Path(__file__).parent.parent / "sketch/mouse.rfs"
        data = json.loads(mouse_file.read_text())

        sketch = Sketch.from_dict(data)
        sketch.solve()

        geo = sketch.to_geometry()
        min_x, min_y, max_x, max_y = geo.rect()

        exporter = GeometrySvgExporter(geo)
        svg_bytes = exporter.export()
        svg_str = svg_bytes.decode("utf-8")

        assert 'd="' in svg_str
        assert len(svg_str) > 300

        width = max_x - min_x
        height = max_y - min_y
        assert f'width="{width + 2:.3f}mm"' in svg_str
        assert f'height="{height + 2:.3f}mm"' in svg_str

        for i, cmd in enumerate(geo.iter_commands()):
            x = cmd[1]
            tx = x - min_x
            if i == 0:
                assert f"M {tx:.6f}" in svg_str

    def test_export_multi_subpath(self):
        """
        Test that multiple subpaths are exported with separate M commands.
        """
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.move_to(20, 0)
        geo.line_to(30, 0)
        geo.move_to(40, 0)
        geo.line_to(50, 0)

        exporter = GeometrySvgExporter(geo)
        svg_bytes = exporter.export()
        svg_str = svg_bytes.decode("utf-8")

        assert "M " in svg_str
        move_count = svg_str.count("M ")
        assert move_count == 3, f"Expected 3 M commands, got {move_count}"

    def test_export_disconnected_shapes(self):
        """Test that disconnected shapes don't get connected."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 10)
        geo.move_to(100, 100)
        geo.line_to(110, 110)

        exporter = GeometrySvgExporter(geo)
        svg_bytes = exporter.export()
        svg_str = svg_bytes.decode("utf-8")

        assert "M " in svg_str
        move_count = svg_str.count("M ")
        assert move_count == 2, f"Expected 2 M commands, got {move_count}"


class TestGeometrySvgExporterRoundTrip:
    def test_roundtrip_line_geometry(self, line_geometry: Geometry):
        original_rect = line_geometry.rect()
        original_distance = line_geometry.distance()

        exporter = GeometrySvgExporter(line_geometry)
        svg_bytes = exporter.export()

        importer = SvgImporter(svg_bytes, Path("test.svg"))
        spec = PassthroughSpec(create_new_layers=False)
        import_result = importer.get_doc_items(vectorization_spec=spec)

        assert import_result is not None
        assert import_result.payload is not None
        assert len(import_result.payload.items) == 1

        from rayforge.core.workpiece import WorkPiece

        imported_wp = import_result.payload.items[0]
        assert isinstance(imported_wp, WorkPiece)

        imported_geo = imported_wp.world_space_boundaries
        assert imported_geo is not None
        assert not imported_geo.is_empty()

        imported_rect = imported_geo.rect()
        assert imported_rect[2] - imported_rect[0] == pytest.approx(
            original_rect[2] - original_rect[0], rel=0.05
        )
        assert imported_rect[3] - imported_rect[1] == pytest.approx(
            original_rect[3] - original_rect[1], rel=0.05
        )
        assert imported_geo.distance() == pytest.approx(
            original_distance, rel=0.05
        )

    def test_roundtrip_rectangle_geometry(self, rectangle_geometry: Geometry):
        original_rect = rectangle_geometry.rect()
        original_distance = rectangle_geometry.distance()

        exporter = GeometrySvgExporter(rectangle_geometry)
        svg_bytes = exporter.export()

        importer = SvgImporter(svg_bytes, Path("test.svg"))
        spec = PassthroughSpec(create_new_layers=False)
        import_result = importer.get_doc_items(vectorization_spec=spec)

        assert import_result is not None
        assert import_result.payload is not None
        from rayforge.core.workpiece import WorkPiece

        imported_wp = import_result.payload.items[0]
        assert isinstance(imported_wp, WorkPiece)

        imported_geo = imported_wp.world_space_boundaries
        assert imported_geo is not None
        assert not imported_geo.is_empty()

        imported_rect = imported_geo.rect()
        assert imported_rect[2] - imported_rect[0] == pytest.approx(
            original_rect[2] - original_rect[0], rel=0.05
        )
        assert imported_rect[3] - imported_rect[1] == pytest.approx(
            original_rect[3] - original_rect[1], rel=0.05
        )
        assert imported_geo.distance() == pytest.approx(
            original_distance, rel=0.05
        )

    def test_roundtrip_arc_geometry(self, arc_geometry: Geometry):
        original_rect = arc_geometry.rect()
        original_distance = arc_geometry.distance()

        exporter = GeometrySvgExporter(arc_geometry)
        svg_bytes = exporter.export()

        importer = SvgImporter(svg_bytes, Path("test.svg"))
        spec = PassthroughSpec(create_new_layers=False)
        import_result = importer.get_doc_items(vectorization_spec=spec)

        assert import_result is not None
        assert import_result.payload is not None
        from rayforge.core.workpiece import WorkPiece

        imported_wp = import_result.payload.items[0]
        assert isinstance(imported_wp, WorkPiece)

        imported_geo = imported_wp.world_space_boundaries
        assert imported_geo is not None
        assert not imported_geo.is_empty()

        imported_rect = imported_geo.rect()
        assert imported_rect[2] - imported_rect[0] == pytest.approx(
            original_rect[2] - original_rect[0], rel=0.05
        )
        assert imported_rect[3] - imported_rect[1] == pytest.approx(
            original_rect[3] - original_rect[1], rel=0.05
        )
        assert imported_geo.distance() == pytest.approx(
            original_distance, rel=0.05
        )

    def test_roundtrip_bezier_geometry(self, bezier_geometry: Geometry):
        original_rect = bezier_geometry.rect()

        exporter = GeometrySvgExporter(bezier_geometry)
        svg_bytes = exporter.export()

        importer = SvgImporter(svg_bytes, Path("test.svg"))
        spec = PassthroughSpec(create_new_layers=False)
        import_result = importer.get_doc_items(vectorization_spec=spec)

        assert import_result is not None
        assert import_result.payload is not None
        from rayforge.core.workpiece import WorkPiece

        imported_wp = import_result.payload.items[0]
        assert isinstance(imported_wp, WorkPiece)

        imported_geo = imported_wp.world_space_boundaries
        assert imported_geo is not None
        assert not imported_geo.is_empty()

        imported_rect = imported_geo.rect()
        assert imported_rect[2] - imported_rect[0] == pytest.approx(
            original_rect[2] - original_rect[0], rel=0.05
        )
        assert imported_rect[3] - imported_rect[1] == pytest.approx(
            original_rect[3] - original_rect[1], rel=0.05
        )

    def test_roundtrip_quarter_arc_cw(self):
        geo = Geometry()
        geo.move_to(50, 0)
        geo.arc_to(0, 50, i=0, j=50, clockwise=True)
        original_rect = geo.rect()
        original_distance = geo.distance()

        exporter = GeometrySvgExporter(geo)
        svg_bytes = exporter.export()

        importer = SvgImporter(svg_bytes, Path("test.svg"))
        spec = PassthroughSpec(create_new_layers=False)
        import_result = importer.get_doc_items(vectorization_spec=spec)

        assert import_result is not None
        assert import_result.payload is not None
        from rayforge.core.workpiece import WorkPiece

        imported_wp = import_result.payload.items[0]
        assert isinstance(imported_wp, WorkPiece)

        imported_geo = imported_wp.world_space_boundaries
        assert imported_geo is not None
        assert not imported_geo.is_empty()

        imported_rect = imported_geo.rect()
        assert imported_rect[2] - imported_rect[0] == pytest.approx(
            original_rect[2] - original_rect[0], rel=0.05
        )
        assert imported_rect[3] - imported_rect[1] == pytest.approx(
            original_rect[3] - original_rect[1], rel=0.05
        )
        assert imported_geo.distance() == pytest.approx(
            original_distance, rel=0.05
        )

    def test_roundtrip_quarter_arc_ccw(self):
        geo = Geometry()
        geo.move_to(50, 0)
        geo.arc_to(0, 50, i=0, j=50, clockwise=False)
        original_rect = geo.rect()
        original_distance = geo.distance()

        exporter = GeometrySvgExporter(geo)
        svg_bytes = exporter.export()

        importer = SvgImporter(svg_bytes, Path("test.svg"))
        spec = PassthroughSpec(create_new_layers=False)
        import_result = importer.get_doc_items(vectorization_spec=spec)

        assert import_result is not None
        assert import_result.payload is not None
        from rayforge.core.workpiece import WorkPiece

        imported_wp = import_result.payload.items[0]
        assert isinstance(imported_wp, WorkPiece)

        imported_geo = imported_wp.world_space_boundaries
        assert imported_geo is not None
        assert not imported_geo.is_empty()

        imported_rect = imported_geo.rect()
        assert imported_rect[2] - imported_rect[0] == pytest.approx(
            original_rect[2] - original_rect[0], rel=0.05
        )
        assert imported_rect[3] - imported_rect[1] == pytest.approx(
            original_rect[3] - original_rect[1], rel=0.05
        )
        assert imported_geo.distance() == pytest.approx(
            original_distance, rel=0.05
        )

    def test_roundtrip_multi_subpath(self):
        """Test round-trip of geometry with multiple subpaths."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(50, 0)
        geo.line_to(50, 50)
        geo.move_to(100, 0)
        geo.line_to(150, 0)
        geo.line_to(150, 50)

        original_distance = geo.distance()

        exporter = GeometrySvgExporter(geo)
        svg_bytes = exporter.export()

        importer = SvgImporter(svg_bytes, Path("test.svg"))
        spec = PassthroughSpec(create_new_layers=False)
        import_result = importer.get_doc_items(vectorization_spec=spec)

        assert import_result is not None
        assert import_result.payload is not None
        from rayforge.core.workpiece import WorkPiece

        imported_wp = import_result.payload.items[0]
        assert isinstance(imported_wp, WorkPiece)

        imported_geo = imported_wp.world_space_boundaries
        assert imported_geo is not None
        assert not imported_geo.is_empty()

        assert imported_geo.distance() == pytest.approx(
            original_distance, rel=0.05
        )


class TestMultiGeometrySvgExporter:
    """Tests for exporting multiple geometries to a single SVG."""

    def test_export_multiple_geometries(self):
        """Test that multiple geometries create multiple path elements."""
        geo1 = Geometry()
        geo1.move_to(0, 0)
        geo1.line_to(10, 10)

        geo2 = Geometry()
        geo2.move_to(100, 100)
        geo2.line_to(110, 110)

        from rayforge.image.svg.exporter import MultiGeometrySvgExporter

        exporter = MultiGeometrySvgExporter([geo1, geo2])
        svg_bytes = exporter.export()
        svg_str = svg_bytes.decode("utf-8")

        assert svg_str.count("<path") == 2

    def test_export_calculates_combined_bounds(self):
        """Test that the viewBox encompasses all geometries."""
        geo1 = Geometry()
        geo1.move_to(0, 0)
        geo1.line_to(10, 10)

        geo2 = Geometry()
        geo2.move_to(100, 50)
        geo2.line_to(150, 100)

        from rayforge.image.svg.exporter import MultiGeometrySvgExporter

        exporter = MultiGeometrySvgExporter([geo1, geo2])
        svg_bytes = exporter.export()
        svg_str = svg_bytes.decode("utf-8")

        assert 'width="152.000mm"' in svg_str
        assert 'height="102.000mm"' in svg_str

    def test_export_empty_list_raises(self):
        """Test that exporting empty geometry list raises."""
        from rayforge.image.svg.exporter import MultiGeometrySvgExporter

        exporter = MultiGeometrySvgExporter([])
        with pytest.raises(ValueError, match="empty"):
            exporter.export()
