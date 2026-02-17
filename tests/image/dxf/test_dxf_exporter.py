import pytest
import io
import ezdxf
from pathlib import Path
from rayforge.core.geo import Geometry
from rayforge.image.dxf.exporter import (
    GeometryDxfExporter,
    MultiGeometryDxfExporter,
)
from rayforge.image.dxf.importer import DxfImporter
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


class TestGeometryDxfExporterInit:
    def test_init_with_geometry(self, line_geometry: Geometry):
        exporter = GeometryDxfExporter(line_geometry)
        assert exporter.geometry is line_geometry

    def test_class_attributes(self):
        assert GeometryDxfExporter.extensions == (".dxf",)
        assert GeometryDxfExporter.mime_types == ("image/vnd.dxf",)


class TestGeometryDxfExporterExport:
    def test_export_line_geometry(self, line_geometry: Geometry):
        exporter = GeometryDxfExporter(line_geometry)
        dxf_bytes = exporter.export()

        assert isinstance(dxf_bytes, bytes)
        dxf_str = dxf_bytes.decode("utf-8")
        assert "SECTION" in dxf_str
        assert "ENTITIES" in dxf_str

    def test_export_rectangle_geometry(self, rectangle_geometry: Geometry):
        exporter = GeometryDxfExporter(rectangle_geometry)
        dxf_bytes = exporter.export()

        buffer = io.StringIO(dxf_bytes.decode("utf-8"))
        dxf_doc = ezdxf.read(buffer)  # type: ignore
        msp = dxf_doc.modelspace()
        entities = list(msp)
        assert len(entities) >= 1
        assert any(e.dxftype() == "LWPOLYLINE" for e in entities)

    def test_export_arc_geometry(self, arc_geometry: Geometry):
        exporter = GeometryDxfExporter(arc_geometry)
        dxf_bytes = exporter.export()

        buffer = io.StringIO(dxf_bytes.decode("utf-8"))
        dxf_doc = ezdxf.read(buffer)  # type: ignore
        msp = dxf_doc.modelspace()
        entities = list(msp)
        assert any(e.dxftype() == "ARC" for e in entities)

    def test_export_bezier_geometry(self, bezier_geometry: Geometry):
        exporter = GeometryDxfExporter(bezier_geometry)
        dxf_bytes = exporter.export()

        buffer = io.StringIO(dxf_bytes.decode("utf-8"))
        dxf_doc = ezdxf.read(buffer)  # type: ignore
        msp = dxf_doc.modelspace()
        entities = list(msp)
        assert any(e.dxftype() == "SPLINE" for e in entities)

    def test_export_empty_geometry_raises(self):
        geo = Geometry()
        exporter = GeometryDxfExporter(geo)
        with pytest.raises(ValueError, match="geometry is empty"):
            exporter.export()

    def test_export_produces_valid_dxf(self, rectangle_geometry: Geometry):
        exporter = GeometryDxfExporter(rectangle_geometry)
        dxf_bytes = exporter.export()

        buffer = io.StringIO(dxf_bytes.decode("utf-8"))
        dxf_doc = ezdxf.read(buffer)  # type: ignore
        assert dxf_doc is not None
        assert dxf_doc.modelspace() is not None

    def test_export_multi_subpath(self):
        """Test that multiple subpaths create separate polylines."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.move_to(20, 0)
        geo.line_to(30, 0)
        geo.move_to(40, 0)
        geo.line_to(50, 0)

        exporter = GeometryDxfExporter(geo)
        dxf_bytes = exporter.export()

        buffer = io.StringIO(dxf_bytes.decode("utf-8"))
        dxf_doc = ezdxf.read(buffer)  # type: ignore
        msp = dxf_doc.modelspace()
        entities = list(msp)

        polylines = [e for e in entities if e.dxftype() == "LWPOLYLINE"]
        assert len(polylines) == 3, (
            f"Expected 3 polylines, got {len(polylines)}"
        )

    def test_export_disconnected_shapes(self):
        """Test that disconnected shapes don't get connected."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 10)
        geo.move_to(100, 100)
        geo.line_to(110, 110)

        exporter = GeometryDxfExporter(geo)
        dxf_bytes = exporter.export()

        buffer = io.StringIO(dxf_bytes.decode("utf-8"))
        dxf_doc = ezdxf.read(buffer)  # type: ignore
        msp = dxf_doc.modelspace()
        entities = list(msp)

        polylines = [e for e in entities if e.dxftype() == "LWPOLYLINE"]
        assert len(polylines) == 2, (
            f"Expected 2 polylines, got {len(polylines)}"
        )


class TestGeometryDxfExporterRoundTrip:
    def test_roundtrip_line_geometry(self, line_geometry: Geometry):
        original_rect = line_geometry.rect()
        original_distance = line_geometry.distance()

        exporter = GeometryDxfExporter(line_geometry)
        dxf_bytes = exporter.export()

        importer = DxfImporter(dxf_bytes, Path("test.dxf"))
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

        exporter = GeometryDxfExporter(rectangle_geometry)
        dxf_bytes = exporter.export()

        importer = DxfImporter(dxf_bytes, Path("test.dxf"))
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

        exporter = GeometryDxfExporter(arc_geometry)
        dxf_bytes = exporter.export()

        importer = DxfImporter(dxf_bytes, Path("test.dxf"))
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

    def test_roundtrip_quarter_arc_cw(self):
        geo = Geometry()
        geo.move_to(50, 0)
        geo.arc_to(0, 50, i=0, j=50, clockwise=True)
        original_rect = geo.rect()
        original_distance = geo.distance()

        exporter = GeometryDxfExporter(geo)
        dxf_bytes = exporter.export()

        importer = DxfImporter(dxf_bytes, Path("test.dxf"))
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

        exporter = GeometryDxfExporter(geo)
        dxf_bytes = exporter.export()

        importer = DxfImporter(dxf_bytes, Path("test.dxf"))
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

        exporter = GeometryDxfExporter(geo)
        dxf_bytes = exporter.export()

        importer = DxfImporter(dxf_bytes, Path("test.dxf"))
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


class TestMultiGeometryDxfExporter:
    """Tests for exporting multiple geometries to a single DXF."""

    def test_export_multiple_geometries(self):
        """Test that multiple geometries create separate entities."""
        geo1 = Geometry()
        geo1.move_to(0, 0)
        geo1.line_to(10, 10)

        geo2 = Geometry()
        geo2.move_to(100, 100)
        geo2.line_to(110, 110)

        exporter = MultiGeometryDxfExporter([geo1, geo2])
        dxf_bytes = exporter.export()

        buffer = io.StringIO(dxf_bytes.decode("utf-8"))
        dxf_doc = ezdxf.read(buffer)  # type: ignore
        msp = dxf_doc.modelspace()
        entities = list(msp)

        polylines = [e for e in entities if e.dxftype() == "LWPOLYLINE"]
        assert len(polylines) == 2

    def test_export_combines_into_one_modelspace(self):
        """Test that all geometries end up in same modelspace."""
        geo1 = Geometry()
        geo1.move_to(0, 0)
        geo1.line_to(10, 0)

        geo2 = Geometry()
        geo2.move_to(0, 10)
        geo2.line_to(10, 10)

        from rayforge.image.dxf.exporter import MultiGeometryDxfExporter

        exporter = MultiGeometryDxfExporter([geo1, geo2])
        dxf_bytes = exporter.export()

        buffer = io.StringIO(dxf_bytes.decode("utf-8"))
        dxf_doc = ezdxf.read(buffer)  # type: ignore
        msp = dxf_doc.modelspace()
        entities = list(msp)

        assert len(entities) >= 2

    def test_export_empty_list_raises(self):
        """Test that exporting empty geometry list raises."""
        from rayforge.image.dxf.exporter import MultiGeometryDxfExporter

        exporter = MultiGeometryDxfExporter([])
        with pytest.raises(ValueError, match="empty"):
            exporter.export()
