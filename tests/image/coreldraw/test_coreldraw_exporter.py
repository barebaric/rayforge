import pytest
from raygeo.formats.coreldraw import parse
from raygeo.geo import Geometry

from rayforge.image.coreldraw.exporter import (
    CorelDrawExporter,
    MultiGeometryCorelDrawExporter,
)
from rayforge.image.coreldraw.importer import CorelDrawImporter

MM_PER_INCH = 25.4


def rectangle(x: float, width: float = 10.0) -> Geometry:
    geometry = Geometry()
    geometry.move_to(x, 0, 0)
    geometry.line_to(x + width, 0, 0)
    geometry.line_to(x + width, 10, 0)
    geometry.line_to(x, 10, 0)
    geometry.line_to(x, 0, 0)
    return geometry


class TestCorelDrawExporter:
    def test_class_attributes(self):
        assert CorelDrawExporter.extensions == (".cdr",)
        assert "application/vnd.corel-draw" in CorelDrawExporter.mime_types

    def test_export_writes_a_drawing(self):
        exporter = CorelDrawExporter(rectangle(0))

        data = exporter.export()

        assert data.startswith(b"RIFF")
        assert data[8:12] == b"CDRC"

    def test_export_round_trips_through_the_importer(self):
        exporter = CorelDrawExporter(rectangle(0))

        importer = CorelDrawImporter(exporter.export(), source_file=None)
        result = importer.parse()

        assert result is not None
        assert result.world_frame_of_reference == pytest.approx(
            (0, 0, 10.0, 10.0)
        )

    def test_export_empty_geometry_raises(self):
        exporter = CorelDrawExporter(Geometry())

        with pytest.raises(ValueError):
            exporter.export()


class TestMultiGeometryCorelDrawExporter:
    def test_export_merges_geometries(self):
        exporter = MultiGeometryCorelDrawExporter(
            [rectangle(0), rectangle(20)]
        )

        document = parse(exporter.export())

        assert document.page_width * MM_PER_INCH == pytest.approx(30.0)
        assert document.page_height * MM_PER_INCH == pytest.approx(10.0)
        assert len(document.layers) == 1

    def test_export_empty_geometries_raises(self):
        exporter = MultiGeometryCorelDrawExporter([Geometry()])

        with pytest.raises(ValueError):
            exporter.export()
