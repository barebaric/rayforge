from pathlib import Path

import pytest
from raygeo.formats.coreldraw import write as write_cdr
from raygeo.geo import Geometry

from rayforge.core.vectorization_spec import (
    LayerImportMode,
    PassthroughSpec,
)
from rayforge.image import ImporterFeature
from rayforge.image.coreldraw.importer import CorelDrawImporter

MM_PER_INCH = 25.4


def rectangle(width: float = 20.0, height: float = 10.0) -> Geometry:
    geometry = Geometry()
    geometry.move_to(0, 0, 0)
    geometry.line_to(width, 0, 0)
    geometry.line_to(width, height, 0)
    geometry.line_to(0, height, 0)
    geometry.line_to(0, 0, 0)
    return geometry


@pytest.fixture
def cdr_data() -> bytes:
    return write_cdr(rectangle())


@pytest.fixture
def importer(cdr_data: bytes) -> CorelDrawImporter:
    return CorelDrawImporter(cdr_data, Path("drawing.cdr"))


class TestCorelDrawImporterContract:
    def test_class_attributes(self):
        assert CorelDrawImporter.label == "CorelDRAW files"
        assert CorelDrawImporter.extensions == (".cdr",)
        assert "application/vnd.corel-draw" in CorelDrawImporter.mime_types
        assert (
            CorelDrawImporter.native_mime_type == "application/vnd.corel-draw"
        )
        assert CorelDrawImporter.features == {
            ImporterFeature.DIRECT_VECTOR,
            ImporterFeature.LAYER_SELECTION,
        }

    def test_scan_returns_manifest(self, importer: CorelDrawImporter):
        manifest = importer.scan()

        assert manifest.title == "drawing.cdr"
        assert manifest.natural_size_mm == pytest.approx((20.0, 10.0))
        assert [layer.name for layer in manifest.layers] == ["Layer 1"]
        assert manifest.layers[0].feature_count == 1
        assert manifest.warnings == []
        assert manifest.errors == []

    def test_scan_handles_invalid_data(self):
        importer = CorelDrawImporter(b"not a cdr", Path("broken.cdr"))
        manifest = importer.scan()

        assert manifest.title == "broken.cdr"
        assert manifest.errors

    def test_parse_returns_parsing_result(self, importer: CorelDrawImporter):
        result = importer.parse()

        assert result is not None
        width = 20.0 / MM_PER_INCH
        height = 10.0 / MM_PER_INCH
        assert result.document_bounds == pytest.approx((0, 0, width, height))
        assert result.native_unit_to_mm == MM_PER_INCH
        assert result.is_y_down is False
        assert result.world_frame_of_reference == pytest.approx(
            (0, 0, 20.0, 10.0)
        )
        assert result.background_world_transform is not None
        assert len(result.layers) == 1
        assert result.layers[0].name == "Layer 1"
        assert result.layers[0].content_bounds == pytest.approx(
            (0, 0, width, height)
        )

    def test_parse_handles_invalid_data(self):
        importer = CorelDrawImporter(b"not a cdr", Path("broken.cdr"))

        assert importer.parse() is None
        assert importer._errors

    def test_vectorize_flattens_layers(self, importer: CorelDrawImporter):
        parse_result = importer.parse()
        assert parse_result is not None
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)

        vectorized = importer.vectorize(parse_result, spec)

        assert None in vectorized.geometries_by_layer
        geometry = vectorized.geometries_by_layer[None]
        assert geometry.rect() == pytest.approx(
            (0, 0, 20.0 / MM_PER_INCH, 10.0 / MM_PER_INCH)
        )

    def test_vectorize_keeps_layers(self, importer: CorelDrawImporter):
        parse_result = importer.parse()
        assert parse_result is not None
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.NEW_WORKFLOWS)

        vectorized = importer.vectorize(parse_result, spec)

        assert list(vectorized.geometries_by_layer) == ["Layer 1"]

    def test_create_source_asset(self, importer: CorelDrawImporter):
        parse_result = importer.parse()
        assert parse_result is not None

        source = importer.create_source_asset(parse_result)

        assert source.width_mm == pytest.approx(20.0)
        assert source.height_mm == pytest.approx(10.0)
        assert source.metadata["is_vector"] is True


class TestCorelDrawImporter:
    def test_import_creates_workpiece(self, importer: CorelDrawImporter):
        result = importer.get_doc_items(vectorization_spec=None)

        assert result is not None
        assert result.payload is not None
        assert result.payload.source is not None
        assert len(result.payload.items) == 1
        assert not result.errors

    def test_import_reports_unreadable_file(self):
        importer = CorelDrawImporter(b"not a cdr", Path("broken.cdr"))

        result = importer.get_doc_items(vectorization_spec=None)

        assert result is not None
        assert result.errors
