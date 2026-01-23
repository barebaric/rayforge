import pytest
import io
from pathlib import Path
import ezdxf
from typing import Optional, Union, cast
from unittest.mock import Mock
from rayforge.core.geo import CMD_TYPE_BEZIER, Geometry
from rayforge.core.layer import Layer
from rayforge.core.matrix import Matrix
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image.dxf.importer import DxfImporter
from rayforge.image.structures import ImportPayload
from rayforge.image.base_importer import ImporterFeature


# Fixtures
@pytest.fixture
def empty_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def line_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    doc.header["$INSUNITS"] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 50))
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def circle_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    doc.header["$INSUNITS"] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_circle(center=(50, 50), radius=25)
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def inches_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    doc.header["$INSUNITS"] = 1  # Inches
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 1))  # 1 inch line
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def multi_layer_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    doc.header["$INSUNITS"] = 4  # Millimeters
    msp = doc.modelspace()

    # Layer 1: Line
    doc.layers.new("Layer1")
    msp.add_line((0, 0), (100, 50), dxfattribs={"layer": "Layer1"})

    # Layer 2: Circle
    doc.layers.new("Layer2")
    msp.add_circle(center=(150, 25), radius=25, dxfattribs={"layer": "Layer2"})

    # Layer 3: Rectangle
    doc.layers.new("Layer3")
    msp.add_lwpolyline(
        [(200, 0), (250, 0), (250, 50), (200, 50)],
        dxfattribs={"layer": "Layer3"},
    )

    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


def _setup_workpiece_with_context(
    importer: DxfImporter,
) -> Optional[WorkPiece]:
    """
    Helper to run importer, correctly link workpiece to its source,
    and mock the document context for rendering tests.
    """
    # Force a merge for simplicity in single-entity tests
    spec = PassthroughSpec(create_new_layers=False)
    import_result = importer.get_doc_items(vectorization_spec=spec)
    if (
        not import_result
        or not import_result.payload
        or not import_result.payload.items
    ):
        return None

    payload = import_result.payload
    # Handle both bare WorkPiece and Layer-wrapped WorkPiece for flexibility
    item = payload.items[0]
    wp: Optional[WorkPiece] = None
    if isinstance(item, WorkPiece):
        wp = item
    elif isinstance(item, Layer) and item.workpieces:
        wp = item.workpieces[0]

    if not wp:
        return None

    source = payload.source

    mock_doc = Mock()
    mock_doc.source_assets = {source.uid: source}
    mock_doc.get_source_asset_by_uid.side_effect = mock_doc.source_assets.get

    mock_parent = Mock()
    mock_parent.doc = mock_doc
    mock_parent.get_world_transform.return_value = Matrix.identity()
    wp.parent = mock_parent

    return wp


@pytest.fixture
def line_workpiece(line_dxf_importer) -> Optional[WorkPiece]:
    return _setup_workpiece_with_context(line_dxf_importer)


@pytest.fixture
def circle_workpiece(circle_dxf_importer) -> Optional[WorkPiece]:
    return _setup_workpiece_with_context(circle_dxf_importer)


@pytest.fixture
def inches_workpiece(inches_dxf_importer) -> Optional[WorkPiece]:
    return _setup_workpiece_with_context(inches_dxf_importer)


@pytest.fixture
def circle_dxf_file():
    return Path(__file__).parent / "circle.dxf"


@pytest.fixture
def acdbcircle_dxf_file():
    return Path(__file__).parent / "acdbcircle.dxf"


@pytest.fixture
def circle_dxf_importer_from_file(circle_dxf_file):
    data = circle_dxf_file.read_bytes()
    return DxfImporter(data)


@pytest.fixture
def acdbcircle_dxf_importer_from_file(acdbcircle_dxf_file):
    data = acdbcircle_dxf_file.read_bytes()
    return DxfImporter(data)


class TestDXFImporterContract:
    """Tests for the Importer contract compliance of DxfImporter."""

    def test_class_attributes(self):
        """Tests that importer class has required attributes."""
        assert DxfImporter.label == "DXF files (2D)"
        assert DxfImporter.mime_types == ("image/vnd.dxf",)
        assert DxfImporter.extensions == (".dxf",)
        assert DxfImporter.features == {
            ImporterFeature.DIRECT_VECTOR,
            ImporterFeature.LAYER_SELECTION,
        }

    def test_scan_returns_manifest(self, line_dxf_importer):
        """Tests that scan() returns ImportManifest with correct data."""
        manifest = line_dxf_importer.scan()

        assert manifest.title == "Untitled"
        assert manifest.natural_size_mm is not None
        width_mm, height_mm = manifest.natural_size_mm
        assert width_mm > 0
        assert height_mm > 0
        assert len(manifest.layers) > 0
        assert len(manifest.warnings) == 0
        assert len(manifest.errors) == 0

    def test_scan_handles_invalid_data(self):
        """Tests that scan() handles invalid DXF data gracefully."""
        importer = DxfImporter(b"not a dxf")
        manifest = importer.scan()

        assert manifest.title == "Untitled"
        assert len(manifest.errors) > 0

    def test_parse_returns_parsing_result(self, line_dxf_importer):
        """Tests that parse() returns ParsingResult with correct data."""
        parse_result = line_dxf_importer.parse()

        assert parse_result is not None
        x, y, width, height = parse_result.document_bounds
        assert x == 0.0
        assert y == 0.0
        assert width == 100.0
        assert height == 50.0
        assert parse_result.is_y_down is False
        assert parse_result.native_unit_to_mm == 1.0
        assert len(parse_result.layers) > 0
        assert parse_result.world_frame_of_reference is not None
        assert parse_result.background_world_transform is not None

    def test_parse_handles_invalid_data(self):
        """Tests that parse() returns None for invalid DXF data."""
        importer = DxfImporter(b"not a dxf")
        parse_result = importer.parse()

        assert parse_result is None
        assert len(importer._errors) > 0

    def test_vectorize_returns_vectorization_result(
        self, line_dxf_importer
    ):
        """Tests that vectorize() returns VectorizationResult."""
        parse_result = line_dxf_importer.parse()
        assert parse_result is not None

        spec = PassthroughSpec(create_new_layers=False)
        vec_result = line_dxf_importer.vectorize(parse_result, spec)

        assert vec_result is not None
        assert vec_result.source_parse_result is parse_result
        assert None in vec_result.geometries_by_layer
        assert isinstance(vec_result.geometries_by_layer[None], Geometry)

    def test_vectorize_with_layer_selection(
        self, multi_layer_dxf_importer: DxfImporter
    ):
        """Tests that vectorize() respects layer selection."""
        parse_result = multi_layer_dxf_importer.parse()
        assert parse_result is not None

        spec = PassthroughSpec(
            active_layer_ids=["Layer1", "Layer3"],
            create_new_layers=True,
        )
        vec_result = multi_layer_dxf_importer.vectorize(
            parse_result, spec
        )

        assert vec_result is not None
        assert "Layer1" in vec_result.geometries_by_layer
        assert "Layer3" in vec_result.geometries_by_layer
        assert "Layer2" not in vec_result.geometries_by_layer

    def test_create_source_asset(self, line_dxf_importer):
        """Tests that create_source_asset() returns SourceAsset."""
        parse_result = line_dxf_importer.parse()
        assert parse_result is not None

        source_asset = line_dxf_importer.create_source_asset(
            parse_result
        )

        assert source_asset.width_mm == 100.0
        assert source_asset.height_mm == 50.0
        assert source_asset.metadata is not None
        assert source_asset.metadata["is_vector"] is True


# Test cases
class TestDXFImporter:
    def test_empty_dxf(self, empty_dxf_importer):
        import_result = empty_dxf_importer.get_doc_items(
            vectorization_spec=None
        )
        assert import_result is not None
        assert import_result.payload.source is not None
        assert len(import_result.payload.items) == 0

    def test_line_conversion(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        # The line starts at (0,0) in the DXF. The engine preserves this.
        assert line_workpiece.pos == pytest.approx((0.0, 0.0))
        assert line_workpiece.size == pytest.approx((100.0, 50.0))

    def test_circle_conversion(self, circle_workpiece: WorkPiece):
        assert circle_workpiece is not None

        # Bbox is (25,25) to (75,75). The engine preserves this absolute
        # position.
        assert circle_workpiece.pos == pytest.approx((25.0, 25.0))
        assert circle_workpiece.size == pytest.approx((50.0, 50.0))

    def test_unit_conversion(self, inches_workpiece: WorkPiece):
        assert inches_workpiece is not None
        assert inches_workpiece.size == pytest.approx((25.4, 25.4))

    def test_get_natural_size(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        size = line_workpiece.natural_size
        assert size is not None
        assert size == pytest.approx((100.0, 50.0))

    def test_natural_size_is_static_after_resize(
        self, line_workpiece: WorkPiece
    ):
        """
        Tests the fix: verifies that the natural size is an intrinsic property
        of the import source and does not change when the workpiece is resized.
        """
        assert line_workpiece is not None
        initial_natural_size = line_workpiece.natural_size
        assert initial_natural_size == pytest.approx((100.0, 50.0))

        # Act: Resize the workpiece to a different size and aspect ratio
        line_workpiece.set_size(200.0, 25.0)
        assert line_workpiece.size == pytest.approx((200.0, 25.0))

        # Assert: The natural size should remain unchanged
        natural_size_after_resize = line_workpiece.natural_size
        assert natural_size_after_resize is not None
        assert natural_size_after_resize == pytest.approx(initial_natural_size)

    def test_get_aspect_ratio(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        ratio = line_workpiece.get_natural_aspect_ratio()
        assert ratio == pytest.approx(2.0)

    def test_render_to_pixels(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        surface = line_workpiece.render_to_pixels(width=200, height=100)
        assert surface is not None
        assert surface.get_width() == 200
        assert surface.get_height() == 100

    def test_render_chunk_generator(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        chunks = list(
            line_workpiece.render_chunk(
                pixels_per_mm_x=1,
                pixels_per_mm_y=1,
                max_chunk_width=40,
                max_chunk_height=30,
            )
        )
        assert len(chunks) == 6  # 3 cols x 2 rows

    def test_invalid_dxf_handling(self):
        invalid_dxf = b"invalid dxf content"
        importer = DxfImporter(invalid_dxf)
        import_result = importer.get_doc_items(vectorization_spec=None)
        assert import_result is not None
        assert import_result.payload is None
        assert import_result.parse_result is None
        assert len(import_result.errors) > 0

    def test_circle_dxf_not_linearized(
        self, circle_dxf_importer_from_file: DxfImporter
    ):
        # Force merge
        spec = PassthroughSpec(create_new_layers=False)
        import_result = circle_dxf_importer_from_file.get_doc_items(
            vectorization_spec=spec
        )
        assert import_result is not None
        payload = import_result.payload
        assert payload is not None
        payload = cast(ImportPayload, payload)
        assert len(payload.items) == 1
        item = payload.items[0]
        assert isinstance(item, WorkPiece)
        wp: Union[WorkPiece, Layer] = item

        boundaries = wp.boundaries
        assert boundaries is not None
        assert not boundaries.is_empty()
        data = boundaries.data
        assert data is not None
        has_bezier_commands = (data[:, 0] == CMD_TYPE_BEZIER).any()
        assert has_bezier_commands, (
            "Circle should contain bezier commands, not be linearized"
        )

    def test_acdbcircle_dxf_not_linearized(
        self, acdbcircle_dxf_importer_from_file: DxfImporter
    ):
        # Force merge
        spec = PassthroughSpec(create_new_layers=False)
        import_result = acdbcircle_dxf_importer_from_file.get_doc_items(
            vectorization_spec=spec
        )
        assert import_result is not None
        payload = import_result.payload
        assert payload is not None
        payload = cast(ImportPayload, payload)
        assert len(payload.items) == 1
        item = payload.items[0]
        assert isinstance(item, WorkPiece)
        wp: Union[WorkPiece, Layer] = item

        boundaries = wp.boundaries
        assert boundaries is not None
        assert not boundaries.is_empty()
        data = boundaries.data
        assert data is not None
        has_bezier_commands = (data[:, 0] == CMD_TYPE_BEZIER).any()
        assert has_bezier_commands, (
            "ACDB circle should contain bezier commands, not be linearized"
        )

    def test_multi_layer_dxf(self, multi_layer_dxf_importer: DxfImporter):
        """Test that multi-layer DXF files are imported correctly."""
        # Force a merge strategy to test the original intent of the test
        spec = PassthroughSpec(create_new_layers=False)
        import_result = multi_layer_dxf_importer.get_doc_items(
            vectorization_spec=spec
        )
        assert import_result is not None
        payload = import_result.payload
        assert payload is not None
        payload = cast(ImportPayload, payload)

        # With merge strategy, all layers are merged into one workpiece
        assert len(payload.items) == 1
        item = payload.items[0]
        assert isinstance(item, WorkPiece)
        wp = item
        assert wp is not None

    def test_multi_layer_dxf_with_layer_selection(
        self, multi_layer_dxf_importer: DxfImporter
    ):
        """Test that layer selection works correctly for multi-layer DXF."""

        # Import only Layer1 and Layer3, creating separate layers
        spec = PassthroughSpec(
            active_layer_ids=["Layer1", "Layer3"], create_new_layers=True
        )
        import_result = multi_layer_dxf_importer.get_doc_items(
            vectorization_spec=spec
        )
        assert import_result is not None
        payload = import_result.payload
        assert payload is not None
        payload = cast(ImportPayload, payload)
        assert len(payload.items) == 2

        # Verify only Layer1 and Layer3 are imported
        layer_names = set()
        for item in payload.items:
            if isinstance(item, Layer):
                layer_names.add(item.name)

        assert layer_names == {"Layer1", "Layer3"}
