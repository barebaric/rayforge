from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from raygeo.geo import Arc, Bezier, Geometry, Matrix

from rayforge.core.layer import Layer
from rayforge.core.vectorization_spec import LayerImportMode, PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image.base_importer import ImporterFeature
from rayforge.image.dxf.importer import DxfImporter
from rayforge.image.structures import ImportPayload

# Fixtures
FIXTURES = Path(__file__).parent


def _importer(name: str) -> DxfImporter:
    """Builds an importer for a checked-in DXF fixture.

    The fixtures are static files rather than documents assembled by a
    third-party DXF library, so the test suite exercises the reader
    against exactly the bytes that are committed.
    """
    path = FIXTURES / name
    return DxfImporter(path.read_bytes(), path)


def _has_curves(commands) -> bool:
    """True when the geometry keeps analytic curves rather than polylines.

    raygeo represents a circle as a few exact arcs, where the previous
    reader emitted beziers; either is acceptable, line segments are not.
    """
    return any(isinstance(cmd, (Arc, Bezier)) for cmd in commands)


@pytest.fixture
def empty_dxf_importer():
    return _importer("empty.dxf")


@pytest.fixture
def line_dxf_importer():
    return _importer("line.dxf")


@pytest.fixture
def circle_dxf_importer():
    return _importer("circle50.dxf")


@pytest.fixture
def inches_dxf_importer():
    return _importer("inches.dxf")


@pytest.fixture
def dxf3d_importer():
    """A DXF whose outlines exist at two Z levels (z=0 and z=-10),
    like a 3D CAD edge export of a 10mm thick part.
    """
    return _importer("z_variation.dxf")


@pytest.fixture
def multi_layer_dxf_importer():
    return _importer("multi_layer.dxf")


@pytest.fixture
def block_layer_dxf_importer():
    """A DXF with a block spanning two layers, where LAYER_B is used
    only inside the block (mirrors issue #773)."""
    return _importer("block_layers.dxf")


@pytest.fixture
def block_layer0_dxf_importer():
    """A DXF whose block content sits on layer "0", so it inherits the
    layer of the block insertion."""
    return _importer("block_layer0.dxf")


@pytest.fixture
def block_arc_dxf_importer():
    """A DXF whose block contains arcs and is inserted with a rotation
    (mirrors issue #834)."""
    return _importer("block_arc_rotation.dxf")


@pytest.fixture
def nested_block_dxf_importer():
    """A DXF whose block inserts another block (mirrors raygeo #197)."""
    return _importer("nested_block.dxf")


def _setup_workpiece_with_context(
    importer: DxfImporter,
) -> WorkPiece | None:
    """
    Helper to run importer, correctly link workpiece to its source,
    and mock the document context for rendering tests.
    """
    # Force a merge for simplicity in single-entity tests
    spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
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
    wp: WorkPiece | None = None
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
def line_workpiece(line_dxf_importer) -> WorkPiece | None:
    return _setup_workpiece_with_context(line_dxf_importer)


@pytest.fixture
def circle_workpiece(circle_dxf_importer) -> WorkPiece | None:
    return _setup_workpiece_with_context(circle_dxf_importer)


@pytest.fixture
def inches_workpiece(inches_dxf_importer) -> WorkPiece | None:
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
            ImporterFeature.Z_FLATTENING,
            ImporterFeature.GEOMETRY_REPAIR,
        }

    def test_scan_returns_manifest(self, line_dxf_importer):
        """Tests that scan() returns ImportManifest with correct data."""
        manifest = line_dxf_importer.scan()

        assert manifest.title == "line.dxf"
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

    def test_vectorize_returns_vectorization_result(self, line_dxf_importer):
        """Tests that vectorize() returns VectorizationResult."""
        parse_result = line_dxf_importer.parse()
        assert parse_result is not None

        spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
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
            layer_import_mode=LayerImportMode.NEW_LAYERS,
        )
        vec_result = multi_layer_dxf_importer.vectorize(parse_result, spec)

        assert vec_result is not None
        assert "Layer1" in vec_result.geometries_by_layer
        assert "Layer3" in vec_result.geometries_by_layer
        assert "Layer2" not in vec_result.geometries_by_layer

    def test_create_source_asset(self, line_dxf_importer):
        """Tests that create_source_asset() returns SourceAsset."""
        parse_result = line_dxf_importer.parse()
        assert parse_result is not None

        source_asset = line_dxf_importer.create_source_asset(parse_result)

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

    def test_unitless_file_without_header_is_metres(self, tmp_path):
        """A drawing that omits the HEADER section entirely carries no
        unit information, and such exports are conventionally in metres.
        Reading them as millimetres shrinks them by a factor of a
        thousand."""
        path = tmp_path / "noheader.dxf"
        path.write_text(
            "  0\nSECTION\n  2\nENTITIES\n"
            "  0\nLINE\n  8\n0\n"
            " 10\n0.0\n 20\n0.0\n 11\n0.1\n 21\n0.0\n"
            "  0\nENDSEC\n  0\nEOF\n"
        )

        manifest = DxfImporter(path.read_bytes(), path).scan()

        assert manifest.natural_size_mm == pytest.approx((100.0, 0.0))

    def test_header_without_insunits_is_native_units(self, tmp_path):
        """A file with a HEADER section but no $INSUNITS is read as
        written, not rescaled."""
        path = tmp_path / "nounits.dxf"
        path.write_text(
            "  0\nSECTION\n  2\nHEADER\n  9\n$ACADVER\n  1\nAC1015\n"
            "  0\nENDSEC\n  0\nSECTION\n  2\nENTITIES\n"
            "  0\nLINE\n  8\n0\n"
            " 10\n0.0\n 20\n0.0\n 11\n0.1\n 21\n0.0\n"
            "  0\nENDSEC\n  0\nEOF\n"
        )

        manifest = DxfImporter(path.read_bytes(), path).scan()

        assert manifest.natural_size_mm == pytest.approx((0.1, 0.0))

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
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
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
        wp: WorkPiece | Layer = item

        boundaries = wp.boundaries
        assert boundaries is not None
        assert not boundaries.is_empty()
        data = boundaries.data
        assert data is not None
        assert _has_curves(data), (
            "Circle should contain curve commands, not be linearized"
        )

    def test_spline_with_simple_interior_knots_has_no_loop(self):
        """A cubic spline whose interior knots have multiplicity one must
        be split into exact bezier spans. Readers that fell back to
        sampled fitting used to emit control points far outside the
        drawing, so the workpiece spanned a loop that does not exist in
        the file.
        """
        importer = _importer("spline_simple_knots.dxf")
        manifest = importer.scan()
        assert manifest.natural_size_mm == pytest.approx((15.0, 40.0))

        parse_result = importer.parse()
        assert parse_result is not None
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
        vec_result = importer.vectorize(parse_result, spec)
        geo = vec_result.geometries_by_layer[None]

        assert geo.rect() == pytest.approx(
            (
                499.5215204169139,
                273.4515044987481,
                514.5215204169139,
                313.4515044987481,
            )
        )
        assert _has_curves(geo.data)

    def test_lines_meeting_at_a_junction_are_not_chained(self):
        """Most lines of this drawing share endpoints with several
        others, so there is no unambiguous continuation. Chaining through
        such a junction used to trace a closed path whose windings cancel
        (zero area), which the contour assembler drops, leaving the
        operation with nothing to cut.
        """
        importer = _importer("junction.dxf")
        parse_result = importer.parse()
        assert parse_result is not None
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
        vec_result = importer.vectorize(parse_result, spec)
        geo = vec_result.geometries_by_layer[None]

        contours = geo.split_drawn_contours()
        assert len(contours) == 12
        drawn_closed = [
            c for c in contours if c.is_closed(1e-6) and c.distance() > 1e-6
        ]
        assert drawn_closed == []

    def test_acdbcircle_dxf_not_linearized(
        self, acdbcircle_dxf_importer_from_file: DxfImporter
    ):
        # Force merge
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
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
        wp: WorkPiece | Layer = item

        boundaries = wp.boundaries
        assert boundaries is not None
        assert not boundaries.is_empty()
        data = boundaries.data
        assert data is not None
        assert _has_curves(data), (
            "ACDB circle should contain curve commands, not be linearized"
        )

    def test_multi_layer_dxf(self, multi_layer_dxf_importer: DxfImporter):
        """Test that multi-layer DXF files are imported correctly."""
        # Force a merge strategy to test the original intent of the test
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
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
            active_layer_ids=["Layer1", "Layer3"],
            layer_import_mode=LayerImportMode.NEW_LAYERS,
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


def _z_values(geo: Geometry) -> set[float]:
    """Collects all Z coordinates used by a geometry's commands."""
    zs = set()
    for cmd in geo.iter_typed_commands():
        end = cmd.end
        zs.add(round(end[2], 6))
        if isinstance(cmd, Bezier):
            zs.add(round(cmd.control1[2], 6))
            zs.add(round(cmd.control2[2], 6))
    return zs


class TestDXFZFlattening:
    def test_features_advertise_z_flattening(self):
        assert ImporterFeature.Z_FLATTENING in DxfImporter.features

    def test_scan_detects_z_variation(self, dxf3d_importer):
        manifest = dxf3d_importer.scan()
        assert manifest is not None
        assert manifest.has_z_variation is True

    def test_scan_flat_dxf_has_no_z_variation(self, line_dxf_importer):
        manifest = line_dxf_importer.scan()
        assert manifest is not None
        assert manifest.has_z_variation is False

    def test_vectorize_flattens_z_by_default(self, dxf3d_importer):
        parse_result = dxf3d_importer.parse()
        assert parse_result is not None

        spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
        vec_result = dxf3d_importer.vectorize(parse_result, spec)

        assert _z_values(vec_result.geometries_by_layer[None]) == {0.0}

    def test_vectorize_preserves_z_when_disabled(self, dxf3d_importer):
        parse_result = dxf3d_importer.parse()
        assert parse_result is not None

        spec = PassthroughSpec(
            layer_import_mode=LayerImportMode.FLATTEN,
            flatten_z=False,
        )
        vec_result = dxf3d_importer.vectorize(parse_result, spec)

        assert _z_values(vec_result.geometries_by_layer[None]) == {
            0.0,
            -10.0,
        }

    def test_flatten_preserves_curves(self, dxf3d_importer):
        """Flattening must not degrade curves into line segments."""
        parse_result = dxf3d_importer.parse()
        assert parse_result is not None

        spec = PassthroughSpec(
            layer_import_mode=LayerImportMode.FLATTEN,
            flatten_z=False,
        )
        unflattened = dxf3d_importer.vectorize(parse_result, spec)
        vec_result = dxf3d_importer.vectorize(
            parse_result,
            PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN),
        )
        flat = vec_result.geometries_by_layer[None]

        def curve_count(geo):
            return sum(
                1
                for cmd in geo.iter_typed_commands()
                if isinstance(cmd, (Arc, Bezier))
            )

        assert curve_count(flat) == curve_count(
            unflattened.geometries_by_layer[None]
        )
        assert curve_count(flat) > 0
        for cmd in flat.iter_typed_commands():
            if isinstance(cmd, Bezier):
                assert cmd.control1[2] == 0.0
                assert cmd.control2[2] == 0.0

    def test_flatten_z_spec_roundtrip(self):
        spec = PassthroughSpec(flatten_z=False)
        restored = PassthroughSpec.from_dict(spec.to_dict())
        assert restored.flatten_z is False

    def test_flatten_z_defaults_true_for_legacy_specs(self):
        legacy = PassthroughSpec().to_dict()
        del legacy["flatten_z"]
        restored = PassthroughSpec.from_dict(legacy)
        assert restored.flatten_z is True


class TestDXFBlockLayers:
    def test_scan_counts_entities_inside_blocks(
        self, block_layer_dxf_importer: DxfImporter
    ):
        manifest = block_layer_dxf_importer.scan()
        counts = {layer.id: layer.feature_count for layer in manifest.layers}
        assert counts["LAYER_A"] == 2
        assert counts["LAYER_B"] == 1

    def test_vectorize_imports_block_only_layer(
        self, block_layer_dxf_importer: DxfImporter
    ):
        parse_result = block_layer_dxf_importer.parse()
        assert parse_result is not None
        spec = PassthroughSpec(
            active_layer_ids=["LAYER_B"],
            layer_import_mode=LayerImportMode.NEW_LAYERS,
        )
        vec_result = block_layer_dxf_importer.vectorize(parse_result, spec)
        geo = vec_result.geometries_by_layer["LAYER_B"]
        assert not geo.is_empty()
        x1, _y1, x2, _y2 = geo.rect()
        assert (x2 - x1) == pytest.approx(10.0)

    def test_block_entities_on_layer_zero_inherit_insert_layer(
        self, block_layer0_dxf_importer: DxfImporter
    ):
        manifest = block_layer0_dxf_importer.scan()
        counts = {layer.id: layer.feature_count for layer in manifest.layers}
        assert counts.get("LAYER_A") == 1

        parse_result = block_layer0_dxf_importer.parse()
        assert parse_result is not None
        spec = PassthroughSpec(
            active_layer_ids=["LAYER_A"],
            layer_import_mode=LayerImportMode.NEW_LAYERS,
        )
        vec_result = block_layer0_dxf_importer.vectorize(parse_result, spec)
        assert not vec_result.geometries_by_layer["LAYER_A"].is_empty()
        assert "0" not in vec_result.geometries_by_layer


class TestDXFBlockTransforms:
    def test_arcs_follow_the_block_insertion(
        self, block_arc_dxf_importer: DxfImporter
    ):
        parse_result = block_arc_dxf_importer.parse()
        assert parse_result is not None
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.NEW_LAYERS)
        vec_result = block_arc_dxf_importer.vectorize(parse_result, spec)

        outside = vec_result.geometries_by_layer["Outside"]
        assert not outside.is_empty()
        assert _has_curves(outside.data)
        min_x, min_y, max_x, max_y = outside.rect()
        assert (min_x, min_y, max_x, max_y) == pytest.approx(
            (3.386, 3.501, 101.830, 104.927), abs=0.01
        )

    def test_nested_block_insert_is_expanded(
        self, nested_block_dxf_importer: DxfImporter
    ):
        manifest = nested_block_dxf_importer.scan()
        assert [layer.id for layer in manifest.layers] == ["LAYER_A"]

        parse_result = nested_block_dxf_importer.parse()
        assert parse_result is not None
        spec = PassthroughSpec(layer_import_mode=LayerImportMode.NEW_LAYERS)
        vec_result = nested_block_dxf_importer.vectorize(parse_result, spec)

        geometry = vec_result.geometries_by_layer["LAYER_A"]
        assert not geometry.is_empty()
        min_x, min_y, max_x, max_y = geometry.rect()
        assert (min_x, min_y, max_x, max_y) == pytest.approx(
            (100.0, 5.0, 110.0, 15.0)
        )
