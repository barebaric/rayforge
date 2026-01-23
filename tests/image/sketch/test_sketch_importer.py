import pytest
import json
from pathlib import Path
from rayforge.core.sketcher import Sketch
from rayforge.core.workpiece import WorkPiece
from rayforge.image.sketch.importer import SketchImporter
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.image.base_importer import ImporterFeature
from rayforge.core.geo import Geometry


@pytest.fixture
def complex_sketch() -> Sketch:
    """Creates a moderately complex sketch for serialization testing."""
    s = Sketch(name="Complex Sketch")  # Give fixture a more descriptive name

    # Parameters
    s.set_param("width", 100)
    s.set_param("height", "width * 0.5")  # 50
    s.set_param("radius", 10)

    # Geometry
    # A simple constrained rectangle
    p0 = s.origin_id  # ID 0
    p1 = s.add_point(100, 0)
    p2 = s.add_point(100, 50)
    p3 = s.add_point(0, 50)

    s.add_line(p0, p1)
    s.add_line(p1, p2)
    s.add_line(p2, p3)
    s.add_line(p3, p0)

    # Constraints
    s.constrain_horizontal(p0, p1)
    s.constrain_vertical(p1, p2)
    s.constrain_horizontal(p3, p2)
    s.constrain_vertical(p0, p3)
    s.constrain_distance(p0, p1, "width")
    s.constrain_distance(p0, p3, "height")

    # An arc for good measure
    p_center = s.add_point(120, 25)
    p_start = s.add_point(130, 25)
    p_end = s.add_point(120, 35)

    arc_id = s.add_arc(p_start, p_end, p_center, clockwise=False)
    s.constrain_radius(entity_id=arc_id, radius="radius")

    s.solve()  # Solve it to ensure a consistent state
    return s


class TestSketchImporterContract:
    """Tests for the Importer contract compliance of SketchImporter."""

    def test_class_attributes(self):
        """Tests that importer class has required attributes."""
        assert SketchImporter.label == "Rayforge Sketch"
        assert SketchImporter.extensions == (".rfs",)
        assert SketchImporter.features == {ImporterFeature.DIRECT_VECTOR}

    def test_scan_returns_manifest(self, complex_sketch: Sketch):
        """Tests that scan() returns ImportManifest with correct data."""
        sketch_dict = complex_sketch.to_dict()
        sketch_bytes = json.dumps(
            sketch_dict, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        importer = SketchImporter(
            data=sketch_bytes, source_file=Path("test.rfs")
        )
        manifest = importer.scan()

        assert manifest.title == "Complex Sketch"
        assert manifest.natural_size_mm is None
        assert len(manifest.warnings) == 0
        assert len(manifest.errors) == 0

    def test_scan_handles_invalid_data(self):
        """Tests that scan() handles invalid JSON gracefully."""
        importer = SketchImporter(b"not json", Path("invalid.rfs"))
        manifest = importer.scan()

        assert manifest.title == "invalid.rfs"
        assert len(manifest.errors) > 0

    def test_parse_returns_parsing_result(self, complex_sketch: Sketch):
        """Tests that parse() returns ParsingResult with correct data."""
        sketch_dict = complex_sketch.to_dict()
        sketch_bytes = json.dumps(
            sketch_dict, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        importer = SketchImporter(
            data=sketch_bytes, source_file=Path("test.rfs")
        )
        parse_result = importer.parse()

        assert parse_result is not None
        assert parse_result.is_y_down is False
        assert parse_result.native_unit_to_mm == 1.0
        assert len(parse_result.layers) == 1
        assert parse_result.layers[0].layer_id == "__default__"
        assert parse_result.layers[0].name == "__default__"
        assert parse_result.world_frame_of_reference is not None
        assert parse_result.background_world_transform is not None

    def test_parse_handles_invalid_data(self):
        """Tests that parse() returns None for invalid JSON."""
        importer = SketchImporter(b"not json")
        parse_result = importer.parse()

        assert parse_result is None
        assert len(importer._errors) > 0

    def test_vectorize_returns_vectorization_result(
        self, complex_sketch: Sketch
    ):
        """Tests that vectorize() returns VectorizationResult."""
        sketch_dict = complex_sketch.to_dict()
        sketch_bytes = json.dumps(
            sketch_dict, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        importer = SketchImporter(
            data=sketch_bytes, source_file=Path("test.rfs")
        )
        parse_result = importer.parse()
        assert parse_result is not None

        spec = PassthroughSpec(create_new_layers=False)
        vec_result = importer.vectorize(parse_result, spec)

        assert vec_result is not None
        assert vec_result.source_parse_result is parse_result
        assert "__default__" in vec_result.geometries_by_layer
        assert isinstance(
            vec_result.geometries_by_layer["__default__"], Geometry
        )
        assert "__default__" in vec_result.fills_by_layer

    def test_create_source_asset(self, complex_sketch: Sketch):
        """Tests that create_source_asset() returns SourceAsset."""
        sketch_dict = complex_sketch.to_dict()
        sketch_bytes = json.dumps(
            sketch_dict, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        importer = SketchImporter(
            data=sketch_bytes, source_file=Path("test.rfs")
        )
        parse_result = importer.parse()
        assert parse_result is not None

        source_asset = importer.create_source_asset(parse_result)

        assert source_asset.original_data == sketch_bytes
        assert source_asset.metadata is not None
        assert source_asset.metadata["is_vector"] is True


def test_sketch_importer_round_trip(complex_sketch: Sketch):
    """
    Tests that a sketch can be serialized, imported, and correctly
    reconstructed as a WorkPiece and Sketch object.
    """
    # 1. Serialize the original sketch to bytes
    original_dict = complex_sketch.to_dict()
    sketch_bytes = json.dumps(
        original_dict, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    # 2. Instantiate the importer with the serialized data
    importer = SketchImporter(
        data=sketch_bytes, source_file=Path("MyTestSketch.rfs")
    )

    # 3. Call get_doc_items() to get the payload
    # Force a merge strategy
    spec = PassthroughSpec(create_new_layers=False)
    import_result = importer.get_doc_items(spec)
    assert import_result is not None, "Importer failed to return result"
    payload = import_result.payload

    assert payload is not None
    assert importer.parsed_sketch is not None
    assert len(payload.sketches) == 1
    imported_sketch_template = payload.sketches[0]

    # 4. Check the WorkPiece in the payload
    assert len(payload.items) == 1
    item = payload.items[0]

    assert isinstance(item, WorkPiece)
    # The importer should prioritize the serialized name over the filename.
    assert item.name == complex_sketch.name

    assert item.source_segment is not None
    assert item.source_segment.source_asset_uid == payload.source.uid

    assert item.sketch_uid == complex_sketch.uid

    # 5. Verify the dimensions were set correctly on the WorkPiece
    geo = complex_sketch.to_geometry()
    min_x, min_y, max_x, max_y = geo.rect()
    expected_width = max_x - min_x
    expected_height = max_y - min_y
    assert item.natural_width_mm == pytest.approx(expected_width)
    assert item.natural_height_mm == pytest.approx(expected_height)
    assert item.natural_size == pytest.approx(
        (expected_width, expected_height)
    )

    # 6. Verify the sketch template itself was parsed correctly
    parsed_sketch_dict = imported_sketch_template.to_dict()
    assert parsed_sketch_dict == original_dict


def test_sketch_importer_naming_logic_serialized_priority(
    complex_sketch: Sketch,
):
    """
    Tests that if the JSON contains a name, it takes precedence over the
    file name.
    """
    complex_sketch.name = "SerializedName"
    data = json.dumps(complex_sketch.to_dict()).encode("utf-8")

    # Pass a conflicting filename
    importer = SketchImporter(data=data, source_file=Path("Filename.rfs"))
    spec = PassthroughSpec(create_new_layers=False)
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload

    assert payload is not None
    # Should use the name from JSON
    assert payload.sketches[0].name == "SerializedName"
    assert payload.items[0].name == "SerializedName"


def test_sketch_importer_naming_logic_filename_fallback(
    complex_sketch: Sketch,
):
    """
    Tests that if the JSON has no name (or empty), it falls back to the
    file name.
    """
    # Create data with empty name
    d = complex_sketch.to_dict()
    d["name"] = ""
    data = json.dumps(d).encode("utf-8")

    importer = SketchImporter(data=data, source_file=Path("MyDesign.rfs"))
    spec = PassthroughSpec(create_new_layers=False)
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload

    assert payload is not None
    # Should fall back to file stem
    assert payload.sketches[0].name == "MyDesign"
    assert payload.items[0].name == "MyDesign"


def test_sketch_importer_naming_logic_default_fallback(complex_sketch: Sketch):
    """
    Tests that if JSON has no name and no file is provided, the name
    defaults to "Untitled".
    """
    # Create data with missing name key entirely
    d = complex_sketch.to_dict()
    del d["name"]
    data = json.dumps(d).encode("utf-8")

    importer = SketchImporter(data=data, source_file=None)
    spec = PassthroughSpec(create_new_layers=False)
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload

    assert payload is not None
    # Should fall back to the default name "Untitled".
    assert payload.sketches[0].name == "Untitled"
    assert payload.items[0].name == "Untitled"


def test_sketch_importer_bad_data():
    """
    Tests that the importer returns ImportResult with None payload for invalid
    or corrupted data.
    """
    bad_data_1 = b"this is not json"
    bad_data_2 = b'{"not": "a sketch"}'

    importer1 = SketchImporter(data=bad_data_1)
    import_result1 = importer1.get_doc_items()
    assert import_result1 is not None
    assert import_result1.payload is None
    assert import_result1.parse_result is None
    assert len(import_result1.errors) > 0
    assert importer1.parsed_sketch is None

    importer2 = SketchImporter(data=bad_data_2)
    import_result2 = importer2.get_doc_items()
    assert import_result2 is not None
    assert import_result2.payload is None
    assert import_result2.parse_result is None
    assert len(import_result2.errors) > 0
    assert importer2.parsed_sketch is None


def test_sketch_importer_round_trip_mouse():
    """
    Tests that the mouse.rfs sketch can be imported and correctly
    reconstructed as a WorkPiece and Sketch object.
    """
    # 1. Read the original mouse.rfs file
    mouse_file = Path(__file__).parent / "mouse.rfs"
    original_data = mouse_file.read_bytes()

    # 2. Parse the original JSON for comparison
    original_dict = json.loads(original_data)

    # 3. Instantiate the importer with the serialized data
    importer = SketchImporter(data=original_data, source_file=mouse_file)

    # 4. Call get_doc_items() to get the payload
    spec = PassthroughSpec(create_new_layers=False)
    import_result = importer.get_doc_items(spec)
    assert import_result is not None, "Importer failed to return result"
    payload = import_result.payload
    assert payload is not None

    assert importer.parsed_sketch is not None
    assert len(payload.sketches) == 1
    imported_sketch_template = payload.sketches[0]

    # 5. Check the WorkPiece in the payload
    assert len(payload.items) == 1
    item = payload.items[0]

    assert isinstance(item, WorkPiece)
    # The importer should prioritize the serialized name over the filename.
    assert item.name == original_dict["name"]

    assert item.source_segment is not None
    assert item.source_segment.source_asset_uid == payload.source.uid

    assert item.sketch_uid == original_dict["uid"]

    # 6. Verify the dimensions were set correctly on the WorkPiece
    # The importer has already solved the sketch, so use the current geometry
    geo = imported_sketch_template.to_geometry()
    min_x, min_y, max_x, max_y = geo.rect()
    expected_width = max_x - min_x
    expected_height = max_y - min_y
    assert item.natural_width_mm == pytest.approx(expected_width)
    assert item.natural_height_mm == pytest.approx(expected_height)
    assert item.natural_size == pytest.approx(
        (expected_width, expected_height)
    )

    # 7. Verify the sketch template itself was parsed correctly
    # Note: Point coordinates may be np.float64 instead of plain floats
    parsed_sketch_dict = imported_sketch_template.to_dict()

    # Check top-level fields
    assert parsed_sketch_dict["uid"] == original_dict["uid"]
    assert parsed_sketch_dict["name"] == original_dict["name"]
    assert parsed_sketch_dict["type"] == original_dict["type"]
    assert parsed_sketch_dict["origin_id"] == original_dict["origin_id"]

    # Check input parameters
    assert (
        parsed_sketch_dict["input_parameters"]
        == original_dict["input_parameters"]
    )

    # Check params
    assert parsed_sketch_dict["params"] == original_dict["params"]

    # Check registry points (values may be np.float64)
    for orig_point, parsed_point in zip(
        original_dict["registry"]["points"],
        parsed_sketch_dict["registry"]["points"],
    ):
        assert orig_point["id"] == parsed_point["id"]
        assert orig_point["fixed"] == parsed_point["fixed"]
        assert pytest.approx(orig_point["x"]) == float(parsed_point["x"])
        assert pytest.approx(orig_point["y"]) == float(parsed_point["y"])

    # Check registry entities
    assert (
        parsed_sketch_dict["registry"]["entities"]
        == original_dict["registry"]["entities"]
    )

    # Check constraints
    assert parsed_sketch_dict["constraints"] == original_dict["constraints"]

    # Check fills (should now be lists, not tuples)
    assert parsed_sketch_dict["fills"] == original_dict["fills"]
