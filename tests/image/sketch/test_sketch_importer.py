import pytest
import json
from pathlib import Path

# Sketcher components
from rayforge.core.sketcher import Sketch
from rayforge.core.workpiece import WorkPiece

# Importer to test
from rayforge.image.sketch.importer import SketchImporter


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
    payload = importer.get_doc_items()
    assert payload is not None, "Importer failed to return payload"
    assert importer.parsed_sketch is not None
    assert len(payload.sketches) == 1
    imported_sketch_template = payload.sketches[0]

    # 4. Check the WorkPiece in the payload
    assert len(payload.items) == 1
    item = payload.items[0]

    assert isinstance(item, WorkPiece)
    # The importer should prioritize the serialized name over the filename.
    assert item.name == complex_sketch.name

    assert item.source_segment is None
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
    payload = importer.get_doc_items()

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
    payload = importer.get_doc_items()

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
    payload = importer.get_doc_items()

    assert payload is not None
    # Should fall back to the default name "Untitled".
    assert payload.sketches[0].name == "Untitled"
    assert payload.items[0].name == "Untitled"


def test_sketch_importer_bad_data():
    """
    Tests that the importer returns None for invalid or corrupted data.
    """
    bad_data_1 = b"this is not json"
    bad_data_2 = b'{"not": "a sketch"}'

    importer1 = SketchImporter(data=bad_data_1)
    assert importer1.get_doc_items() is None
    assert importer1.parsed_sketch is None

    importer2 = SketchImporter(data=bad_data_2)
    assert importer2.get_doc_items() is None
    assert importer2.parsed_sketch is None
