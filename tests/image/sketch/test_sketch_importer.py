import pytest
import json

# Sketcher components
from rayforge.core.sketcher import Sketch

# Importer to test
from rayforge.image.sketch.importer import SketchImporter


@pytest.fixture
def complex_sketch() -> Sketch:
    """Creates a moderately complex sketch for serialization testing."""
    s = Sketch()

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
    Tests that a sketch can be serialized to bytes and then perfectly
    deserialized back into an identical Sketch object.
    """
    # 1. Serialize the original sketch to bytes
    original_dict = complex_sketch.to_dict()
    sketch_bytes = json.dumps(
        original_dict, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    # 2. Instantiate the importer with the serialized data
    importer = SketchImporter(data=sketch_bytes)

    # 3. Call the simplified get_doc_items()
    # For this step, we expect it to return None but populate
    # self.parsed_sketch
    payload = importer.get_doc_items()
    assert payload is None, "Importer should not return a payload in Step 2"
    assert importer.parsed_sketch is not None, (
        "Importer failed to parse sketch"
    )

    # 4. Get the dictionary representation of the deserialized sketch
    parsed_sketch_dict = importer.parsed_sketch.to_dict()

    # 5. Assert that the original and deserialized sketches are identical
    # Comparing dictionaries is the most reliable way to check for equality.
    assert parsed_sketch_dict == original_dict


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
