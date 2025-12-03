import pytest
import json

# Core components
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece

# Sketcher components
from rayforge.core.sketcher import Sketch

# Exporter to test
from rayforge.image.sketch.exporter import SketchExporter


@pytest.fixture
def simple_sketch() -> Sketch:
    """Creates a simple sketch object."""
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 10)
    sketch.add_line(p1, p2)
    sketch.constrain_distance(p1, p2, "14.14")
    return sketch


def test_sketch_exporter_success(simple_sketch: Sketch):
    """
    Tests that the SketchExporter correctly extracts the sketch definition
    from a properly configured sketch-based WorkPiece.
    """
    # 1. Create a document and add the sketch to its registry.
    doc = Doc()
    doc.add_asset(simple_sketch)

    # 2. Create a workpiece and link it to the sketch via UID.
    #    For this test, it doesn't need a source_segment.
    workpiece = WorkPiece(name="MySketchWP")
    workpiece.sketch_uid = simple_sketch.uid
    doc.add_workpiece(workpiece)

    # 3. Instantiate the exporter and run it
    exporter = SketchExporter(workpiece)
    exported_bytes = exporter.export()

    # 4. Assert that the exported data matches the original sketch data
    expected_dict = simple_sketch.to_dict()
    exported_dict = json.loads(exported_bytes)
    assert exported_dict == expected_dict


def test_sketch_exporter_wrong_source_type():
    """
    Tests that the SketchExporter raises a ValueError if the WorkPiece
    is not based on a sketch (i.e., has no sketch_uid).
    """
    # 1. Create a workpiece that is NOT linked to a sketch
    workpiece = WorkPiece(name="NotASketch")
    doc = Doc()
    doc.add_workpiece(workpiece)

    # 2. Assert that creating the exporter and calling export raises an error
    exporter = SketchExporter(workpiece)
    with pytest.raises(ValueError, match="not based on a sketch"):
        exporter.export()


def test_sketch_exporter_init_with_wrong_item_type():
    """
    Tests that the SketchExporter raises a TypeError if initialized with
    something other than a WorkPiece.
    """
    doc = Doc()  # A DocItem that is not a WorkPiece
    with pytest.raises(TypeError, match="can only export WorkPiece items"):
        # This line is intentionally incorrect for testing purposes.
        # We tell the static type checker to ignore it.
        SketchExporter(doc)  # type: ignore
