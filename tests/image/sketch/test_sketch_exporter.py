import pytest
import json
from pathlib import Path
import warnings

# Core components
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.geo import Geometry

# Sketcher components
from rayforge.core.sketcher import Sketch

# Exporter to test
from rayforge.image.sketch.exporter import SketchExporter

# Mock Renderer for testing
from rayforge.image.base_renderer import Renderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips


class MockBaseRenderer(Renderer):
    """A mock renderer to satisfy type hints and simulate identity."""

    def render_base_image(self, data, width, height, **kwargs) -> pyvips.Image:
        # Not used in this test
        return pyvips.Image.black(1, 1)


@pytest.fixture
def simple_sketch_data() -> tuple[Sketch, bytes]:
    """Creates a simple sketch and its serialized bytes."""
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 10)
    sketch.add_line(p1, p2)
    sketch.constrain_distance(p1, p2, "14.14")
    sketch_dict = sketch.to_dict()
    # Use separators=(',', ':') for a compact, predictable representation
    sketch_bytes = json.dumps(
        sketch_dict, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sketch, sketch_bytes


def test_sketch_exporter_success(simple_sketch_data):
    """
    Tests that the SketchExporter correctly extracts the original_data
    from a sketch-based WorkPiece.
    """
    _, sketch_bytes = simple_sketch_data

    # 1. Create a mock renderer class whose name matches the target.
    #    The architectural document specifies checking for "SketchImporter".
    class SketchImporter(MockBaseRenderer):
        pass

    # 2. Create the SourceAsset containing the sketch data
    source_asset = SourceAsset(
        source_file=Path("internal.rfs"),
        original_data=sketch_bytes,
        renderer=SketchImporter(),  # The class name of this instance matters
    )

    # 3. Create a document and a workpiece linked to the source
    doc = Doc()
    doc.add_source_asset(source_asset)

    segment = SourceAssetSegment(
        source_asset_uid=source_asset.uid,
        segment_mask_geometry=Geometry(),
        vectorization_spec=PassthroughSpec(),
    )
    workpiece = WorkPiece(name="MySketchWP", source_segment=segment)
    doc.add_workpiece(workpiece)

    # 4. Instantiate the exporter and run it
    exporter = SketchExporter(workpiece)
    exported_bytes = exporter.export()

    # 5. Assert that the exported data matches the original sketch bytes
    assert exported_bytes == sketch_bytes


def test_sketch_exporter_wrong_source_type():
    """
    Tests that the SketchExporter raises a ValueError if the WorkPiece
    is not based on a sketch.
    """

    # 1. Use a different renderer to simulate an SVG or other source
    class SvgImporter(MockBaseRenderer):
        pass

    source_asset = SourceAsset(
        source_file=Path("image.svg"),
        original_data=b"<svg></svg>",
        renderer=SvgImporter(),
    )

    # 2. Create document and workpiece
    doc = Doc()
    doc.add_source_asset(source_asset)
    segment = SourceAssetSegment(
        source_asset_uid=source_asset.uid,
        segment_mask_geometry=Geometry(),
        vectorization_spec=PassthroughSpec(),
    )
    workpiece = WorkPiece(name="MySvgWP", source_segment=segment)
    doc.add_workpiece(workpiece)

    # 3. Assert that creating the exporter and calling export raises an error
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
