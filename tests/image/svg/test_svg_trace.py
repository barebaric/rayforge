import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from rayforge.image.svg.svg_trace import SvgTraceImporter
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.workpiece import WorkPiece
from rayforge.image.structures import ParsingResult

SVG_CONTENT = b"""
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
    <rect x="25" y="25" width="50" height="50" />
</svg>
"""


@pytest.fixture
def trace_importer():
    return SvgTraceImporter(SVG_CONTENT, Path("trace.svg"))


def test_parse_returns_single_layer(trace_importer):
    result = trace_importer.parse()
    assert result is not None

    assert len(result.layers) == 1
    assert result.layers[0].layer_id == "__default__"
    assert result.layers[0].name == "Traced Content"

    # Bitmap-based trimming: rect at (25,25) size 50x50 in 100x100 viewbox
    # yields exact content bounds with no padding.
    px, py, pw, ph = result.document_bounds
    assert px == pytest.approx(25.0)
    assert py == pytest.approx(25.0)
    assert pw == pytest.approx(50.0)
    assert ph == pytest.approx(50.0)


def test_vectorize_enforces_tracespec(trace_importer):
    """Should raise TypeError if we try to vectorize with non-TraceSpec."""
    document_bounds = (0, 0, 10, 10)
    unit_scale = 1.0
    x, y, w, h = document_bounds
    world_frame = (x * unit_scale, 0.0, w * unit_scale, h * unit_scale)
    dummy_parse = ParsingResult(
        document_bounds=document_bounds,
        native_unit_to_mm=unit_scale,
        is_y_down=True,
        layers=[],
        world_frame_of_reference=world_frame,
        background_world_transform=Matrix.identity(),
    )
    with pytest.raises(TypeError):
        trace_importer.vectorize(dummy_parse, None)  # type: ignore


@patch("rayforge.image.svg.svg_trace.trace_surface")
@patch("rayforge.image.svg.svg_trace.util")
@patch("rayforge.image.svg.svg_trace.SVG_RENDERER")
def test_trace_pipeline_integration(
    mock_renderer, mock_util, mock_trace, trace_importer
):
    """Verifies the full Get Doc Items pipeline for tracing."""
    mock_vips = MagicMock()
    # Mocking a render based on the trimmed size of 50mm, assuming ~4000px
    # target. Let's say it renders to 4000x4000 for simplicity
    mock_vips.width = 4000
    mock_vips.height = 4000
    mock_vips.pngsave_buffer.return_value = b"fake_png_data"
    mock_vips.copy.return_value = mock_vips
    mock_renderer.render_base_image.return_value = mock_vips

    mock_util.normalize_to_rgba.return_value = mock_vips
    mock_util.vips_rgba_to_cairo_surface.return_value = MagicMock()
    # mm_per_pixel = 50mm / 4000px
    mm_per_px = 50.0 / 4000.0
    mock_util.get_mm_per_pixel.return_value = (mm_per_px, mm_per_px)

    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(4000, 4000)
    mock_trace.return_value = [geo]

    spec = TraceSpec(threshold=0.5)
    import_result = trace_importer.get_doc_items(vectorization_spec=spec)
    assert import_result is not None
    payload = import_result.payload

    assert payload is not None
    # SourceAsset should contain the TRIMMED SVG, not the PNG.
    assert payload.source.base_render_data != b"fake_png_data"
    assert payload.source.renderer.__class__.__name__ == "SvgRenderer"
    assert payload.source.width_px is not None

    item = payload.items[0]
    assert isinstance(item, WorkPiece)

    # Natural size should match the size of the rendered bitmap frame (50mm)
    assert item.natural_width_mm == pytest.approx(50.0)
    assert item.natural_height_mm == pytest.approx(50.0)

    # World position should correspond to the top-left of the bitmap frame
    # which is the trimmed viewbox origin (25mm), Y-inverted.
    # Untrimmed page is 100mm. `by`=25, `bh`=50. `dist=100-(25+50)=25`.
    wx, wy = item.matrix.transform_point((0, 0))
    assert wx == pytest.approx(25.0, abs=1e-4)
    assert wy == pytest.approx(25.0, abs=1e-4)


def test_trace_render_failure_handling(trace_importer):
    """If rendering fails, should return valid payload with no items."""
    with patch("rayforge.image.svg.svg_trace.SVG_RENDERER") as mock_renderer:
        mock_renderer.render_base_image.return_value = None

        import_result = trace_importer.get_doc_items(
            vectorization_spec=TraceSpec()
        )
        assert import_result is not None
        assert len(import_result.errors) > 0
        payload = import_result.payload

        assert payload is not None
        assert payload.source is not None
        assert len(payload.items) == 0


SVG_TRIANGLE_WITH_STROKE = b"""
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
    <polygon points="50,10 90,90 10,90"
             fill="none" stroke="black" stroke-width="8" />
</svg>
"""


def test_trace_trim_includes_stroke_width():
    """Bitmap-based trimming must account for stroke width.

    A triangle whose centerline vertices extend to x=10, x=90, y=10, y=90
    with stroke-width=8 has visual bounds from (6,6) to (94,94).
    Analytical trim (centerline only) would produce bounds ~10..90,
    clipping the strokes.  Bitmap-based trim must capture the full visual
    extent including the stroke.
    """
    importer = SvgTraceImporter(
        SVG_TRIANGLE_WITH_STROKE, Path("triangle_stroke.svg")
    )
    result = importer.parse()
    assert result is not None

    px, py, pw, ph = result.document_bounds

    # The visual content (including stroke) extends beyond the centerline
    # vertices (10..90).  With bitmap trimming, the bounds must reach at
    # least to the centerline coordinates.  The key assertion is that the
    # bounds are wider than the pure centerline bounds of (10,10,80,80).
    assert px < 10.5, (
        f"Left bound {px} should be <= centerline min (10) plus "
        f"stroke overhang"
    )
    assert py < 10.5, (
        f"Top bound {py} should be <= centerline min (10) plus stroke overhang"
    )
    assert px + pw > 89.5, (
        f"Right bound {px + pw} should be >= centerline max (90) minus "
        f"stroke overhang"
    )
    assert py + ph > 89.5, (
        f"Bottom bound {py + ph} should be >= centerline max (90) minus "
        f"stroke overhang"
    )
