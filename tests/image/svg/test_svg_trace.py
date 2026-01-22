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

    # Content is 50x50. Padding is 50*0.01=0.5 on each side.
    # New viewbox is 24.5, 24.5, 51, 51
    px, py, pw, ph = result.document_bounds
    assert px == pytest.approx(24.5)
    assert py == pytest.approx(24.5)
    assert pw == pytest.approx(51.0)
    assert ph == pytest.approx(51.0)


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
@patch("rayforge.image.svg.svg_trace.image_util")
@patch("rayforge.image.svg.svg_trace.SVG_RENDERER")
def test_trace_pipeline_integration(
    mock_renderer, mock_util, mock_trace, trace_importer
):
    """Verifies the full Get Doc Items pipeline for tracing."""
    mock_vips = MagicMock()
    # Mocking a render based on the padded size of 51mm, assuming ~4000px
    # target. Let's say it renders to 4000x4000 for simplicity
    mock_vips.width = 4000
    mock_vips.height = 4000
    mock_vips.pngsave_buffer.return_value = b"fake_png_data"
    mock_vips.copy.return_value = mock_vips
    mock_renderer.render_base_image.return_value = mock_vips

    mock_util.normalize_to_rgba.return_value = mock_vips
    mock_util.vips_rgba_to_cairo_surface.return_value = MagicMock()
    # mm_per_pixel = 51mm / 4000px
    mm_per_px = 51.0 / 4000.0
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

    # Natural size should match the size of the rendered bitmap frame (51mm)
    assert item.natural_width_mm == pytest.approx(51.0)
    assert item.natural_height_mm == pytest.approx(51.0)

    # World position should correspond to the top-left of the bitmap frame
    # which is the padded viewbox origin (24.5mm), Y-inverted.
    # Untrimmed page is 100mm. `by`=24.5, `bh`=51. `dist=100-(24.5+51)=24.5`.
    wx, wy = item.matrix.transform_point((0, 0))
    assert wx == pytest.approx(24.5)
    assert wy == pytest.approx(24.5)


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
