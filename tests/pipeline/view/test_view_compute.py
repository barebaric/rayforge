import numpy as np
from raygeo.ops import Ops

from rayforge.core.color import ColorSet
from rayforge.pipeline.artifact import (
    RenderContext,
    WorkPieceArtifact,
)
from rayforge.pipeline.view.view_compute import (
    _get_content_bbox,
    calculate_render_dimensions,
    render_workpiece_view_in_process,
    stitch_chunk_to_bitmap,
)


def create_test_color_set(spec: dict) -> ColorSet:
    """Creates a mock resolved ColorSet for testing without GTK."""
    resolved_data = {}
    for key, colors in spec.items():
        lut = np.zeros((256, 4), dtype=np.float32)
        if key == "cut":
            lut[:, 0] = np.linspace(0, 1, 256)
            lut[:, 3] = 1.0
        elif key == "engrave":
            lut[:, 0] = np.linspace(0, 1, 256)
            lut[:, 1] = np.linspace(0, 1, 256)
            lut[:, 2] = np.linspace(0, 1, 256)
            lut[:, 3] = 1.0
        resolved_data[key] = lut
    return ColorSet(_data=resolved_data)


def _make_vector_artifact():
    ops = Ops()
    ops.set_power(1.0)
    ops.move_to(5.0, 5.0, 0.0)
    ops.line_to(15.0, 5.0, 0.0)
    ops.line_to(15.0, 15.0, 0.0)
    ops.line_to(5.0, 15.0, 0.0)
    ops.line_to(5.0, 5.0, 0.0)
    return WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        generation_size=(20, 20),
        generation_id=0,
    )


def _make_texture_artifact():
    ops = Ops()
    for mm_y in range(1, 51):
        power_values = bytearray([128] * 50)
        ops.move_to(0.0, float(mm_y), 0.0)
        ops.scan_to(50.0, float(mm_y), 0.0, power_values=power_values)
    return WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        generation_size=(50, 50),
        generation_id=0,
    )


# ──────────────────────────────────────────────────────────────────
# render_workpiece_view_in_process
# ──────────────────────────────────────────────────────────────────


def test_render_workpiece_view_vector():
    """Render a vector-only workpiece."""
    artifact = _make_vector_artifact()
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=1,
        color_set_dict=color_set.to_dict(),
    )

    result = render_workpiece_view_in_process(artifact, context)

    assert result is not None
    bitmap, bbox, wp_size = result
    assert bbox == (5.0, 5.0, 10.0, 10.0)
    assert wp_size == (20, 20)
    assert bitmap.shape[2] == 4
    assert bitmap.shape[0] > 0
    assert bitmap.shape[1] > 0


def test_render_workpiece_view_texture():
    """Render a texture (raster) workpiece."""
    artifact = _make_texture_artifact()
    color_set = create_test_color_set({"engrave": ("#000", "#FFF")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = render_workpiece_view_in_process(artifact, context)

    assert result is not None
    bitmap, bbox, wp_size = result
    assert bbox == (0.0, 0.0, 50.0, 50.0)
    assert wp_size == (50, 50)
    assert bitmap.shape[2] == 4
    assert bitmap.shape[0] > 0
    assert bitmap.shape[1] > 0


def test_render_workpiece_view_empty_ops():
    """Empty ops with no texture returns None."""
    ops = Ops()
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        generation_size=(20.0, 20.0),
        generation_id=0,
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = render_workpiece_view_in_process(artifact, context)
    assert result is None


def test_render_workpiece_view_travel_moves_shown():
    """Travel moves are rendered when enabled."""
    ops = Ops()
    ops.set_power(0.0)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.set_power(1.0)
    ops.line_to(10.0, 10.0, 0.0)
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        generation_size=(20.0, 20.0),
        generation_id=0,
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=True,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = render_workpiece_view_in_process(artifact, context)

    assert result is not None
    bitmap, bbox, wp_size = result
    assert bitmap.shape[2] == 4


# ──────────────────────────────────────────────────────────────────
# stitch_chunk_to_bitmap
# ──────────────────────────────────────────────────────────────────


def test_stitch_chunk_to_bitmap():
    """Stitch a chunk into a pre-allocated bitmap."""
    ops = Ops()
    ops.set_power(1.0)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        generation_size=(10.0, 10.0),
        generation_id=0,
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    bitmap = np.zeros((10, 10, 4), dtype=np.uint8)
    view_bbox_mm = (0.0, 0.0, 10.0, 10.0)

    result = stitch_chunk_to_bitmap(artifact, context, bitmap, view_bbox_mm)

    assert result is True
    assert bitmap[:, :, 3].max() > 0


def test_stitch_chunk_to_bitmap_texture():
    """Stitch a texture chunk into a pre-allocated bitmap."""
    ops = Ops()
    for mm_y in range(1, 6):
        power_values = bytearray([128] * 10)
        ops.move_to(0.0, float(mm_y), 0.0)
        ops.scan_to(10.0, float(mm_y), 0.0, power_values=power_values)
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        generation_size=(10, 10),
        generation_id=0,
    )

    color_set = create_test_color_set({"engrave": ("#000", "#FFF")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    bitmap = np.zeros((10, 10, 4), dtype=np.uint8)
    view_bbox_mm = (0.0, 0.0, 10.0, 10.0)

    result = stitch_chunk_to_bitmap(artifact, context, bitmap, view_bbox_mm)

    assert result is True


# ──────────────────────────────────────────────────────────────────
# _get_content_bbox
# ──────────────────────────────────────────────────────────────────


def test_get_content_bbox_vector():
    """Content bbox of a vector-only workpiece."""
    artifact = _make_vector_artifact()
    bbox = _get_content_bbox(artifact, show_travel=False)
    assert bbox is not None
    x, y, w, h = bbox
    assert x == 5.0
    assert y == 5.0
    assert w == 10.0
    assert h == 10.0


def test_get_content_bbox_texture():
    """Content bbox of a texture workpiece covers generation_size."""
    artifact = _make_texture_artifact()
    bbox = _get_content_bbox(artifact, show_travel=False)
    assert bbox is not None
    x, y, w, h = bbox
    assert x == 0.0
    assert y == 0.0
    assert w == 50.0
    assert h == 50.0


def test_get_content_bbox_empty():
    """Empty ops with scalable artifact returns None."""
    ops = Ops()
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        generation_size=(20.0, 20.0),
        generation_id=0,
    )
    bbox = _get_content_bbox(artifact, show_travel=False)
    assert bbox is None


# ──────────────────────────────────────────────────────────────────
# calculate_render_dimensions
# ──────────────────────────────────────────────────────────────────


def test_calculate_render_dimensions():
    """Valid bbox produces valid dimensions."""
    bbox = (0.0, 0.0, 10.0, 10.0)
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = calculate_render_dimensions(bbox, context)

    assert result is not None
    width_px, height_px, eff_ppm_x, eff_ppm_y = result
    assert width_px == 10
    assert height_px == 10
    assert eff_ppm_x == 1.0
    assert eff_ppm_y == 1.0


def test_calculate_render_dimensions_invalid():
    """Degenerate bbox returns None."""
    bbox = (0.0, 0.0, 0.0, 0.0)
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = calculate_render_dimensions(bbox, context)

    assert result is None
