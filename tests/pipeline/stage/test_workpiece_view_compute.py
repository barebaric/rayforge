import numpy as np

from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    SetPowerCommand,
    ScanLinePowerCommand,
)
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact import (
    RenderContext,
    WorkPieceArtifact,
    WorkPieceViewArtifact,
)
from rayforge.pipeline.stage.workpiece_view_compute import (
    compute_view_dimensions,
    compute_workpiece_view,
    compute_workpiece_view_to_buffer,
    _get_content_bbox,
    render_chunk_to_buffer,
    _encode_vertex_and_texture_data,
    _calculate_render_dimensions,
    _setup_cairo_context,
    _draw_travel_vertices,
    _draw_zero_power_vertices,
    _prepare_powered_vertices_for_batching,
    _draw_powered_vertices_batch,
)
from rayforge.shared.util.colors import ColorSet


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


def test_compute_workpiece_view_vector_returns_valid_artifact():
    """
    Test that compute_workpiece_view returns a valid WorkPieceViewArtifact
    for vector data.
    """
    ops = Ops()
    ops.add(SetPowerCommand(1.0))
    ops.add(MoveToCommand((5.0, 5.0, 0.0)))
    ops.add(LineToCommand((15.0, 5.0, 0.0)))
    ops.add(LineToCommand((15.0, 15.0, 0.0)))
    ops.add(LineToCommand((5.0, 15.0, 0.0)))
    ops.add(LineToCommand((5.0, 5.0, 0.0)))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20, 20),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=1,
        color_set_dict=color_set.to_dict(),
    )

    result = compute_workpiece_view(artifact, context)

    assert result is not None
    assert isinstance(result, WorkPieceViewArtifact)
    assert result.bbox_mm == (5.0, 5.0, 10.0, 10.0)
    assert result.bitmap_data.shape == (12, 12, 4)


def test_compute_workpiece_view_texture_returns_valid_artifact():
    """
    Test that compute_workpiece_view returns a valid WorkPieceViewArtifact
    for texture data.
    """
    ops = Ops()
    for mm_y in range(1, 51):
        power_values = bytearray([128] * 50)
        ops.add(MoveToCommand((0.0, float(mm_y), 0.0)))
        ops.add(ScanLinePowerCommand((50.0, float(mm_y), 0.0), power_values))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(50, 50),
    )

    color_set = create_test_color_set({"engrave": ("#000", "#FFF")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = compute_workpiece_view(artifact, context)

    assert result is not None
    assert isinstance(result, WorkPieceViewArtifact)
    assert result.bbox_mm == (0.0, 0.0, 50.0, 50.0)
    assert result.bitmap_data.shape == (50, 50, 4)


def test_compute_workpiece_view_with_progress_callback(mock_progress_context):
    """Test that compute_workpiece_view calls progress callback."""
    ops = Ops()
    ops.add(SetPowerCommand(1.0))
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = compute_workpiece_view(artifact, context, mock_progress_context)

    assert result is not None
    assert isinstance(result, WorkPieceViewArtifact)
    assert result.bbox_mm == (0.0, 0.0, 10.0, 10.0)
    assert result.bitmap_data.shape == (10, 10, 4)

    assert len(mock_progress_context.progress_calls) > 0
    assert mock_progress_context.progress_calls[-1][0] == 1.0


def test_compute_workpiece_view_empty_ops_returns_none():
    """Test that compute_workpiece_view returns None for empty ops."""
    ops = Ops()
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = compute_workpiece_view(artifact, context)

    assert result is None


def test_compute_workpiece_view_travel_moves_shown():
    """Test that compute_workpiece_view renders travel moves when enabled."""
    ops = Ops()
    ops.add(SetPowerCommand(0.0))
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 0.0, 0.0)))
    ops.add(SetPowerCommand(1.0))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=True,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = compute_workpiece_view(artifact, context)

    assert result is not None
    assert isinstance(result, WorkPieceViewArtifact)
    assert result.bitmap_data.shape == (10, 10, 4)


def test_compute_view_dimensions_vector():
    """Test compute_view_dimensions with vector data."""
    ops = Ops()
    ops.add(SetPowerCommand(1.0))
    ops.add(MoveToCommand((5.0, 5.0, 0.0)))
    ops.add(LineToCommand((15.0, 15.0, 0.0)))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20, 20),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=1,
        color_set_dict=color_set.to_dict(),
    )

    result = compute_view_dimensions(artifact, context)

    assert result is not None
    x_mm, y_mm, w_mm, h_mm, width_px, height_px = result
    assert x_mm == 5.0
    assert y_mm == 5.0
    assert w_mm == 10.0
    assert h_mm == 10.0
    assert width_px == 12
    assert height_px == 12


def test_compute_view_dimensions_texture():
    """Test compute_view_dimensions with texture data."""
    ops = Ops()
    for mm_y in range(1, 11):
        power_values = bytearray([128] * 10)
        ops.add(MoveToCommand((0.0, float(mm_y), 0.0)))
        ops.add(ScanLinePowerCommand((10.0, float(mm_y), 0.0), power_values))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(10, 10),
    )

    color_set = create_test_color_set({"engrave": ("#000", "#FFF")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = compute_view_dimensions(artifact, context)

    assert result is not None
    x_mm, y_mm, w_mm, h_mm, width_px, height_px = result
    assert x_mm == 0.0
    assert y_mm == 0.0
    assert w_mm == 10.0
    assert h_mm == 10.0
    assert width_px == 10
    assert height_px == 10


def test_compute_view_dimensions_empty_ops():
    """Test compute_view_dimensions returns None for empty ops."""
    ops = Ops()
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    result = compute_view_dimensions(artifact, context)

    assert result is None


def test_compute_workpiece_view_to_buffer():
    """Test compute_workpiece_view_to_buffer renders to buffer."""
    ops = Ops()
    ops.add(SetPowerCommand(1.0))
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    bitmap = np.zeros((10, 10, 4), dtype=np.uint8)

    result = compute_workpiece_view_to_buffer(artifact, context, bitmap)

    assert result is not None
    x, y, w, h = result
    assert x == 0.0
    assert y == 0.0
    assert w == 10.0
    assert h == 10.0


def test_compute_workpiece_view_to_buffer_with_progress(
    mock_progress_context,
):
    """Test compute_workpiece_view_to_buffer with progress callback."""
    ops = Ops()
    ops.add(SetPowerCommand(1.0))
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    bitmap = np.zeros((10, 10, 4), dtype=np.uint8)

    result = compute_workpiece_view_to_buffer(
        artifact, context, bitmap, progress_context=mock_progress_context
    )

    assert result is not None
    assert len(mock_progress_context.progress_calls) > 0
    assert mock_progress_context.progress_calls[-1][0] == 1.0


def test_compute_workpiece_view_to_buffer_empty_ops():
    """Test compute_workpiece_view_to_buffer returns None for empty ops."""
    ops = Ops()
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    bitmap = np.zeros((10, 10, 4), dtype=np.uint8)

    result = compute_workpiece_view_to_buffer(artifact, context, bitmap)

    assert result is None


def test_render_chunk_to_buffer():
    """Test render_chunk_to_buffer renders chunk to buffer."""
    ops = Ops()
    ops.add(SetPowerCommand(1.0))
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(10.0, 10.0),
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

    result = render_chunk_to_buffer(artifact, context, bitmap, view_bbox_mm)

    assert result is True


def test_render_chunk_to_buffer_texture():
    """Test render_chunk_to_buffer with texture data."""
    ops = Ops()
    for mm_y in range(1, 6):
        power_values = bytearray([128] * 10)
        ops.add(MoveToCommand((0.0, float(mm_y), 0.0)))
        ops.add(ScanLinePowerCommand((10.0, float(mm_y), 0.0), power_values))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(10, 10),
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

    result = render_chunk_to_buffer(artifact, context, bitmap, view_bbox_mm)

    assert result is True


def test_get_content_bbox_powered_vertices():
    """Test _get_content_bbox with powered vertices."""
    from rayforge.pipeline.artifact.base import VertexData

    vertex_data = VertexData(
        powered_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        ),
        powered_colors=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        travel_vertices=np.array([]),
        zero_power_vertices=np.array([]),
    )

    result = _get_content_bbox(vertex_data, None, False)

    assert result is not None
    x, y, w, h = result
    assert x == 0.0
    assert y == 0.0
    assert w == 10.0
    assert h == 0.0


def test_get_content_bbox_travel_vertices():
    """Test _get_content_bbox with travel vertices."""
    from rayforge.pipeline.artifact.base import VertexData

    vertex_data = VertexData(
        powered_vertices=np.array([]),
        powered_colors=np.array([]),
        travel_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
            ]
        ),
        zero_power_vertices=np.array([]),
    )

    result = _get_content_bbox(vertex_data, None, True)

    assert result is not None
    x, y, w, h = result
    assert x == 0.0
    assert y == 0.0
    assert w == 10.0
    assert h == 10.0


def test_get_content_bbox_texture_data():
    """Test _get_content_bbox with texture data."""
    from rayforge.pipeline.artifact.base import (
        VertexData,
        TextureData,
    )

    vertex_data = VertexData(
        powered_vertices=np.array([]),
        powered_colors=np.array([]),
        travel_vertices=np.array([]),
        zero_power_vertices=np.array([]),
    )

    texture_data = TextureData(
        power_texture_data=np.zeros((10, 10), dtype=np.uint8),
        dimensions_mm=(10.0, 10.0),
        position_mm=(0.0, 0.0),
    )

    result = _get_content_bbox(vertex_data, texture_data, False)

    assert result is not None
    x, y, w, h = result
    assert x == 0.0
    assert y == 0.0
    assert w == 10.0
    assert h == 10.0


def test_get_content_bbox_empty():
    """Test _get_content_bbox returns None for empty data."""
    from rayforge.pipeline.artifact.base import VertexData

    vertex_data = VertexData(
        powered_vertices=np.array([]),
        powered_colors=np.array([]),
        travel_vertices=np.array([]),
        zero_power_vertices=np.array([]),
    )

    result = _get_content_bbox(vertex_data, None, False)

    assert result is None


def test_encode_vertex_and_texture_data_vector():
    """Test _encode_vertex_and_texture_data with vector data."""
    ops = Ops()
    ops.add(SetPowerCommand(1.0))
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    vertex_data, texture_data = _encode_vertex_and_texture_data(
        artifact, context
    )

    assert vertex_data is not None
    assert vertex_data.powered_vertices.size > 0
    assert texture_data is None


def test_encode_vertex_and_texture_data_texture():
    """Test _encode_vertex_and_texture_data with texture data."""
    ops = Ops()
    for mm_y in range(1, 6):
        power_values = bytearray([128] * 10)
        ops.add(MoveToCommand((0.0, float(mm_y), 0.0)))
        ops.add(ScanLinePowerCommand((10.0, float(mm_y), 0.0), power_values))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(10, 10),
    )

    color_set = create_test_color_set({"engrave": ("#000", "#FFF")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    vertex_data, texture_data = _encode_vertex_and_texture_data(
        artifact, context
    )

    assert vertex_data is not None
    assert texture_data is not None
    assert texture_data.power_texture_data.size > 0


def test_calculate_render_dimensions():
    """Test _calculate_render_dimensions returns valid dimensions."""
    bbox = (0.0, 0.0, 10.0, 10.0)
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=1,
        color_set_dict=color_set.to_dict(),
    )

    result = _calculate_render_dimensions(bbox, context)

    assert result is not None
    width_px, height_px, effective_ppm_x, effective_ppm_y = result
    assert width_px == 12
    assert height_px == 12
    assert effective_ppm_x == 1.0
    assert effective_ppm_y == 1.0


def test_calculate_render_dimensions_invalid():
    """Test _calculate_render_dimensions returns None for invalid bbox."""
    bbox = (0.0, 0.0, 0.0, 0.0)
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=1,
        color_set_dict=color_set.to_dict(),
    )

    result = _calculate_render_dimensions(bbox, context)

    assert result is None


def test_setup_cairo_context():
    """Test _setup_cairo_context creates valid cairo context."""
    bitmap = np.zeros((10, 10, 4), dtype=np.uint8)
    bbox = (0.0, 0.0, 10.0, 10.0)
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )

    ctx, line_width_mm = _setup_cairo_context(bitmap, bbox, context)

    assert ctx is not None
    assert line_width_mm == 1.0


def test_draw_travel_vertices():
    """Test _draw_travel_vertices draws travel vertices."""
    from rayforge.pipeline.artifact.base import VertexData
    import cairo

    vertex_data = VertexData(
        powered_vertices=np.array([]),
        powered_colors=np.array([]),
        travel_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
            ]
        ),
        zero_power_vertices=np.array([]),
    )

    color_set = create_test_color_set(
        {"cut": ("#000", "#F00"), "travel": ("#00F", "#00F")}
    )
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 20, 20)
    ctx = cairo.Context(surface)

    _draw_travel_vertices(ctx, vertex_data, color_set)


def test_draw_zero_power_vertices():
    """Test _draw_zero_power_vertices draws zero-power vertices."""
    from rayforge.pipeline.artifact.base import VertexData
    import cairo

    vertex_data = VertexData(
        powered_vertices=np.array([]),
        powered_colors=np.array([]),
        travel_vertices=np.array([]),
        zero_power_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
            ]
        ),
    )

    color_set = create_test_color_set(
        {"cut": ("#000", "#F00"), "zero_power": ("#0F0", "#0F0")}
    )
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 20, 20)
    ctx = cairo.Context(surface)

    _draw_zero_power_vertices(ctx, vertex_data, color_set)


def test_prepare_powered_vertices_for_batching():
    """Test _prepare_powered_vertices_for_batching."""
    from rayforge.pipeline.artifact.base import VertexData

    vertex_data = VertexData(
        powered_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
            ]
        ),
        powered_colors=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ),
        travel_vertices=np.array([]),
        zero_power_vertices=np.array([]),
    )

    color_set = create_test_color_set({"cut": ("#000", "#F00")})

    powered_v, unique_colors, inverse_indices = (
        _prepare_powered_vertices_for_batching(vertex_data, color_set)
    )

    assert powered_v.shape == (2, 2, 3)
    assert unique_colors.shape[0] > 0
    assert len(inverse_indices) == 2


def test_draw_powered_vertices_batch():
    """Test _draw_powered_vertices_batch draws a batch of vertices."""
    import cairo

    powered_v = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
        ]
    )
    unique_colors = np.array([[1.0, 0.0, 0.0, 1.0]])
    inverse_indices = np.array([0, 0])

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 20, 20)
    ctx = cairo.Context(surface)

    _draw_powered_vertices_batch(
        ctx, powered_v, unique_colors, inverse_indices, 0, 2
    )
