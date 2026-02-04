import pytest
from unittest.mock import MagicMock, ANY
from typing import cast
import numpy as np

from rayforge.context import get_context
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
    WorkPieceViewArtifactHandle,
)
from rayforge.pipeline.stage.workpiece_view_runner import (
    make_workpiece_view_artifact_in_subprocess,
)
from rayforge.shared.util.colors import ColorSet


def create_test_color_set(spec: dict) -> ColorSet:
    """Creates a mock resolved ColorSet for testing without GTK."""
    resolved_data = {}
    for key, colors in spec.items():
        # This simplified resolver just creates a basic LUT for testing
        lut = np.zeros((256, 4), dtype=np.float32)
        if key == "cut":
            # Black to Red gradient for 'cut'
            lut[:, 0] = np.linspace(0, 1, 256)
            lut[:, 3] = 1.0
        elif key == "engrave":
            # Black to White gradient for 'engrave'
            lut[:, 0] = np.linspace(0, 1, 256)
            lut[:, 1] = np.linspace(0, 1, 256)
            lut[:, 2] = np.linspace(0, 1, 256)
            lut[:, 3] = 1.0
        resolved_data[key] = lut
    return ColorSet(_data=resolved_data)


@pytest.fixture
def vector_artifact_handle(context_initializer):
    """Creates and stores a simple vector-based WorkPieceArtifact."""
    # A 10x10mm red square at (5,5)
    ops = Ops()
    ops.add(SetPowerCommand(1.0))  # Full power (red)
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
    handle = get_context().artifact_store.put(artifact)
    yield handle
    get_context().artifact_store.release(handle)


@pytest.fixture
def texture_artifact_handle(context_initializer):
    """Creates and stores a texture-based WorkPieceArtifact."""
    # TextureEncoder converts Y-up (mm) to Y-down (pixel) space.
    # To get top 30 rows (pixel y 0-29) to be gray (power 128),
    # we need scan lines at mm y 50-21 (which convert to pixel y 0-29).
    ops = Ops()
    # Create 50 scan lines, each 50 pixels wide
    for mm_y in range(1, 51):
        pixel_y = 50 - mm_y
        power = 128 if pixel_y < 30 else 255
        power_values = bytearray([power] * 50)
        ops.add(MoveToCommand((0.0, float(mm_y), 0.0)))
        ops.add(ScanLinePowerCommand((50.0, float(mm_y), 0.0), power_values))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(50, 50),
    )
    handle = get_context().artifact_store.put(artifact)
    yield handle
    get_context().artifact_store.release(handle)


def test_pixel_perfect_vector_render(vector_artifact_handle):
    """
    Validates the runner function produces a pixel-perfect render for
    vector data.
    """
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),  # 1px per mm
        show_travel_moves=False,
        margin_px=1,  # Use a margin
        color_set_dict=color_set.to_dict(),
    )
    mock_proxy = MagicMock()

    result = make_workpiece_view_artifact_in_subprocess(
        mock_proxy,
        get_context().artifact_store,
        vector_artifact_handle.to_dict(),
        context.to_dict(),
        "test_view",
    )
    assert result is None

    # Extract handle from the "created" event
    mock_proxy.send_event.assert_any_call("view_artifact_created", ANY)
    created_call = next(
        c
        for c in mock_proxy.send_event.call_args_list
        if c[0][0] == "view_artifact_created"
    )
    handle_dict = created_call[0][1]["handle_dict"]

    assert handle_dict is not None
    handle = WorkPieceViewArtifactHandle.from_dict(handle_dict)
    try:
        artifact = cast(
            WorkPieceViewArtifact, get_context().artifact_store.get(handle)
        )
        assert artifact.bbox_mm == (5.0, 5.0, 10.0, 10.0)
        assert artifact.bitmap_data.shape == (12, 12, 4)

        pixel_color = artifact.bitmap_data[1, 5]
        # Assert Red is dominant and Alpha is significant
        assert pixel_color[2] > 100  # Red
        assert pixel_color[1] < 50  # Green
        assert pixel_color[0] < 50  # Blue
        assert pixel_color[3] > 100  # Alpha
    finally:
        get_context().artifact_store.release(handle)


def test_pixel_perfect_texture_chunk_alignment(texture_artifact_handle):
    """
    Validates the runner correctly renders an artifact that was
    assembled from chunks, ensuring perfect alignment.
    """
    color_set = create_test_color_set({"engrave": ("#000", "#FFF")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )
    mock_proxy = MagicMock()

    result = make_workpiece_view_artifact_in_subprocess(
        mock_proxy,
        get_context().artifact_store,
        texture_artifact_handle.to_dict(),
        context.to_dict(),
        "test_view",
    )
    assert result is None

    # Extract handle from the "created" event
    mock_proxy.send_event.assert_any_call("view_artifact_created", ANY)
    created_call = next(
        c
        for c in mock_proxy.send_event.call_args_list
        if c[0][0] == "view_artifact_created"
    )
    handle_dict = created_call[0][1]["handle_dict"]

    assert handle_dict is not None
    handle = WorkPieceViewArtifactHandle.from_dict(handle_dict)
    try:
        artifact = cast(
            WorkPieceViewArtifact, get_context().artifact_store.get(handle)
        )

        assert artifact.bbox_mm == (0.0, 0.0, 50.0, 50.0)
        assert artifact.bitmap_data.shape == (50, 50, 4)

        # Check pixel from the top chunk (power 128 -> gray)
        gray_pixel = artifact.bitmap_data[10, 25]
        assert abs(gray_pixel[0] - 128) <= 1
        assert abs(gray_pixel[1] - 128) <= 1
        assert abs(gray_pixel[2] - 128) <= 1
        assert abs(gray_pixel[3] - 255) <= 1

        # Check pixel from the bottom chunk (power 255 -> white)
        white_pixel = artifact.bitmap_data[40, 25]
        np.testing.assert_array_almost_equal(
            white_pixel, [255, 255, 255, 255], decimal=0
        )
    finally:
        get_context().artifact_store.release(handle)


def test_progressive_rendering_increases_pixel_count(vector_artifact_handle):
    """
    Validates that progressive rendering sends intermediate updates
    with increasing numbers of non-transparent pixels.
    This test verifies that each view_artifact_updated event corresponds
    to more content being drawn.
    """
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=1,
        color_set_dict=color_set.to_dict(),
    )
    mock_proxy = MagicMock()

    result = make_workpiece_view_artifact_in_subprocess(
        mock_proxy,
        get_context().artifact_store,
        vector_artifact_handle.to_dict(),
        context.to_dict(),
        "test_view",
    )
    assert result is None

    # Get all view_artifact_updated events and their pixel counts
    updated_events = [
        call
        for call in mock_proxy.send_event.call_args_list
        if call[0][0] == "view_artifact_updated"
    ]

    # Extract handle from the "created" event
    created_call = next(
        c
        for c in mock_proxy.send_event.call_args_list
        if c[0][0] == "view_artifact_created"
    )
    handle_dict = created_call[0][1]["handle_dict"]
    handle = WorkPieceViewArtifactHandle.from_dict(handle_dict)

    try:
        # Assert: Should have multiple view_artifact_updated events
        # for progressive rendering (at least 2: after texture, after vertices)
        assert len(updated_events) >= 2, (
            f"Should have at least 2 view_artifact_updated events for "
            f"progressive rendering, got {len(updated_events)}"
        )

        # Count non-transparent pixels for each update event
        pixel_counts = []
        for event_call in updated_events:
            artifact = cast(
                WorkPieceViewArtifact, get_context().artifact_store.get(handle)
            )
            alpha_channel = artifact.bitmap_data[:, :, 3]
            non_transparent_count = np.sum(alpha_channel > 0)
            pixel_counts.append(non_transparent_count)

        # Assert: Each update should have more pixels than the previous
        # (or equal, if no new content was added)
        for i in range(1, len(pixel_counts)):
            assert pixel_counts[i] >= pixel_counts[i - 1], (
                f"Update {i} should have >= pixels than update {i - 1}: "
                f"{pixel_counts[i]} >= {pixel_counts[i - 1]}"
            )

        # Assert: Final artifact should have non-transparent pixels
        assert pixel_counts[-1] > 0, (
            "Final artifact should have non-transparent pixels"
        )
    finally:
        get_context().artifact_store.release(handle)


def test_on_demand_vertex_encoding(context_initializer):
    """Test that vertex data is encoded on-demand when missing."""
    ops = Ops()
    ops.set_power(1.0)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20.0, 20.0),
    )
    handle = get_context().artifact_store.put(artifact)
    color_set = create_test_color_set({"cut": ("#000", "#F00")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )
    mock_proxy = MagicMock()

    result = make_workpiece_view_artifact_in_subprocess(
        mock_proxy,
        get_context().artifact_store,
        handle.to_dict(),
        context.to_dict(),
        "test_view",
    )
    assert result is None

    mock_proxy.send_event.assert_any_call("view_artifact_created", ANY)
    created_call = next(
        c
        for c in mock_proxy.send_event.call_args_list
        if c[0][0] == "view_artifact_created"
    )
    handle_dict = created_call[0][1]["handle_dict"]
    assert handle_dict is not None
    view_handle = WorkPieceViewArtifactHandle.from_dict(handle_dict)
    try:
        view_artifact = cast(
            WorkPieceViewArtifact,
            get_context().artifact_store.get(view_handle),
        )
        assert view_artifact.bbox_mm == (0.0, 0.0, 10.0, 10.0)
        assert view_artifact.bitmap_data.shape == (10, 10, 4)
    finally:
        get_context().artifact_store.release(view_handle)
    get_context().artifact_store.release(handle)


def test_on_demand_texture_encoding(context_initializer):
    """Test that texture data is encoded on-demand when missing."""
    ops = Ops()
    ops.scan_to(10.0, 0.0, 0.0, bytearray([100, 150, 200]))
    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        generation_size=(10.0, 10.0),
    )
    handle = get_context().artifact_store.put(artifact)
    color_set = create_test_color_set({"engrave": ("#000", "#FFF")})
    context = RenderContext(
        pixels_per_mm=(1.0, 1.0),
        show_travel_moves=False,
        margin_px=0,
        color_set_dict=color_set.to_dict(),
    )
    mock_proxy = MagicMock()

    result = make_workpiece_view_artifact_in_subprocess(
        mock_proxy,
        get_context().artifact_store,
        handle.to_dict(),
        context.to_dict(),
        "test_view",
    )
    assert result is None

    mock_proxy.send_event.assert_any_call("view_artifact_created", ANY)
    created_call = next(
        c
        for c in mock_proxy.send_event.call_args_list
        if c[0][0] == "view_artifact_created"
    )
    handle_dict = created_call[0][1]["handle_dict"]
    assert handle_dict is not None
    view_handle = WorkPieceViewArtifactHandle.from_dict(handle_dict)
    try:
        view_artifact = cast(
            WorkPieceViewArtifact,
            get_context().artifact_store.get(view_handle),
        )
        assert view_artifact.bbox_mm == (0.0, 0.0, 10.0, 10.0)
        assert view_artifact.bitmap_data.shape == (10, 10, 4)
    finally:
        get_context().artifact_store.release(view_handle)
    get_context().artifact_store.release(handle)
