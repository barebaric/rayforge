import pytest
from unittest.mock import MagicMock, ANY
from typing import cast
import numpy as np

from rayforge.context import get_context
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact.base import VertexData, TextureData
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
    verts = np.array(
        [
            [5, 5, 0],
            [15, 5, 0],
            [15, 5, 0],
            [15, 15, 0],
            [15, 15, 0],
            [5, 15, 0],
            [5, 15, 0],
            [5, 5, 0],
        ],
        dtype=np.float32,
    )
    # Full power (red)
    colors = np.full((verts.shape[0], 4), [1, 0, 0, 1], np.float32)
    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        vertex_data=VertexData(powered_vertices=verts, powered_colors=colors),
    )
    handle = get_context().artifact_store.put(artifact)
    yield handle
    get_context().artifact_store.release(handle)


@pytest.fixture
def texture_artifact_handle(context_initializer):
    """Creates and stores a texture-based WorkPieceArtifact."""
    # Y=0 in numpy is the top of the workpiece.
    # Top 30 rows are gray (power 128)
    top_chunk_data = np.full((30, 50), 128, dtype=np.uint8)
    # Bottom 20 rows are white (power 255)
    bottom_chunk_data = np.full((20, 50), 255, dtype=np.uint8)

    full_texture_data = np.zeros((50, 50), dtype=np.uint8)
    full_texture_data[0:30, :] = top_chunk_data
    full_texture_data[30:50, :] = bottom_chunk_data

    artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        texture_data=TextureData(
            power_texture_data=full_texture_data,
            dimensions_mm=(50.0, 50.0),
            position_mm=(0.0, 0.0),
        ),
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
        vector_artifact_handle.to_dict(),
        context.to_dict(),
        generation_id=1,
        creator_tag="test_view",
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
        texture_artifact_handle.to_dict(),
        context.to_dict(),
        generation_id=1,
        creator_tag="test_view",
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
