import pytest
from unittest.mock import MagicMock
import numpy as np
import logging

from rayforge.context import get_context
from rayforge.core.geo import Geometry
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.machine.models.machine import Laser
from rayforge.pipeline.artifact import (
    create_handle_from_dict,
    WorkPieceArtifact,
)
from rayforge.pipeline.modifier import MakeTransparent, ToGrayscale
from rayforge.pipeline.producer.edge import EdgeTracer
from rayforge.pipeline.producer.depth import DepthEngraver
from rayforge.pipeline.stage.workpiece_runner import (
    make_workpiece_artifact_in_subprocess,
)
from rayforge.pipeline.transformer.multipass import MultiPassTransformer


@pytest.fixture
def mock_proxy():
    """Mocks the ExecutionContextProxy passed to the subprocess."""
    proxy = MagicMock()
    proxy.sub_context.return_value = proxy  # Allow chaining
    proxy.parent_log_level = logging.DEBUG
    return proxy


@pytest.fixture
def base_workpiece():
    """Creates a WorkPiece with basic vector data."""
    geo = Geometry()
    geo.move_to(0, 0, 0)
    geo.line_to(10, 0, 0)
    geo.line_to(10, 10, 0)
    geo.line_to(0, 10, 0)
    geo.close_path()
    wp = WorkPiece(name="test_wp", vectors=geo)
    wp.set_size(25, 25)  # Set a physical size
    return wp


# This data is used by multiple tests to create the ImportSource.
SVG_DATA = b"""
<svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
<rect width="50" height="30" fill="black"/>
</svg>"""


@pytest.fixture
def rasterable_workpiece():
    """
    Creates a WorkPiece with a renderer and data, suitable for raster ops.
    """
    wp = WorkPiece(name="raster_wp.svg")
    # In a real app, this would be managed by the Doc, but we simulate it here
    # for the isolated subprocess test.
    wp._data = SVG_DATA
    wp._renderer = SVG_RENDERER
    wp.set_size(50, 30)
    return wp


def test_vector_producer_returns_artifact_with_vertex_data(
    mock_proxy, base_workpiece
):
    # Arrange
    step = Step(typelabel="Contour")
    step.opsproducer_dict = EdgeTracer().to_dict()
    settings = step.get_settings()
    laser = Laser()
    generation_id = 1
    generation_size = (25.0, 25.0)
    handle = None

    try:
        # Act
        result_gen_id = make_workpiece_artifact_in_subprocess(
            mock_proxy,
            base_workpiece.to_dict(),
            step.opsproducer_dict,
            [],
            [],
            laser.to_dict(),
            settings,
            generation_id,
            generation_size,
        )

        # Assert
        # 1. Check for the event and extract the handle
        event_call = next(
            (
                c
                for c in mock_proxy.send_event.call_args_list
                if c[0][0] == "__internal_artifact_created"
            ),
            None,
        )
        assert event_call is not None
        _, event_data = event_call[0]
        result_dict = event_data["handle_dict"]

        # 2. Continue with original assertions
        assert result_dict is not None
        handle = create_handle_from_dict(result_dict)
        reconstructed_artifact = get_context().artifact_store.get(handle)

        assert isinstance(reconstructed_artifact, WorkPieceArtifact)
        assert not reconstructed_artifact.ops.is_empty()
        assert reconstructed_artifact.generation_size == generation_size
        assert result_gen_id == generation_id
        # Verify vertex data was created and texture data was not
        assert reconstructed_artifact.vertex_data is not None
        assert reconstructed_artifact.texture_data is None
        assert reconstructed_artifact.vertex_data.powered_vertices.size > 0
        assert reconstructed_artifact.vertex_data.powered_colors.size > 0
    finally:
        # Cleanup
        if handle:
            get_context().artifact_store.release(handle)


def test_raster_producer_returns_artifact_with_raster_data(
    mock_proxy, rasterable_workpiece
):
    # Arrange
    step = Step(typelabel="Engrave")
    step.opsproducer_dict = DepthEngraver().to_dict()
    modifiers = [MakeTransparent().to_dict(), ToGrayscale().to_dict()]
    settings = step.get_settings()
    laser = Laser()
    generation_id = 2
    generation_size = (50.0, 30.0)
    handle = None

    # Hydrate the workpiece dictionary like Pipeline does
    workpiece_dict = rasterable_workpiece.to_dict()
    workpiece_dict["data"] = rasterable_workpiece.data
    workpiece_dict["renderer_name"] = (
        rasterable_workpiece.renderer.__class__.__name__
    )

    try:
        # Act
        result_gen_id = make_workpiece_artifact_in_subprocess(
            mock_proxy,
            workpiece_dict,
            step.opsproducer_dict,
            modifiers,
            [],
            laser.to_dict(),
            settings,
            generation_id,
            generation_size,
        )

        # Assert
        event_call = next(
            (
                c
                for c in mock_proxy.send_event.call_args_list
                if c[0][0] == "__internal_artifact_created"
            ),
            None,
        )
        assert event_call is not None
        _, event_data = event_call[0]
        result_dict = event_data["handle_dict"]

        assert result_dict is not None
        handle = create_handle_from_dict(result_dict)
        reconstructed_artifact = get_context().artifact_store.get(handle)

        assert isinstance(reconstructed_artifact, WorkPieceArtifact)
        assert reconstructed_artifact.texture_data is not None
        assert reconstructed_artifact.vertex_data is not None

        texture = reconstructed_artifact.texture_data.power_texture_data
        assert isinstance(texture, np.ndarray)
        assert reconstructed_artifact.generation_size == generation_size
        assert result_gen_id == generation_id

        # For a raster artifact, powered vertices should be empty (handled by
        # texture), but travel/zero-power moves (like overscan) should exist.
        assert reconstructed_artifact.vertex_data.powered_vertices.size == 0
        assert reconstructed_artifact.vertex_data.powered_colors.size == 0
    finally:
        # Cleanup
        if handle:
            get_context().artifact_store.release(handle)


def test_empty_producer_result_returns_none(mock_proxy):
    # Arrange: Create a workpiece with no renderable data
    empty_workpiece = WorkPiece(name="empty_wp")
    empty_workpiece.set_size(10, 10)

    step = Step(typelabel="Contour")
    step.opsproducer_dict = EdgeTracer().to_dict()
    settings = step.get_settings()
    laser = Laser()
    generation_id = 3
    generation_size = (10.0, 10.0)

    # Act
    result_gen_id = make_workpiece_artifact_in_subprocess(
        mock_proxy,
        empty_workpiece.to_dict(),
        step.opsproducer_dict,
        [],
        [],
        laser.to_dict(),
        settings,
        generation_id,
        generation_size,
    )

    # Assert
    # The runner returns the generation_id to signal completion.
    assert result_gen_id == generation_id
    # No event should be sent if no artifact was created.
    was_called = any(
        c.args[0] == "__internal_artifact_created"
        for c in mock_proxy.send_event.call_args_list
    )
    assert not was_called


def test_transformers_are_applied_before_put(mock_proxy, base_workpiece):
    # Arrange
    step = Step(typelabel="Contour")
    step.opsproducer_dict = EdgeTracer().to_dict()
    transformers = [MultiPassTransformer(passes=2).to_dict()]
    settings = step.get_settings()
    laser = Laser()
    generation_id = 4
    generation_size = (25.0, 25.0)
    handle = None

    # Expected command count:
    # 4 initial state + 8 from EdgeTracer = 12 commands
    # MultiPass(2) duplicates the whole block -> 12 * 2 = 24 commands

    try:
        # Act
        _ = make_workpiece_artifact_in_subprocess(
            mock_proxy,
            base_workpiece.to_dict(),
            step.opsproducer_dict,
            [],
            transformers,
            laser.to_dict(),
            settings,
            generation_id,
            generation_size,
        )

        # Assert
        event_call = next(
            (
                c
                for c in mock_proxy.send_event.call_args_list
                if c[0][0] == "__internal_artifact_created"
            ),
            None,
        )
        assert event_call is not None
        _, event_data = event_call[0]
        result_dict = event_data["handle_dict"]

        assert result_dict is not None
        handle = create_handle_from_dict(result_dict)
        reconstructed_artifact = get_context().artifact_store.get(handle)

        assert isinstance(reconstructed_artifact, WorkPieceArtifact)
        assert reconstructed_artifact.vertex_data is not None
        assert len(reconstructed_artifact.ops.commands) == 24
    finally:
        # Cleanup
        if handle:
            get_context().artifact_store.release(handle)
