import pytest
from unittest.mock import MagicMock
import numpy as np
import logging

# Imports from the application
from rayforge.image import SVG_RENDERER
from rayforge.core.workpiece import WorkPiece
from rayforge.core.geo import Geometry
from rayforge.core.step import Step
from rayforge.machine.models.machine import Laser, Machine
from rayforge.pipeline.artifact.handle import ArtifactHandle
from rayforge.pipeline.artifact.base import Artifact
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.producer.edge import EdgeTracer
from rayforge.pipeline.producer.depth import DepthEngraver
from rayforge.pipeline.transformer.multipass import MultiPassTransformer
from rayforge.pipeline.steprunner import run_step_in_subprocess
from rayforge.pipeline.modifier import MakeTransparent, ToGrayscale


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    """Provides a realistic machine configuration for tests."""
    test_laser = Laser()
    test_laser.max_power = 1000
    test_machine = Machine()
    test_machine.dimensions = (200, 150)
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.heads.clear()
    test_machine.add_head(test_laser)

    class TestConfig:
        machine = test_machine

    test_config = TestConfig()
    mocker.patch("rayforge.config.config", test_config)
    mocker.patch("builtins._", lambda s: s, create=True)
    return test_config


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
        result_dict, result_gen_id = run_step_in_subprocess(
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
        assert result_dict is not None
        handle = ArtifactHandle.from_dict(result_dict)
        reconstructed_artifact = ArtifactStore.get(handle)

        assert isinstance(reconstructed_artifact, Artifact)
        assert not reconstructed_artifact.ops.is_empty()
        assert reconstructed_artifact.generation_size == generation_size
        assert result_gen_id == generation_id
        # Verify vertex data was created and raster data was not
        assert reconstructed_artifact.vertex_data is not None
        assert reconstructed_artifact.raster_data is None
        assert reconstructed_artifact.vertex_data["powered_vertices"].size > 0
        assert reconstructed_artifact.vertex_data["powered_colors"].size > 0
    finally:
        # Cleanup
        if handle:
            ArtifactStore.release(handle)


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

    # Hydrate the workpiece dictionary like OpsGenerator does
    workpiece_dict = rasterable_workpiece.to_dict()
    workpiece_dict["data"] = rasterable_workpiece.data
    workpiece_dict["renderer_name"] = (
        rasterable_workpiece.renderer.__class__.__name__
    )

    try:
        # Act
        result_dict, result_gen_id = run_step_in_subprocess(
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
        assert result_dict is not None
        handle = ArtifactHandle.from_dict(result_dict)
        reconstructed_artifact = ArtifactStore.get(handle)

        assert isinstance(reconstructed_artifact, Artifact)
        assert reconstructed_artifact.raster_data is not None
        assert reconstructed_artifact.vertex_data is not None

        texture = reconstructed_artifact.raster_data["power_texture_data"]
        assert isinstance(texture, np.ndarray)
        assert reconstructed_artifact.generation_size == generation_size
        assert result_gen_id == generation_id

        # For a raster artifact, powered vertices should be empty (handled by
        # texture), but travel/zero-power moves (like overscan) should exist.
        assert reconstructed_artifact.vertex_data["powered_vertices"].size == 0
        assert reconstructed_artifact.vertex_data["powered_colors"].size == 0
    finally:
        # Cleanup
        if handle:
            ArtifactStore.release(handle)


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
    result, result_gen_id = run_step_in_subprocess(
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
    assert result is None
    assert result_gen_id == generation_id


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
        result_dict, result_gen_id = run_step_in_subprocess(
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
        assert result_dict is not None
        handle = ArtifactHandle.from_dict(result_dict)
        reconstructed_artifact = ArtifactStore.get(handle)

        assert isinstance(reconstructed_artifact, Artifact)
        assert reconstructed_artifact.vertex_data is not None
        assert len(reconstructed_artifact.ops.commands) == 24
    finally:
        # Cleanup
        if handle:
            ArtifactStore.release(handle)
