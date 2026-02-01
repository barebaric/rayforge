import pytest
from unittest.mock import MagicMock
import numpy as np
import logging
from pathlib import Path

from rayforge.context import get_context
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.image import SVG_RENDERER
from rayforge.machine.models.machine import Laser
from rayforge.pipeline.artifact import (
    create_handle_from_dict,
    WorkPieceArtifact,
)
from rayforge.pipeline.modifier import MakeTransparent, ToGrayscale
from rayforge.pipeline.producer.contour import ContourProducer
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
    # The geometry created here represents the Y-down, normalized mask.
    geo = Geometry()
    geo.move_to(0, 0, 0)
    geo.line_to(1, 0, 0)
    geo.line_to(1, 1, 0)
    geo.line_to(0, 1, 0)
    geo.close_path()

    # In a real app, a SourceAsset would be created by the importer.
    # We simulate a minimal one here.
    source = SourceAsset(
        source_file=Path("test.dxf"), original_data=b"", renderer=MagicMock()
    )

    # Create the segment that defines the shape.
    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=geo,
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )

    # Create the workpiece using the new constructor.
    wp = WorkPiece(name="test_wp", source_segment=segment)
    wp.set_size(25, 25)  # Set a physical size
    return wp


# This data is used by multiple tests to create the SourceAsset.
SVG_DATA = b"""
<svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
<rect width="50" height="30" fill="black"/>
</svg>"""


@pytest.fixture
def rasterable_workpiece():
    """
    Creates a WorkPiece with a renderer and data, suitable for raster ops.
    """
    # Create a dummy 1x1 geometry for the segment mask.
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 0)
    geo.line_to(1, 1)
    geo.line_to(0, 1)
    geo.close_path()

    source = SourceAsset(
        source_file=Path("raster.svg"),
        original_data=SVG_DATA,
        renderer=SVG_RENDERER,
    )
    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=geo,
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )

    wp = WorkPiece(name="raster_wp.svg", source_segment=segment)
    # The _data and _renderer are hydrated for the subprocess test.
    wp._data = SVG_DATA
    wp._renderer = SVG_RENDERER
    wp.set_size(50, 30)
    return wp


def test_vector_producer_returns_artifact_with_vertex_data(
    mock_proxy, base_workpiece
):
    # Arrange
    step = Step(typelabel="Contour")
    step.opsproducer_dict = ContourProducer().to_dict()
    settings = step.get_settings()
    laser = Laser()
    generation_id = 1
    generation_size = (25.0, 25.0)
    handle = None

    try:
        # Act
        result_gen_id = make_workpiece_artifact_in_subprocess(
            mock_proxy,
            get_context().artifact_store,
            base_workpiece.to_dict(),
            step.opsproducer_dict,
            [],
            [],
            laser.to_dict(),
            settings,
            generation_id,
            generation_size,
            "test_workpiece",
        )

        # Assert
        # 1. Check for the event and extract the handle
        event_call = next(
            (
                c
                for c in mock_proxy.send_event.call_args_list
                if c[0][0] == "artifact_created"
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

    try:
        # Act
        result_gen_id = make_workpiece_artifact_in_subprocess(
            mock_proxy,
            get_context().artifact_store,
            workpiece_dict,
            step.opsproducer_dict,
            modifiers,
            [],
            laser.to_dict(),
            settings,
            generation_id,
            generation_size,
            "test_workpiece",
        )

        # Assert
        event_call = next(
            (
                c
                for c in mock_proxy.send_event.call_args_list
                if c[0][0] == "artifact_created"
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
    # Arrange: Create a workpiece with no renderable data by giving it an
    # empty geometry in its source segment.
    empty_source = SourceAsset(
        source_file=Path("empty"), original_data=b"", renderer=MagicMock()
    )
    empty_segment = SourceAssetSegment(
        source_asset_uid=empty_source.uid,
        pristine_geometry=Geometry(),  # Empty geometry
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )
    empty_workpiece = WorkPiece(name="empty_wp", source_segment=empty_segment)
    empty_workpiece.set_size(10, 10)

    step = Step(typelabel="Contour")
    step.opsproducer_dict = ContourProducer().to_dict()
    settings = step.get_settings()
    laser = Laser()
    generation_id = 3
    generation_size = (10.0, 10.0)

    # Act
    result_gen_id = make_workpiece_artifact_in_subprocess(
        mock_proxy,
        get_context().artifact_store,
        empty_workpiece.to_dict(),
        step.opsproducer_dict,
        [],
        [],
        laser.to_dict(),
        settings,
        generation_id,
        generation_size,
        "test_workpiece",
    )

    # Assert
    # The runner returns the generation_id to signal completion.
    assert result_gen_id == generation_id
    # No "artifact_created" event should be sent if no artifact was created.
    was_called = any(
        c[0][0] == "artifact_created"
        for c in mock_proxy.send_event.call_args_list
    )
    assert not was_called


def test_transformers_are_applied_before_put(mock_proxy, base_workpiece):
    # Arrange
    step = Step(typelabel="Contour")
    step.opsproducer_dict = ContourProducer().to_dict()
    transformers = [MultiPassTransformer(passes=2).to_dict()]
    settings = step.get_settings()
    laser = Laser()
    generation_id = 4
    generation_size = (25.0, 25.0)
    handle = None

    # Expected command count:
    # 4 initial state + 8 from ContourProducer = 12 commands
    # It may contain more due to inner/outer edge separation
    # MultiPass(2) duplicates the whole block -> 12 * 2 = 24 commands

    try:
        # Act
        _ = make_workpiece_artifact_in_subprocess(
            mock_proxy,
            get_context().artifact_store,
            base_workpiece.to_dict(),
            step.opsproducer_dict,
            [],
            transformers,
            laser.to_dict(),
            settings,
            generation_id,
            generation_size,
            "test_workpiece",
        )

        # Assert
        event_call = next(
            (
                c
                for c in mock_proxy.send_event.call_args_list
                if c[0][0] == "artifact_created"
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
        assert len(reconstructed_artifact.ops.commands) >= 24
    finally:
        # Cleanup
        if handle:
            get_context().artifact_store.release(handle)
