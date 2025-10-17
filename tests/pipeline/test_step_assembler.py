import pytest
from unittest.mock import MagicMock
import numpy as np

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    StepArtifact,
    ArtifactStore,
    create_handle_from_dict,
    TextureData,
)
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.core.matrix import Matrix
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)


@pytest.fixture
def machine():
    m = Machine()
    m.add_head(Laser())
    return m


def test_step_assembler_correctly_scales_and_places_ops(machine):
    """
    Test that the assembler correctly scales a scalable artifact and then
    applies a placement-only transform.
    """
    # Arrange
    doc = Doc()
    layer = doc.active_layer
    wp = WorkPiece(name="wp1")
    # Final state: 20x10mm size, translated to (50, 60)
    wp.matrix = Matrix.translation(50, 60) @ Matrix.scale(20, 10)
    layer.add_workpiece(wp)

    # Source artifact is scalable (e.g., from SVG) with 100x50 unit dimensions
    base_ops = Ops()
    base_ops.line_to(100, 0)  # A 100-unit line in local source coordinates
    base_artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        source_dimensions=(100, 50),
    )
    base_handle = ArtifactStore.put(base_artifact)

    assembly_info = [
        {
            "artifact_handle_dict": base_handle.to_dict(),
            "world_transform_list": wp.get_world_transform().to_list(),
            "workpiece_dict": wp.to_dict(),
        }
    ]
    mock_proxy = MagicMock()

    # Act
    result = make_step_artifact_in_subprocess(
        proxy=mock_proxy,
        workpiece_assembly_info=assembly_info,
        step_uid="step1",
        generation_id=1,
        per_step_transformers_dicts=[],
    )
    assert result is not None
    final_handle_dict, _ = result

    # Assert
    final_handle = create_handle_from_dict(final_handle_dict)
    final_artifact = ArtifactStore.get(final_handle)
    assert isinstance(final_artifact, StepArtifact)

    # 1. Ops are scaled from 100 units to the workpiece width of 20mm.
    #    The line is now from (0,0) to (20,0) in local mm.
    # 2. Placement (translation by 50,60) is applied.
    #    The final line is from (50,60) to (70,60) in world mm.
    expected_end = (70.0, 60.0, 0.0)
    line_cmd = next(
        c for c in final_artifact.ops if isinstance(c, LineToCommand)
    )
    assert line_cmd.end == pytest.approx(expected_end)

    ArtifactStore.release(base_handle)
    ArtifactStore.release(final_handle)


def test_step_assembler_handles_texture_data(machine):
    """
    Tests that texture data is correctly packaged into a TextureInstance
    with the correct final transformation matrix.
    """
    doc = Doc()
    layer = doc.active_layer
    wp = WorkPiece(name="wp1")
    wp_transform = (
        Matrix.translation(50, 60) @ Matrix.rotation(90) @ Matrix.scale(20, 10)
    )
    wp.matrix = wp_transform
    layer.add_workpiece(wp)

    # This texture chunk covers the whole workpiece.
    texture = TextureData(
        power_texture_data=np.array([[255]], dtype=np.uint8),
        dimensions_mm=(20, 10),
        position_mm=(0, 0),
    )
    base_artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        texture_data=texture,
    )
    base_handle = ArtifactStore.put(base_artifact)
    assembly_info = [
        {
            "artifact_handle_dict": base_handle.to_dict(),
            "world_transform_list": wp.get_world_transform().to_list(),
            "workpiece_dict": wp.to_dict(),
        }
    ]
    mock_proxy = MagicMock()

    result = make_step_artifact_in_subprocess(
        proxy=mock_proxy,
        workpiece_assembly_info=assembly_info,
        step_uid="step1",
        generation_id=1,
        per_step_transformers_dicts=[],
    )
    assert result is not None
    final_handle_dict, _ = result
    final_handle = create_handle_from_dict(final_handle_dict)
    final_artifact = ArtifactStore.get(final_handle)

    assert isinstance(final_artifact, StepArtifact)
    assert len(final_artifact.texture_instances) == 1
    instance = final_artifact.texture_instances[0]

    # Re-calculate the expected transform to verify the assembler's logic
    expected_transform_matrix = wp_transform

    np.testing.assert_allclose(
        instance.world_transform,
        expected_transform_matrix.to_4x4_numpy(),
        atol=1e-6,
    )

    # The texture data itself should be passed through unmodified
    np.testing.assert_array_equal(
        instance.texture_data.power_texture_data,
        texture.power_texture_data,
    )

    ArtifactStore.release(base_handle)
    ArtifactStore.release(final_handle)
