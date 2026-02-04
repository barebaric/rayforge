import pytest
from unittest.mock import MagicMock
import numpy as np

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    StepRenderArtifact,
    StepOpsArtifact,
    create_handle_from_dict,
)
from rayforge.context import get_context
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.core.matrix import Matrix
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)


@pytest.fixture
def machine(context_initializer):
    m = Machine(context_initializer)
    m.max_cut_speed = 5000
    m.max_travel_speed = 10000
    m.acceleration = 1000
    m.add_head(Laser())
    return m


def test_step_runner_correctly_scales_and_places_ops(machine):
    """
    Tests that ops are correctly scaled from source dimensions to
    workpiece size, and then placed in world coordinates.
    """
    doc = Doc()
    layer = doc.active_layer
    wp = WorkPiece(name="wp1")
    wp.set_size(20, 10)
    # Final placement: translate to (50, 60) and rotate 90 degrees.
    wp.pos = 50, 60
    wp.angle = 90
    layer.add_workpiece(wp)

    # Create an artifact with source dimensions of 100x100 units.
    # This should be scaled to workpiece size of 20x10mm.
    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)
    base_artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(20, 10),
    )
    base_handle = get_context().artifact_store.put(base_artifact)
    assembly_info = [
        {
            "artifact_handle_dict": base_handle.to_dict(),
            "world_transform_list": wp.get_world_transform().to_list(),
            "workpiece_dict": wp.in_world().to_dict(),
        }
    ]
    mock_proxy = MagicMock()

    result = make_step_artifact_in_subprocess(
        proxy=mock_proxy,
        artifact_store=get_context().artifact_store,
        workpiece_assembly_info=assembly_info,
        step_uid="step1",
        generation_id=1,
        per_step_transformers_dicts=[],
        cut_speed=machine.max_cut_speed,
        travel_speed=machine.max_travel_speed,
        acceleration=machine.acceleration,
        creator_tag="test_step",
    )
    assert result == 1

    # Check that both ops and render artifacts were created
    calls = mock_proxy.send_event.call_args_list
    assert len(calls) == 3

    # Find and validate render artifact
    render_call = next(c for c in calls if c[0][0] == "render_artifact_ready")
    render_handle_dict = render_call[0][1]["handle_dict"]
    render_handle = create_handle_from_dict(render_handle_dict)
    render_artifact = get_context().artifact_store.get(render_handle)
    assert isinstance(render_artifact, StepRenderArtifact)

    # Find and validate ops artifact
    ops_call = next(c for c in calls if c[0][0] == "ops_artifact_ready")
    ops_handle_dict = ops_call[0][1]["handle_dict"]
    ops_handle = create_handle_from_dict(ops_handle_dict)
    ops_artifact = get_context().artifact_store.get(ops_handle)
    assert isinstance(ops_artifact, StepOpsArtifact)

    # 1. Ops are scaled from 100 units to workpiece width of 20mm.
    #    The line is now from (0,0) to (20,0) in local mm.
    # 2. Placement (translation by 50,60 and rotation by 90 degrees)
    #    is applied.
    #    The rotation is around the workpiece's center (0.5,0.5 in
    #    normalized coords), which is at (60,65) in world space after
    #    translation.
    #    The line endpoint (20,0) rotates 90 degrees around the origin to
    #    (0,20), then is translated to (65,75) in world mm.
    expected_end = (65.0, 75.0, 0.0)
    line_cmd = next(
        c for c in ops_artifact.ops if isinstance(c, LineToCommand)
    )
    assert line_cmd.end == pytest.approx(expected_end)

    get_context().artifact_store.release(base_handle)
    get_context().artifact_store.release(render_handle)
    get_context().artifact_store.release(ops_handle)


def test_step_runner_handles_texture_data(machine):
    """
    Tests that texture data is correctly packaged into a TextureInstance
    with correct final transformation matrix.
    """
    doc = Doc()
    layer = doc.active_layer
    wp = WorkPiece(name="wp1")
    wp.set_size(20, 10)
    # Final placement: translate to (50, 60) and rotate 90 degrees.
    wp.pos = 50, 60
    wp.angle = 90
    layer.add_workpiece(wp)

    base_artifact = WorkPieceArtifact(
        ops=Ops(),
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        generation_size=(20, 10),
    )
    base_handle = get_context().artifact_store.put(base_artifact)
    assembly_info = [
        {
            "artifact_handle_dict": base_handle.to_dict(),
            "world_transform_list": wp.get_world_transform().to_list(),
            "workpiece_dict": wp.in_world().to_dict(),
        }
    ]
    mock_proxy = MagicMock()

    result = make_step_artifact_in_subprocess(
        proxy=mock_proxy,
        artifact_store=get_context().artifact_store,
        workpiece_assembly_info=assembly_info,
        step_uid="step1",
        generation_id=1,
        per_step_transformers_dicts=[],
        cut_speed=machine.max_cut_speed,
        travel_speed=machine.max_travel_speed,
        acceleration=machine.acceleration,
        creator_tag="test_step",
    )
    assert result == 1

    assert mock_proxy.send_event.call_count == 3
    render_call = next(
        c
        for c in mock_proxy.send_event.call_args_list
        if c[0][0] == "render_artifact_ready"
    )
    render_handle_dict = render_call[0][1]["handle_dict"]
    render_handle = create_handle_from_dict(render_handle_dict)
    render_artifact = get_context().artifact_store.get(render_handle)

    assert isinstance(render_artifact, StepRenderArtifact)
    assert len(render_artifact.texture_instances) == 1
    instance = render_artifact.texture_instances[0]

    # The final transform should be:
    # WorldPlacement @ LocalTranslation @ LocalScale
    # The runner correctly extracts placement (pos/rot) and discards scale
    # from workpiece's full world transform. The test must replicate this.
    full_world_transform = wp.get_world_transform()
    (tx, ty, angle, sx, sy, skew) = full_world_transform.decompose()
    world_placement_matrix = Matrix.compose(
        tx, ty, angle, 1.0, np.copysign(1.0, sy), skew
    )

    local_translation_matrix = Matrix.translation(0, 0)
    local_scale_matrix = Matrix.scale(20, 10)
    expected_transform_matrix = (
        world_placement_matrix @ local_translation_matrix @ local_scale_matrix
    )

    np.testing.assert_allclose(
        instance.world_transform,
        expected_transform_matrix.to_4x4_numpy(),
        atol=1e-6,
    )

    # Texture data dimensions are pixel dimensions (1000x500 at 50ppm)
    assert instance.texture_data.dimensions_mm == (1000.0, 500.0)
    assert instance.texture_data.position_mm == (0, 0)

    get_context().artifact_store.release(base_handle)
    get_context().artifact_store.release(render_handle)
