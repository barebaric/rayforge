import pytest
from unittest.mock import patch

from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops, LineToCommand
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    StepOpsArtifact,
    create_handle_from_dict,
)
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)
from rayforge.pipeline.transformer.registry import transformer_registry


@pytest.fixture
def machine(context_initializer):
    m = Machine(context_initializer)
    m.max_cut_speed = 5000
    m.max_travel_speed = 10000
    m.acceleration = 1000
    m.add_head(Laser())
    return m


def test_step_runner_correctly_scales_and_places_ops(
    machine, adopting_mock_proxy
):
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
        generation_id=1,
    )
    base_handle = get_context().artifact_store.put(base_artifact)
    assembly_info = [
        {
            "artifact_handle_dict": base_handle.to_dict(),
            "world_transform_list": wp.get_world_transform().to_list(),
            "workpiece_dict": wp.in_world().to_dict(),
        }
    ]

    result = make_step_artifact_in_subprocess(
        proxy=adopting_mock_proxy,
        artifact_store=get_context().artifact_store,
        workpiece_assembly_info=assembly_info,
        step_uid="step1",
        generation_id=1,
        per_step_transformers_dicts=[],
        cut_speed=machine.max_cut_speed,
        travel_speed=machine.max_travel_speed,
        acceleration=machine.acceleration,
        creator_tag="tstp",
    )
    assert result == 1

    calls = adopting_mock_proxy.send_event_and_wait.call_args_list
    assert len(calls) == 1

    ops_call = calls[0]
    assert ops_call[0][0] == "ops_artifact_ready"
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
    get_context().artifact_store.release(ops_handle)


def test_step_runner_instantiates_transformers_from_dict(
    machine, adopting_mock_proxy
):
    """
    Tests that the runner correctly instantiates transformers from
    dictionaries before calling compute.
    """
    doc = Doc()
    layer = doc.active_layer
    wp = WorkPiece(name="wp1")
    wp.set_size(20, 10)
    wp.pos = 50, 60
    wp.angle = 90
    layer.add_workpiece(wp)

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(100, 0)
    base_artifact = WorkPieceArtifact(
        ops=base_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(20, 10),
        generation_id=1,
    )
    base_handle = get_context().artifact_store.put(base_artifact)
    assembly_info = [
        {
            "artifact_handle_dict": base_handle.to_dict(),
            "world_transform_list": wp.get_world_transform().to_list(),
            "workpiece_dict": wp.in_world().to_dict(),
        }
    ]

    transformer_dict = {
        "name": "Optimize",
        "enabled": True,
    }

    with patch(
        "rayforge.pipeline.stage.step_compute.compute_step_artifacts"
    ) as mock_compute:
        mock_compute.return_value = StepOpsArtifact(ops=Ops(), generation_id=1)

        result = make_step_artifact_in_subprocess(
            proxy=adopting_mock_proxy,
            artifact_store=get_context().artifact_store,
            workpiece_assembly_info=assembly_info,
            step_uid="step1",
            generation_id=1,
            per_step_transformers_dicts=[transformer_dict],
            cut_speed=machine.max_cut_speed,
            travel_speed=machine.max_travel_speed,
            acceleration=machine.acceleration,
            creator_tag="tstp",
        )

        assert result == 1
        assert mock_compute.called

        call_args = mock_compute.call_args
        transformers = call_args[1]["transformers"]
        assert len(transformers) == 1
        Optimize = transformer_registry.get("Optimize")
        assert Optimize is not None
        assert isinstance(transformers[0], Optimize)

    get_context().artifact_store.release(base_handle)
