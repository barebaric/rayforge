import pytest
import json
from unittest.mock import MagicMock
from pathlib import Path
from dataclasses import asdict

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand, MoveToCommand
from rayforge.core.import_source import ImportSource
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.transformer.multipass import MultiPassTransformer
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact.base import Artifact
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.handle import ArtifactHandle
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.core.matrix import Matrix
from rayforge.pipeline.jobrunner import (
    run_job_assembly_in_subprocess,
    JobDescription,
    WorkItemInstruction,
)
from rayforge import config


@pytest.fixture
def machine():
    m = Machine()
    m.dimensions = (200, 150)
    m.y_axis_down = True
    m.add_head(Laser())
    return m


def test_jobrunner_assembles_correctly(machine):
    """
    Test that the jobrunner subprocess correctly scales, transforms, and
    assembles ops in a synchronous unit test.
    """
    # --- Arrange ---
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    # Temporarily set a global config machine for the step factory
    config.config = MagicMock()
    config.config.machine = machine
    step = create_contour_step()
    config.config = None  # Clean up global

    multi_pass = MultiPassTransformer(passes=2)
    step.per_step_transformers_dicts = [multi_pass.to_dict()]
    layer.workflow.add_step(step)

    source = ImportSource(
        Path("wp1.svg"),
        b'<svg width="10" height="10" />',
        renderer=SVG_RENDERER,
    )
    doc.add_import_source(source)

    wp = WorkPiece(name="wp1.svg")
    wp.matrix = (
        Matrix.translation(50, 60) @ Matrix.rotation(90) @ Matrix.scale(40, 30)
    )
    wp.import_source_uid = source.uid
    layer.add_workpiece(wp)

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(10, 0)
    base_artifact = Artifact(
        ops=base_ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(10, 10),
    )
    base_handle = ArtifactStore.put(base_artifact)

    instruction = WorkItemInstruction(
        artifact_handle_dict=base_handle.to_dict(),
        world_transform_list=wp.get_world_transform().to_list(),
        workpiece_dict=wp.to_dict(),
    )

    job_desc = JobDescription(
        work_items_by_step={step.uid: [instruction]},
        per_step_transformers_by_step={
            step.uid: step.per_step_transformers_dicts
        },
        machine_dict=machine.to_dict(),
        doc_dict=doc.to_dict(),
    )

    mock_proxy = MagicMock()

    # --- Act ---
    _time, final_handle_dict = run_job_assembly_in_subprocess(
        mock_proxy, asdict(job_desc)
    )

    # --- Assert ---
    assert final_handle_dict is not None, (
        "final_handle_dict should not be None"
    )
    final_handle = ArtifactHandle.from_dict(final_handle_dict)
    final_artifact = ArtifactStore.get(final_handle)
    final_ops = final_artifact.ops
    final_cmds = list(final_ops)

    # We expect 2 passes, so 2 line commands.
    assert len([c for c in final_cmds if c.is_cutting_command()]) == 2

    move_cmds = [c for c in final_cmds if isinstance(c, MoveToCommand)]
    line_cmds = [c for c in final_cmds if isinstance(c, LineToCommand)]

    # Expected coordinates calculated in the original test:
    # (50, 90) and (50, 50)
    assert move_cmds[0].end == pytest.approx((50.0, 90.0, 0.0))
    assert line_cmds[0].end == pytest.approx((50.0, 50.0, 0.0))
    # Verify second pass is identical
    assert move_cmds[1].end == move_cmds[0].end
    assert line_cmds[1].end == line_cmds[0].end

    # Assert comprehensive artifact content
    assert final_artifact.artifact_type == "final_job"
    assert final_artifact.vertex_data is not None
    assert final_artifact.gcode_bytes is not None
    assert final_artifact.op_map_bytes is not None

    gcode_str = final_artifact.gcode_bytes.tobytes().decode("utf-8")
    op_map_str = final_artifact.op_map_bytes.tobytes().decode("utf-8")
    op_map = json.loads(op_map_str)

    assert "G1" in gcode_str  # Check for a linear move command
    assert isinstance(op_map, dict)
    assert len(op_map) > 0

    # --- Cleanup ---
    ArtifactStore.release(base_handle)
    ArtifactStore.release(final_handle)
