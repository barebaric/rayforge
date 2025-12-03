import pytest
import json
from unittest.mock import MagicMock
from dataclasses import asdict

from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops, LineToCommand
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.artifact import (
    StepOpsArtifact,
    JobArtifact,
    create_handle_from_dict,
)
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.stage.job_runner import (
    make_job_artifact_in_subprocess,
    JobDescription,
)


@pytest.fixture
def machine(context_initializer):
    m = Machine(context_initializer)
    m.dimensions = (200, 150)
    m.add_head(Laser())
    # This fixture uses the context to create the machine, but the test also
    # needs to configure the active machine on the context.
    context_initializer.config.set_machine(m)
    return m


def test_jobrunner_assembles_step_artifacts_correctly(
    context_initializer, machine
):
    """
    Test that the jobrunner subprocess correctly assembles pre-computed
    StepOpsArtifacts into a final JobArtifact.
    """
    # Arrange
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    # This test simulates that step-level transformers have already run.
    # We create a StepOpsArtifact that contains the *result* of those.
    step = create_contour_step(context_initializer)

    layer.workflow.add_step(step)

    # 1. Define the final Ops as they should exist in the StepOpsArtifact
    # (i.e., after multi-pass and world transforms have been applied).
    # Two passes of a line from (50, 90) to (50, 50).
    final_ops_for_step = Ops()
    final_ops_for_step.move_to(50, 90)
    final_ops_for_step.line_to(50, 50)
    final_ops_for_step.move_to(50, 90)
    final_ops_for_step.line_to(50, 50)

    # 2. Create and store the StepOpsArtifact
    step_artifact = StepOpsArtifact(ops=final_ops_for_step)
    step_handle = get_context().artifact_store.put(step_artifact)

    # 3. Create the JobDescription pointing to the StepOpsArtifact handle
    job_desc = JobDescription(
        step_artifact_handles_by_uid={step.uid: step_handle.to_dict()},
        machine_dict=machine.to_dict(),
        doc_dict=doc.to_dict(),
    )

    mock_proxy = MagicMock()

    # Act
    result = make_job_artifact_in_subprocess(
        mock_proxy, asdict(job_desc), "test_job"
    )

    # Assert
    assert result is None  # Runner returns None now
    mock_proxy.send_event.assert_called_once()
    event_name, event_data = mock_proxy.send_event.call_args[0]
    assert event_name == "artifact_created"

    final_handle_dict = event_data["handle_dict"]
    assert final_handle_dict is not None
    final_handle = create_handle_from_dict(final_handle_dict)
    final_artifact = get_context().artifact_store.get(final_handle)
    assert isinstance(final_artifact, JobArtifact)

    final_ops = final_artifact.ops
    line_cmds = [c for c in final_ops if isinstance(c, LineToCommand)]

    # The jobrunner should just aggregate the ops from the StepOpsArtifact.
    # No further transformation occurs.
    assert len(line_cmds) == 2
    assert line_cmds[0].end == pytest.approx((50.0, 50.0, 0.0))
    assert line_cmds[1].end == pytest.approx((50.0, 50.0, 0.0))

    # Assert comprehensive artifact content
    assert final_artifact.vertex_data is not None
    assert final_artifact.machine_code_bytes is not None
    assert final_artifact.op_map_bytes is not None

    gcode_str = final_artifact.machine_code_bytes.tobytes().decode("utf-8")
    op_map_str = final_artifact.op_map_bytes.tobytes().decode("utf-8")
    op_map = json.loads(op_map_str)

    assert "G1" in gcode_str  # Check for a linear move command
    assert isinstance(op_map, dict) and len(op_map) > 0

    # Cleanup
    get_context().artifact_store.release(step_handle)
    get_context().artifact_store.release(final_handle)
