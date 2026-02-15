import pytest
import json
import numpy as np

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.ops import Ops, LineToCommand
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.artifact import StepOpsArtifact, JobArtifact
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.stage.job_compute import (
    compute_job_artifact,
    _assemble_final_ops,
    _calculate_time_estimate,
    _calculate_distance,
    _encode_gcode_and_opmap,
    _encode_vertex_data,
)


@pytest.fixture
def machine(context_initializer):
    m = Machine(context_initializer)
    m.dimensions = (200, 150)
    m.add_head(Laser())
    context_initializer.config.set_machine(m)
    return m


def test_job_compute_assembles_step_artifacts_correctly(
    context_initializer, machine, mock_progress_context
):
    """
    Test that compute_job_artifact correctly assembles StepOpsArtifacts
    into a JobArtifact.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    final_ops_for_step = Ops()
    final_ops_for_step.move_to(50, 90)
    final_ops_for_step.line_to(50, 50)
    final_ops_for_step.move_to(50, 90)
    final_ops_for_step.line_to(50, 50)

    step_artifact = StepOpsArtifact(ops=final_ops_for_step)
    step_artifacts_by_uid = {step.uid: step_artifact}

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
        mock_progress_context,
    )

    assert isinstance(result, JobArtifact)

    final_ops = result.ops
    line_cmds = [c for c in final_ops if isinstance(c, LineToCommand)]

    assert len(line_cmds) == 2
    assert line_cmds[0].end == pytest.approx((50.0, 50.0, 0.0))
    assert line_cmds[1].end == pytest.approx((50.0, 50.0, 0.0))

    assert result.vertex_data is not None
    assert result.machine_code_bytes is not None
    assert result.op_map_bytes is not None

    gcode_str = result.machine_code_bytes.tobytes().decode("utf-8")
    op_map_str = result.op_map_bytes.tobytes().decode("utf-8")
    op_map = json.loads(op_map_str)

    assert "G1" in gcode_str
    assert isinstance(op_map, dict) and len(op_map) > 0

    assert len(mock_progress_context.progress_calls) > 0
    assert mock_progress_context.progress_calls[0][0] == 0.0
    assert mock_progress_context.progress_calls[-1][0] == 1.0


def test_job_compute_handles_empty_ops(context_initializer, machine):
    """
    Test that compute_job_artifact handles empty ops gracefully.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    empty_ops = Ops()
    step_artifact = StepOpsArtifact(ops=empty_ops)
    step_artifacts_by_uid = {step.uid: step_artifact}

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
    )

    assert isinstance(result, JobArtifact)
    assert result.machine_code_bytes is not None


def test_job_compute_without_progress_callback(
    context_initializer, machine, mock_progress_context
):
    """
    Test that compute_job_artifact works without a progress callback.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    final_ops_for_step = Ops()
    final_ops_for_step.move_to(50, 90)
    final_ops_for_step.line_to(50, 50)

    step_artifact = StepOpsArtifact(ops=final_ops_for_step)
    step_artifacts_by_uid = {step.uid: step_artifact}

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
        mock_progress_context,
    )

    assert isinstance(result, JobArtifact)
    assert result.machine_code_bytes is not None


def test_job_compute_multiple_steps(context_initializer, machine):
    """
    Test that compute_job_artifact correctly handles multiple steps.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step1 = create_contour_step(context_initializer)
    step1.name = "Step 1"
    layer.workflow.add_step(step1)

    step2 = create_contour_step(context_initializer)
    step2.name = "Step 2"
    layer.workflow.add_step(step2)

    ops1 = Ops()
    ops1.move_to(10, 10)
    ops1.line_to(20, 20)

    ops2 = Ops()
    ops2.move_to(30, 30)
    ops2.line_to(40, 40)

    step_artifacts_by_uid = {
        step1.uid: StepOpsArtifact(ops=ops1),
        step2.uid: StepOpsArtifact(ops=ops2),
    }

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
    )

    assert isinstance(result, JobArtifact)
    line_cmds = [c for c in result.ops if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 2


def test_job_compute_empty_step_artifacts(context_initializer, machine):
    """
    Test that compute_job_artifact handles empty step artifacts.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    step_artifacts_by_uid = {}

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
    )

    assert isinstance(result, JobArtifact)
    assert result.machine_code_bytes is not None


def test_job_compute_multiple_layers(context_initializer, machine):
    """
    Test that compute_job_artifact correctly handles multiple layers.
    """
    doc = Doc()
    layer1 = doc.active_layer
    assert layer1.workflow is not None

    step1 = create_contour_step(context_initializer)
    step1.name = "Step 1"
    layer1.workflow.add_step(step1)

    layer2 = Layer("Layer 2")
    doc.add_layer(layer2)

    step2 = create_contour_step(context_initializer)
    step2.name = "Step 2"
    if layer2.workflow:
        layer2.workflow.add_step(step2)

    ops1 = Ops()
    ops1.move_to(10, 10)
    ops1.line_to(20, 20)

    ops2 = Ops()
    ops2.move_to(30, 30)
    ops2.line_to(40, 40)

    step_artifacts_by_uid = {
        step1.uid: StepOpsArtifact(ops=ops1),
        step2.uid: StepOpsArtifact(ops=ops2),
    }

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
    )

    assert isinstance(result, JobArtifact)
    line_cmds = [c for c in result.ops if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 2


def test_job_compute_layer_without_workflow(context_initializer, machine):
    """
    Test that compute_job_artifact handles layers without workflow.
    """
    doc = Doc()
    layer1 = doc.active_layer

    step = create_contour_step(context_initializer)
    if layer1.workflow:
        layer1.workflow.add_step(step)

    layer2 = Layer("Layer 2")
    doc.add_layer(layer2)

    ops = Ops()
    ops.move_to(10, 10)
    ops.set_power(1.0)
    ops.line_to(20, 20)

    step_artifacts_by_uid = {step.uid: StepOpsArtifact(ops=ops)}

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
    )

    assert isinstance(result, JobArtifact)
    line_cmds = [c for c in result.ops if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 1


def test_job_compute_missing_step_artifact(context_initializer, machine):
    """
    Test that compute_job_artifact handles missing step artifacts.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step1 = create_contour_step(context_initializer)
    step1.name = "Step 1"
    layer.workflow.add_step(step1)

    step2 = create_contour_step(context_initializer)
    step2.name = "Step 2"
    layer.workflow.add_step(step2)

    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    step_artifacts_by_uid = {step1.uid: StepOpsArtifact(ops=ops)}

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
    )

    assert isinstance(result, JobArtifact)
    line_cmds = [c for c in result.ops if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 1


def test_job_compute_vertex_data_generation(context_initializer, machine):
    """
    Test that compute_job_artifact generates vertex data.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    ops = Ops()
    ops.set_power(1.0)
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    step_artifact = StepOpsArtifact(ops=ops)
    step_artifacts_by_uid = {step.uid: step_artifact}

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
    )

    assert isinstance(result, JobArtifact)
    assert result.vertex_data is not None
    assert result.vertex_data.powered_vertices.size > 0


def test_job_compute_time_and_distance(context_initializer, machine):
    """
    Test that compute_job_artifact calculates time and distance.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    step_artifact = StepOpsArtifact(ops=ops)
    step_artifacts_by_uid = {step.uid: step_artifact}

    result = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
    )

    assert isinstance(result, JobArtifact)
    assert result.distance > 0
    if result.time_estimate is not None:
        assert result.time_estimate > 0


def test_assemble_final_ops_single_step(
    context_initializer, mock_progress_context
):
    """
    Test _assemble_final_ops with a single step.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    step_artifacts_by_uid = {step.uid: StepOpsArtifact(ops=ops)}

    result = _assemble_final_ops(
        doc, step_artifacts_by_uid, mock_progress_context
    )

    assert isinstance(result, Ops)
    line_cmds = [c for c in result if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 1


def test_assemble_final_ops_multiple_steps(context_initializer):
    """
    Test _assemble_final_ops with multiple steps.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step1 = create_contour_step(context_initializer)
    step1.name = "Step 1"
    layer.workflow.add_step(step1)

    step2 = create_contour_step(context_initializer)
    step2.name = "Step 2"
    layer.workflow.add_step(step2)

    ops1 = Ops()
    ops1.move_to(10, 10)
    ops1.line_to(20, 20)

    ops2 = Ops()
    ops2.move_to(30, 30)
    ops2.line_to(40, 40)

    step_artifacts_by_uid = {
        step1.uid: StepOpsArtifact(ops=ops1),
        step2.uid: StepOpsArtifact(ops=ops2),
    }

    result = _assemble_final_ops(doc, step_artifacts_by_uid)

    assert isinstance(result, Ops)
    line_cmds = [c for c in result if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 2


def test_assemble_final_ops_multiple_layers(context_initializer):
    """
    Test _assemble_final_ops with multiple layers.
    """
    doc = Doc()
    layer1 = doc.active_layer
    assert layer1.workflow is not None

    step1 = create_contour_step(context_initializer)
    step1.name = "Step 1"
    layer1.workflow.add_step(step1)

    layer2 = Layer("Layer 2")
    doc.add_layer(layer2)

    step2 = create_contour_step(context_initializer)
    step2.name = "Step 2"
    if layer2.workflow:
        layer2.workflow.add_step(step2)

    ops1 = Ops()
    ops1.move_to(10, 10)
    ops1.line_to(20, 20)

    ops2 = Ops()
    ops2.move_to(30, 30)
    ops2.line_to(40, 40)

    step_artifacts_by_uid = {
        step1.uid: StepOpsArtifact(ops=ops1),
        step2.uid: StepOpsArtifact(ops=ops2),
    }

    result = _assemble_final_ops(doc, step_artifacts_by_uid)

    assert isinstance(result, Ops)
    line_cmds = [c for c in result if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 2


def test_assemble_final_ops_empty_artifacts(context_initializer):
    """
    Test _assemble_final_ops with empty step artifacts.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    step_artifacts_by_uid = {}

    result = _assemble_final_ops(doc, step_artifacts_by_uid)

    assert isinstance(result, Ops)


def test_assemble_final_ops_missing_step(context_initializer):
    """
    Test _assemble_final_ops with missing step artifacts.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step1 = create_contour_step(context_initializer)
    step1.name = "Step 1"
    layer.workflow.add_step(step1)

    step2 = create_contour_step(context_initializer)
    step2.name = "Step 2"
    layer.workflow.add_step(step2)

    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    step_artifacts_by_uid = {step1.uid: StepOpsArtifact(ops=ops)}

    result = _assemble_final_ops(doc, step_artifacts_by_uid)

    assert isinstance(result, Ops)
    line_cmds = [c for c in result if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 1


def test_assemble_final_ops_without_progress(context_initializer):
    """
    Test _assemble_final_ops without progress callback.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None

    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)

    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    step_artifacts_by_uid = {step.uid: StepOpsArtifact(ops=ops)}

    result = _assemble_final_ops(doc, step_artifacts_by_uid, None)

    assert isinstance(result, Ops)
    line_cmds = [c for c in result if isinstance(c, LineToCommand)]
    assert len(line_cmds) == 1


def test_calculate_time_estimate(machine, mock_progress_context):
    """
    Test _calculate_time_estimate.
    """
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    result = _calculate_time_estimate(ops, machine, mock_progress_context)

    assert result is not None
    assert result > 0


def test_calculate_time_estimate_empty_ops(machine):
    """
    Test _calculate_time_estimate with empty ops.
    """
    ops = Ops()

    result = _calculate_time_estimate(ops, machine)

    assert result == 0.0


def test_calculate_time_estimate_without_progress(machine):
    """
    Test _calculate_time_estimate without progress callback.
    """
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    result = _calculate_time_estimate(ops, machine, None)

    assert result is not None
    assert result > 0


def test_calculate_distance():
    """
    Test _calculate_distance.
    """
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    result = _calculate_distance(ops)

    assert result > 0


def test_calculate_distance_empty_ops():
    """
    Test _calculate_distance with empty ops.
    """
    ops = Ops()

    result = _calculate_distance(ops)

    assert result == 0


def test_encode_gcode_and_opmap(
    context_initializer, machine, mock_progress_context
):
    """
    Test _encode_gcode_and_opmap.
    """
    doc = Doc()
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    machine_code, op_map = _encode_gcode_and_opmap(
        ops, doc, machine, mock_progress_context
    )

    assert machine_code is not None
    assert op_map is not None
    assert isinstance(machine_code, type(np.array([])))
    assert isinstance(op_map, type(np.array([])))

    gcode_str = machine_code.tobytes().decode("utf-8")
    assert "G1" in gcode_str


def test_encode_gcode_and_opmap_without_progress(context_initializer, machine):
    """
    Test _encode_gcode_and_opmap without progress callback.
    """
    doc = Doc()
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    machine_code, op_map = _encode_gcode_and_opmap(ops, doc, machine, None)

    assert machine_code is not None
    assert op_map is not None


def test_encode_vertex_data(mock_progress_context):
    """
    Test _encode_vertex_data.
    """
    ops = Ops()
    ops.set_power(1.0)
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    result = _encode_vertex_data(ops, mock_progress_context)

    assert result is not None
    assert result.powered_vertices.size > 0


def test_encode_vertex_data_empty_ops(mock_progress_context):
    """
    Test _encode_vertex_data with empty ops.
    """
    ops = Ops()

    result = _encode_vertex_data(ops, mock_progress_context)

    assert result is not None


def test_encode_vertex_data_without_progress():
    """
    Test _encode_vertex_data without progress callback.
    """
    ops = Ops()
    ops.set_power(1.0)
    ops.move_to(10, 10)
    ops.line_to(20, 20)

    result = _encode_vertex_data(ops, None)

    assert result is not None
    assert result.powered_vertices.size > 0
