from __future__ import annotations
import logging
import json
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np

from ...core.doc import Doc
from ...core.ops import Ops
from ...machine.models.machine import Machine
from ...shared.tasker.progress import ProgressContext, set_progress
from ..artifact import JobArtifact, StepOpsArtifact
from ..artifact.base import VertexData
from ..encoder.vertexencoder import VertexEncoder

logger = logging.getLogger(__name__)


def _assemble_final_ops(
    doc: Doc,
    step_artifacts_by_uid: Dict[str, StepOpsArtifact],
    context: Optional[ProgressContext] = None,
) -> Ops:
    """
    Assembles final ops from step artifacts across layers.

    Args:
        doc: The document containing layers and workflow steps.
        step_artifacts_by_uid: Dictionary mapping step UIDs to their
            StepOpsArtifacts.
        context: Optional ProgressContext for progress reporting.

    Returns:
        An Ops object containing the assembled operations.
    """
    set_progress(context, 0.0, _("Assembling final job..."))
    final_ops = Ops()
    final_ops.job_start()

    total_steps = len(step_artifacts_by_uid)
    processed_steps = 0

    for layer in doc.layers:
        if not layer.workflow:
            continue
        final_ops.layer_start(layer_uid=layer.uid)

        for step in layer.workflow.steps:
            if step.uid not in step_artifacts_by_uid:
                continue

            processed_steps += 1
            progress = processed_steps / total_steps if total_steps > 0 else 0
            set_progress(
                context,
                progress,
                _("Processing final ops for '{step}'").format(step=step.name),
            )

            artifact = step_artifacts_by_uid[step.uid]

            if (
                isinstance(artifact, StepOpsArtifact)
                and not artifact.ops.is_empty()
            ):
                final_ops.extend(artifact.ops)

        final_ops.layer_end(layer_uid=layer.uid)

    final_ops.job_end()

    if final_ops.is_empty():
        logger.info(
            "Final ops are empty. "
            "Generating G-code with preamble/postscript only."
        )

    return final_ops


def _calculate_time_estimate(
    final_ops: Ops,
    machine: Machine,
    context: Optional[ProgressContext] = None,
) -> Optional[float]:
    """
    Calculates time estimate from operations.

    Args:
        final_ops: The operations to estimate time for.
        machine: The machine model for parameters.
        context: Optional ProgressContext for progress reporting.

    Returns:
        Estimated time in seconds, or None if not calculable.
    """
    set_progress(
        context, 0.7, _("Calculating final time and distance estimates...")
    )
    return final_ops.estimate_time(
        default_cut_speed=machine.max_cut_speed,
        default_travel_speed=machine.max_travel_speed,
        acceleration=machine.acceleration,
    )


def _calculate_distance(final_ops: Ops) -> float:
    """
    Calculates total distance from operations.

    Args:
        final_ops: The operations to calculate distance for.

    Returns:
        Total distance traveled.
    """
    return final_ops.distance()


def _encode_gcode_and_opmap(
    final_ops: Ops,
    doc: Doc,
    machine: Machine,
    context: Optional[ProgressContext] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encodes operations to G-code and operation map.

    Args:
        final_ops: The operations to encode.
        doc: The document for encoding context.
        machine: The machine model for encoding.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A tuple of (machine_code_bytes, op_map_bytes).
    """
    set_progress(context, 0.8, _("Generating G-code..."))
    gcode_str, op_map_obj = machine.encode_ops(final_ops, doc)

    machine_code_bytes = np.frombuffer(
        gcode_str.encode("utf-8"), dtype=np.uint8
    )
    op_map_str = json.dumps(asdict(op_map_obj))
    op_map_bytes = np.frombuffer(op_map_str.encode("utf-8"), dtype=np.uint8)

    return machine_code_bytes, op_map_bytes


def _encode_vertex_data(
    final_ops: Ops,
    context: Optional[ProgressContext] = None,
) -> VertexData:
    """
    Encodes operations to vertex data for preview.

    Args:
        final_ops: The operations to encode.
        context: Optional ProgressContext for progress reporting.

    Returns:
        VertexData object containing vertex arrays for preview rendering.
    """
    set_progress(context, 0.9, _("Encoding paths for preview..."))
    vertex_encoder = VertexEncoder()
    return vertex_encoder.encode(final_ops)


def compute_job_artifact(
    doc: Doc,
    step_artifacts_by_uid: Dict[str, StepOpsArtifact],
    machine: Machine,
    context: Optional[ProgressContext] = None,
) -> JobArtifact:
    """
    Computes a JobArtifact from a Doc and StepOpsArtifacts.

    This function aggregates pre-computed StepOpsArtifacts, combines their
    Ops, and encodes the result into a final JobArtifact containing G-code
    and preview data.

    Args:
        doc: The document containing layers and workflow steps.
        step_artifacts_by_uid: Dictionary mapping step UIDs to their
            StepOpsArtifacts.
        machine: The machine model for encoding and parameters.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A JobArtifact containing the final G-code, vertex data, and
        metadata.
    """
    final_ops = _assemble_final_ops(doc, step_artifacts_by_uid, context)

    final_time = _calculate_time_estimate(final_ops, machine, context)
    final_distance = _calculate_distance(final_ops)

    machine_code_bytes, op_map_bytes = _encode_gcode_and_opmap(
        final_ops, doc, machine, context
    )

    vertex_data = _encode_vertex_data(final_ops, context)

    set_progress(context, 1.0, _("Job finalization complete"))

    return JobArtifact(
        ops=final_ops,
        distance=final_distance,
        vertex_data=vertex_data,
        machine_code_bytes=machine_code_bytes,
        op_map_bytes=op_map_bytes,
        time_estimate=final_time,
    )
