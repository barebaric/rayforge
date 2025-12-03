from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Any
from ...context import get_context
from ...core.ops import Ops
from ...core.doc import Doc
from ...machine.models.machine import Machine
from ...shared.tasker.proxy import ExecutionContextProxy
from ..artifact import (
    JobArtifact,
    create_handle_from_dict,
    StepOpsArtifact,
)
from ..encoder.vertexencoder import VertexEncoder


logger = logging.getLogger(__name__)


@dataclass
class JobDescription:
    """A complete, serializable description of a job for the subprocess."""

    step_artifact_handles_by_uid: Dict[str, Dict[str, Any]]
    machine_dict: Dict[str, Any]
    doc_dict: Dict[str, Any]


def make_job_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    job_description_dict: Dict[str, Any],
    creator_tag: str,
) -> None:
    """
    The main entry point for assembling, post-processing, and encoding a
    full job in a background process.

    This function consumes pre-computed StepArtifacts, combines their Ops,
    and encodes the result into a final JobArtifact containing G-code and
    preview data.
    """
    job_desc = JobDescription(**job_description_dict)
    machine = Machine.from_dict(job_desc.machine_dict, is_inert=True)
    doc = Doc.from_dict(job_desc.doc_dict)
    handles_by_uid = job_desc.step_artifact_handles_by_uid
    artifact_store = get_context().artifact_store

    proxy.set_message(_("Assembling final job..."))
    final_ops = Ops()
    final_ops.job_start()

    total_steps = len(handles_by_uid)
    processed_steps = 0

    for layer in doc.layers:
        if not layer.workflow:
            continue
        final_ops.layer_start(layer_uid=layer.uid)

        for step in layer.workflow.steps:
            if step.uid not in handles_by_uid:
                continue

            processed_steps += 1
            proxy.set_progress(
                processed_steps / total_steps if total_steps > 0 else 0
            )
            proxy.set_message(
                _("Processing final ops for '{step}'").format(step=step.name)
            )

            handle_dict = handles_by_uid[step.uid]
            handle = create_handle_from_dict(handle_dict)
            artifact = artifact_store.get(handle)

            if (
                isinstance(artifact, StepOpsArtifact)
                and not artifact.ops.is_empty()
            ):
                final_ops.extend(artifact.ops)

        final_ops.layer_end(layer_uid=layer.uid)

    final_ops.job_end()

    # If the final ops are empty, still proceed to encoding to generate a file
    # with the necessary preamble and postscript.
    if final_ops.is_empty():
        logger.info(
            "Final ops are empty. "
            "Generating G-code with preamble/postscript only."
        )

    proxy.set_message(_("Calculating final time and distance estimates..."))
    final_time = final_ops.estimate_time(
        default_cut_speed=machine.max_cut_speed,
        default_travel_speed=machine.max_travel_speed,
        acceleration=machine.acceleration,
    )
    final_distance = final_ops.distance()

    proxy.set_message(_("Generating G-code..."))
    machine_code_bytes, op_map_bytes = machine.encode_ops(final_ops, doc)

    # Generate vertex data for UI preview/simulation
    # NOTE: The preview uses the original Y-Up final_ops. The 3D view camera
    # handles the visual orientation for Y-down machines.
    proxy.set_message(_("Encoding paths for preview..."))
    vertex_encoder = VertexEncoder()
    vertex_data = vertex_encoder.encode(final_ops)

    proxy.set_message(_("Storing final job artifact..."))
    final_artifact = JobArtifact(
        ops=final_ops,
        distance=final_distance,
        vertex_data=vertex_data,
        machine_code_bytes=machine_code_bytes,
        op_map_bytes=op_map_bytes,
        time_estimate=final_time,
    )
    final_handle = artifact_store.put(final_artifact, creator_tag=creator_tag)

    proxy.send_event(
        "artifact_created", {"handle_dict": final_handle.to_dict()}
    )

    proxy.set_progress(1.0)
    proxy.set_message(_("Job finalization complete"))
    return
