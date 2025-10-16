import logging
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np

from ...machine.models.machine import Machine
from ...shared.tasker.proxy import ExecutionContextProxy
from ...core.ops import Ops
from ...core.doc import Doc
from ..encoder.gcode import GcodeEncoder
from ..encoder.vertexencoder import VertexEncoder
from ..artifact import (
    ArtifactStore,
    JobArtifact,
    create_handle_from_dict,
    StepArtifact,
)
from ..coord import CoordinateSystem


logger = logging.getLogger(__name__)


@dataclass
class JobDescription:
    """A complete, serializable description of a job for the subprocess."""

    step_artifact_handles_by_uid: Dict[str, Dict[str, Any]]
    machine_dict: Dict[str, Any]
    doc_dict: Dict[str, Any]


def make_job_artifact_in_subprocess(
    proxy: ExecutionContextProxy, job_description_dict: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
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
            artifact = ArtifactStore.get(handle)

            if (
                isinstance(artifact, StepArtifact)
                and not artifact.ops.is_empty()
            ):
                final_ops.extend(artifact.ops)

        final_ops.layer_end(layer_uid=layer.uid)

    final_ops.job_end()

    proxy.set_message(_("Calculating final time estimate..."))
    final_time = final_ops.estimate_time(
        default_cut_speed=machine.max_cut_speed,
        default_travel_speed=machine.max_travel_speed,
        acceleration=machine.acceleration,
    )

    proxy.set_message(_("Generating G-code..."))
    encoder = GcodeEncoder.for_machine(machine)
    gcode_str, op_map_obj = encoder.encode(final_ops, machine, doc)

    # Encode G-code and map to byte arrays for storage in the artifact
    gcode_bytes = np.frombuffer(gcode_str.encode("utf-8"), dtype=np.uint8)
    op_map_str = json.dumps(asdict(op_map_obj))
    op_map_bytes = np.frombuffer(op_map_str.encode("utf-8"), dtype=np.uint8)

    # Generate vertex data for UI preview/simulation
    proxy.set_message(_("Encoding paths for preview..."))
    vertex_encoder = VertexEncoder()
    vertex_data = vertex_encoder.encode(final_ops)

    proxy.set_message(_("Storing final job artifact..."))
    final_artifact = JobArtifact(
        ops=final_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        vertex_data=vertex_data,
        gcode_bytes=gcode_bytes,
        op_map_bytes=op_map_bytes,
        time_estimate=final_time,
    )
    final_handle = ArtifactStore.put(final_artifact)

    proxy.set_progress(1.0)
    proxy.set_message(_("Job finalization complete"))
    return final_handle.to_dict()
