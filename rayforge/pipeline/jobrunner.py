import logging
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from ..machine.models.machine import Machine
from ..shared.tasker.proxy import ExecutionContextProxy
from ..core.ops import Ops, ScanLinePowerCommand
from ..core.workpiece import WorkPiece
from ..core.doc import Doc
from ..pipeline.transformer import OpsTransformer, transformer_by_name
from ..pipeline.encoder.gcode import GcodeEncoder
from .artifact.store import ArtifactStore
from .artifact.handle import ArtifactHandle
from .artifact.base import Artifact
from .coord import CoordinateSystem


logger = logging.getLogger(__name__)


@dataclass
class WorkItemInstruction:
    """A serializable instruction for processing one workpiece in a job."""

    artifact_handle_dict: Dict[str, Any]
    world_transform_list: List[List[float]]
    workpiece_dict: Dict[str, Any]


@dataclass
class JobDescription:
    """A complete, serializable description of a job for the subprocess."""

    work_items_by_step: Dict[str, List[WorkItemInstruction]]
    per_step_transformers_by_step: Dict[str, List[Dict[str, Any]]]
    machine_dict: Dict[str, Any]
    doc_dict: Dict[str, Any]


def _instantiate_transformers_from_step_dict(
    transformer_dicts: List[Dict[str, Any]],
) -> List[OpsTransformer]:
    """Helper to create transformer instances from a step's config dict."""
    transformers: List[OpsTransformer] = []
    for t_dict in transformer_dicts:
        if not t_dict.get("enabled", True):
            continue
        cls_name = t_dict.get("name")
        if cls_name and cls_name in transformer_by_name:
            cls = transformer_by_name[cls_name]
            try:
                transformers.append(cls.from_dict(t_dict))
            except Exception as e:
                logger.error(
                    f"Failed to instantiate transformer '{cls_name}': {e}",
                    exc_info=True,
                )
    return transformers


def _transform_and_clip_workpiece_ops(
    ops: Ops,
    workpiece_world_transform_list: List[List[float]],
    workpiece_name: str,
    machine: Machine,
    clip_rect: tuple[float, float, float, float],
) -> Ops:
    """
    Applies workpiece-specific transforms using its world matrix,
    converts to machine coordinates, and clips the result.
    """
    from ..core.matrix import Matrix

    pre_transform_scanlines = sum(
        1 for cmd in ops.commands if isinstance(cmd, ScanLinePowerCommand)
    )
    logger.debug(
        f"JobRunner: Pre-transform/clip for '{workpiece_name}': "
        f"{pre_transform_scanlines} ScanLinePowerCommands."
    )

    world_matrix = Matrix.from_list(workpiece_world_transform_list)
    (x, y, angle, _, _, _) = world_matrix.decompose()

    transform_4x4 = np.identity(4)
    rad = np.radians(angle)
    c, s = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([[c, -s], [s, c]])
    transform_4x4[0:2, 0:2] = rotation_matrix
    transform_4x4[0:2, 3] = [x, y]

    final_transform = transform_4x4
    if machine.y_axis_down:
        machine_height = machine.dimensions[1]
        y_down_mat = np.identity(4)
        y_down_mat[1, 1] = -1.0
        y_down_mat[1, 3] = machine_height
        final_transform = y_down_mat @ transform_4x4

    ops.transform(final_transform)
    clipped_ops = ops.clip(clip_rect)

    post_clip_scanlines = sum(
        1
        for cmd in clipped_ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    logger.debug(
        f"JobRunner: Post-transform/clip for '{workpiece_name}': "
        f"{post_clip_scanlines} ScanLinePowerCommands."
    )
    return clipped_ops


def run_job_assembly_in_subprocess(
    proxy: ExecutionContextProxy, job_description_dict: Dict[str, Any]
) -> Tuple[float, str, Optional[Dict[str, Any]]]:
    """
    The main entry point for assembling, post-processing, and encoding a
    full job in a background process.
    Returns the final time, G-code path, and a handle to the final Ops
    artifact.
    """
    # When deserialized, the dataclass becomes a dict.
    job_desc_dict = job_description_dict
    machine = Machine.from_dict(job_desc_dict["machine_dict"], is_inert=True)
    doc = Doc.from_dict(job_desc_dict["doc_dict"])

    proxy.set_message(_("Assembling final job..."))
    final_ops = Ops()
    final_ops.job_start()
    machine_width, machine_height = machine.dimensions
    clip_rect = 0, 0, machine_width, machine_height

    total_items = sum(
        len(items) for items in job_desc_dict["work_items_by_step"].values()
    )
    processed_items = 0

    for layer in doc.layers:
        if not layer.workflow:
            continue
        final_ops.layer_start(layer_uid=layer.uid)

        for step in layer.workflow.steps:
            step_uid = step.uid
            if step_uid not in job_desc_dict["work_items_by_step"]:
                continue

            step_combined_ops = Ops()
            work_items = job_desc_dict["work_items_by_step"][step_uid]

            for item_dict in work_items:
                processed_items += 1
                proxy.set_progress(
                    processed_items / total_items if total_items > 0 else 0
                )
                workpiece = WorkPiece.from_dict(item_dict["workpiece_dict"])
                proxy.set_message(
                    _("Processing '{workpiece}' in '{step}'").format(
                        workpiece=workpiece.name, step=step.name
                    )
                )

                handle = ArtifactHandle.from_dict(
                    item_dict["artifact_handle_dict"]
                )
                artifact = ArtifactStore.get(handle)
                if not artifact or artifact.ops.is_empty():
                    continue

                workpiece_ops = artifact.ops.copy()

                # Scale the ops from their source size to the workpiece's size
                if artifact.is_scalable and artifact.source_dimensions:
                    target_w, target_h = workpiece.size
                    source_w, source_h = artifact.source_dimensions
                    if source_w > 1e-9 and source_h > 1e-9:
                        scale_x = target_w / source_w
                        scale_y = target_h / source_h
                        workpiece_ops.scale(scale_x, scale_y)

                ops_with_markers = Ops()
                ops_with_markers.workpiece_start(workpiece_uid=workpiece.uid)
                ops_with_markers.extend(workpiece_ops)
                ops_with_markers.workpiece_end(workpiece_uid=workpiece.uid)

                clipped_ops = _transform_and_clip_workpiece_ops(
                    ops_with_markers,
                    item_dict["world_transform_list"],
                    workpiece.name,
                    machine,
                    clip_rect,
                )
                step_combined_ops.extend(clipped_ops)

            transformer_dicts = job_desc_dict[
                "per_step_transformers_by_step"
            ].get(step_uid, [])
            per_step_transformers = _instantiate_transformers_from_step_dict(
                transformer_dicts
            )
            for transformer in per_step_transformers:
                proxy.set_message(
                    _("Applying '{transformer}' to '{step}'").format(
                        transformer=transformer.label, step=step.name
                    )
                )
                transformer.run(step_combined_ops, context=proxy)

            final_ops.extend(step_combined_ops)

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
    gcode_str, op_to_line_map = encoder.encode(final_ops, machine, doc)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".gcode", encoding="utf-8"
    ) as f:
        f.write(gcode_str)
        gcode_file_path = f.name

    proxy.set_message(_("Storing final job artifact..."))
    final_artifact = Artifact(
        ops=final_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
    )
    final_handle = ArtifactStore.put(final_artifact)

    proxy.set_progress(1.0)
    proxy.set_message(_("Job finalization complete"))
    return final_time, gcode_file_path, final_handle.to_dict()
