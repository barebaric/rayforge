"""
Handles the final assembly of machine operations for an entire job.
"""

import asyncio
import logging
import numpy as np
from typing import Optional, List, Dict
from ..machine.models.machine import Machine
from ..shared.tasker.context import ExecutionContext
from ..shared.tasker.manager import CancelledError
from ..core.doc import Doc
from ..core.ops import Ops
from ..pipeline.generator import OpsGenerator
from ..core.step import Step
from ..core.workpiece import WorkPiece
from ..pipeline.transformer import OpsTransformer, transformer_by_name


logger = logging.getLogger(__name__)


def _instantiate_transformers_from_step(step: Step) -> List[OpsTransformer]:
    """Helper to create transformer instances from a step's config."""
    transformers: List[OpsTransformer] = []
    for t_dict in step.post_step_transformers_dicts:
        if not t_dict.get("enabled", True):
            continue
        cls_name = t_dict.get("name")
        if cls_name and cls_name in transformer_by_name:
            cls = transformer_by_name[cls_name]
            try:
                transformers.append(cls.from_dict(t_dict))
            except Exception as e:
                logger.error(
                    f"Failed to instantiate transformer '{cls_name}' "
                    f"for step '{step.name}': {e}",
                    exc_info=True,
                )
    return transformers


def _transform_and_clip_workpiece_ops(
    ops: Ops,
    workpiece: WorkPiece,
    machine: Machine,
    clip_rect: tuple[float, float, float, float],
) -> Ops:
    """
    Applies workpiece-specific transforms using its world matrix,
    converts to machine coordinates, and clips the result.
    """
    # The ops from the generator are already scaled to the workpiece's final
    # size in millimeters, but located at the origin. We only need to apply
    # the rotation and translation components from the world matrix.

    # 1. Decompose the world matrix to isolate its components.
    world_matrix = workpiece.get_world_transform()
    (x, y, angle, sx, sy, _) = world_matrix.decompose()

    # 2. Create a new 4x4 transform matrix containing only rotation and
    # translation.
    # We build it in numpy for the Ops class.
    transform_4x4 = np.identity(4)

    # Apply rotation around the ops' local origin (0,0)
    rad = np.radians(angle)
    c, s = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([[c, -s], [s, c]])
    transform_4x4[0:2, 0:2] = rotation_matrix

    # Apply translation
    transform_4x4[0:2, 3] = [x, y]

    # 3. Combine workpiece world transform with machine coordinate transform.
    # This avoids multiple passes over the ops data.
    final_transform = transform_4x4
    if machine.y_axis_down:
        machine_height = machine.dimensions[1]
        y_down_mat = np.identity(4)
        y_down_mat[1, 1] = -1.0
        y_down_mat[1, 3] = machine_height
        # The machine coordinate transform is applied AFTER the workpiece's
        # world transform.
        final_transform = y_down_mat @ transform_4x4

    # 4. Apply the single, combined transformation matrix.
    ops.transform(final_transform)

    # 5. Clip to machine boundaries
    return ops.clip(clip_rect)


async def generate_job_ops(
    doc: Doc,
    machine: Machine,
    ops_generator: OpsGenerator,
    context: Optional[ExecutionContext] = None,
) -> Ops:
    """
    Assembles all workpiece Ops into a single, final job for a machine.

    This function iterates through all visible step/workpiece pairs in a
    document, groups them by step, and generates the final machine operations.

    For each step, it performs the following:
    1. For each associated workpiece, fetches the pre-generated, cached `Ops`
       from the `OpsGenerator`.
    2. Applies transformations to place these local `Ops` into the machine's
       global coordinate space. This includes rotation, translation, and
       Y-axis flipping for the machine.
    3. Clips the transformed `Ops` to the machine's physical boundaries.
    4. Combines the clipped `Ops` from all workpieces of the step into a
       single `Ops` object.
    5. Applies any configured post-step transformers (e.g., multi-pass) to
       this combined `Ops` object.

    The final results from all steps are aggregated into a single job `Ops`.

    Args:
        doc: The document containing all layers, workflows, and workpieces.
        machine: The target machine, used for its dimensions and properties.
        ops_generator: The instance of the OpsGenerator that holds the cached,
            pre-generated Ops for each workpiece.
        context: An optional ExecutionContext for reporting progress and
            handling cancellation in an async task environment.

    Returns:
        A single, combined Ops object representing the entire job, ready to be
        encoded or sent to a driver.
    """
    final_ops = Ops()
    machine_width, machine_height = machine.dimensions
    clip_rect = 0, 0, machine_width, machine_height

    # Group visible work items by their step
    work_items_by_step: Dict[Step, List[WorkPiece]] = {}
    total_items = 0
    for layer in doc.layers:
        renderable_items = layer.get_renderable_items()
        for step, workpiece in renderable_items:
            if step not in work_items_by_step:
                work_items_by_step[step] = []
            work_items_by_step[step].append(workpiece)
        total_items += len(renderable_items)

    if not total_items:
        return final_ops

    processed_items = 0
    for step, workpieces in work_items_by_step.items():
        step_combined_ops = Ops()

        for workpiece in workpieces:
            if context:
                if context.is_cancelled():
                    raise CancelledError("Operation cancelled")
                processed_items += 1
                context.set_progress(processed_items / total_items)
                context.set_message(
                    _("Processing '{workpiece}' in '{step}'").format(
                        workpiece=workpiece.source_file.name, step=step.name
                    )
                )
                await asyncio.sleep(0)

            # This is the critical hand-off from the generator to the
            # assembler.
            workpiece_ops = ops_generator.get_ops(step, workpiece)
            if not workpiece_ops:
                continue

            clipped_ops = _transform_and_clip_workpiece_ops(
                workpiece_ops, workpiece, machine, clip_rect
            )
            step_combined_ops += clipped_ops

        # Apply post-step transformers to the combined ops for this step
        post_transformers = _instantiate_transformers_from_step(step)
        for transformer in post_transformers:
            if context:
                context.set_message(
                    _("Applying '{transformer}' to '{step}'").format(
                        transformer=transformer.label, step=step.name
                    )
                )
                await asyncio.sleep(0)
            transformer.run(step_combined_ops, context=context)

        final_ops += step_combined_ops

    if context:
        context.set_progress(1.0)
        context.set_message(_("Job assembly complete"))
        context.flush()
    return final_ops
