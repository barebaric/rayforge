"""
Handles the final assembly of machine operations for an entire job.
"""

import asyncio
import logging
from typing import Optional
from ..machine.models.machine import Machine
from ..shared.tasker.context import ExecutionContext
from ..shared.tasker.manager import CancelledError
from ..core.doc import Doc
from ..core.ops import Ops
from ..pipeline.generator import OpsGenerator


logger = logging.getLogger(__name__)


async def generate_job_ops(
    doc: Doc,
    machine: Machine,
    ops_generator: OpsGenerator,
    context: Optional[ExecutionContext] = None,
) -> Ops:
    """
    Assembles all workpiece Ops into a single, final job for a machine.

    This function iterates through all visible step/workpiece pairs in a
    document. For each pair, it fetches the pre-generated, cached `Ops` from
    the `OpsGenerator`. It then applies the final transformations to place
    these local `Ops` into the machine's global coordinate space.

    The transformations include:
    1. Rotating the `Ops` around the workpiece's local center.
    2. Translating the `Ops` to the workpiece's final position on the work
       area.
    3. Flipping the Y-axis if the machine's coordinate system requires it.
    4. Clipping the final `Ops` to the machine's physical boundaries.
    5. Applying the number of passes specified in the step.

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

    work_items = []
    for layer in doc.layers:
        work_items.extend(layer.get_renderable_items())

    if not work_items:
        return final_ops

    total_items = len(work_items)
    for i, (step, workpiece) in enumerate(work_items):
        if context:
            if context.is_cancelled():
                raise CancelledError("Operation cancelled")
            context.set_progress(i / total_items)
            context.set_message(
                _("Processing '{workpiece}' in '{step}'").format(
                    workpiece=workpiece.name, step=step.name
                )
            )
            await asyncio.sleep(0)

        # This is the critical hand-off from the generator to the assembler.
        step_ops = ops_generator.get_ops(step, workpiece)
        if not step_ops:
            continue

        # 1. Rotate the ops around its local center.
        wp_angle = workpiece.angle
        if wp_angle != 0:
            wp_w, wp_h = workpiece.size
            cx, cy = wp_w / 2, wp_h / 2
            step_ops.rotate(-wp_angle, cx, cy)

        # 2. Translate to final canonical position on the work area
        step_ops.translate(*workpiece.pos)

        # 3. Convert from canonical (Y-up) to machine-native coords
        if machine.y_axis_down:
            step_ops.scale(1, -1)
            step_ops.translate(0, machine_height)

        # 4. Clip to machine boundaries and apply post-transformers
        clipped_ops = step_ops.clip(clip_rect)
        final_ops += clipped_ops * step.passes

    if context:
        context.set_progress(1.0)
        context.flush()
    return final_ops
