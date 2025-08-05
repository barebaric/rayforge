import asyncio
import logging
from typing import Optional
from ..machine.models.machine import Machine
from ..shared.tasker.context import ExecutionContext
from ..shared.tasker.manager import CancelledError
from ..core.doc import Doc
from ..core.ops import Ops


logger = logging.getLogger(__name__)


async def generate_job_ops(
    doc: Doc, machine: Machine, context: Optional[ExecutionContext] = None
) -> Ops:
    """
    Executes all steps in all assigned workflows for a document and returns
    the final, combined Ops for the entire job, tailored for a specific
    machine.
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

        step_ops = workpiece.layer.get_ops(step, workpiece)
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
