from __future__ import annotations
import logging
import asyncio
from typing import List, Optional
from ..tasker.manager import CancelledError
from ..tasker.context import ExecutionContext
from ..config import config
from .workpiece import WorkPiece
from .workstep import WorkStep, Outline
from .ops import Ops
from blinker import Signal


logger = logging.getLogger(__name__)


class WorkPlan:
    """
    Represents a sequence of worksteps that define a laser job.
    """

    def __init__(self, doc, name: str):
        self.doc = doc
        self.name: str = name
        self.worksteps: List[WorkStep] = []
        self._workstep_ref_for_pyreverse: WorkStep
        self.changed = Signal()
        self.add_workstep(self.create_workstep(Outline))

    def __iter__(self):
        return iter(self.worksteps)

    def create_workstep(self, workstep_cls, name=None) -> WorkStep:
        """Factory method to create a new workstep with correct config."""
        return workstep_cls(
            doc=self.doc,
            laser=config.machine.heads[0],
            max_cut_speed=config.machine.max_cut_speed,
            max_travel_speed=config.machine.max_travel_speed,
            name=name,
        )

    def set_workpieces(self, workpieces: List[WorkPiece]):
        for step in self.worksteps:
            step.set_workpieces(workpieces)

    def add_workstep(self, step: WorkStep):
        self.worksteps.append(step)
        step.set_workpieces(self.doc.workpieces)
        self.changed.send(self)

    def remove_workstep(self, workstep: WorkStep):
        self.worksteps.remove(workstep)
        self.changed.send(self)

    def set_worksteps(self, worksteps: List[WorkStep]):
        """
        Replace all worksteps.
        """
        self.worksteps = worksteps
        self.changed.send(self)

    def has_steps(self) -> bool:
        return len(self.worksteps) > 0

    async def execute(self, context: Optional[ExecutionContext] = None) -> Ops:
        """
        Executes all visible worksteps and returns the final, combined Ops.

        This method asynchronously collects, transforms, and
        optimizes operations from all steps for all workpieces.

        Returns:
            A single Ops object containing the fully processed operations.
        """
        final_ops = Ops()
        machine_width, machine_height = config.machine.dimensions
        clip_rect = 0, 0, machine_width, machine_height

        work_items = []
        for step in self.worksteps:
            if not step.visible:
                continue
            for workpiece in step.workpieces():
                if not workpiece.pos or not workpiece.size:
                    continue  # workpiece is not added to canvas
                work_items.append((step, workpiece))

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

            # Get the pre-scaled ops (still in local, canonical coords)
            step_ops = await asyncio.to_thread(step.get_ops, workpiece)
            if not step_ops:
                continue

            # 1. Rotate the ops around its local center. G-code uses CCW.
            wp_angle = workpiece.angle
            if wp_angle != 0:
                wp_w, wp_h = workpiece.size
                cx, cy = wp_w / 2, wp_h / 2
                step_ops.rotate(-wp_angle, cx, cy)

            # 2. Translate to final canonical position on the work area
            step_ops.translate(*workpiece.pos)

            # 3. Convert from canonical (Y-up) to machine-native coords
            if config.machine.y_axis_down:
                step_ops.scale(1, -1)
                step_ops.translate(0, machine_height)

            # 4. Clip to machine boundaries and apply post-transformers
            clipped_ops = step_ops.clip(clip_rect)
            for transformer in step.opstransformers:
                await asyncio.to_thread(transformer.run, clipped_ops)
            final_ops += clipped_ops * step.passes

        if context:
            context.set_progress(1.0)
            context.flush()
        return final_ops
