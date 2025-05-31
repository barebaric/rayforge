from __future__ import annotations
import logging
import asyncio
from abc import ABC
from typing import List, Dict, Iterator, Tuple, Optional
from copy import deepcopy
from ..task import task_mgr, CancelledError
from ..config import config, getflag
from ..modifier import Modifier, MakeTransparent, ToGrayscale
from ..opsproducer import OpsProducer, OutlineTracer, EdgeTracer, Rasterizer
from ..opstransformer import OpsTransformer, Optimize, Smooth, ArcWeld
from .workpiece import WorkPiece
from .machine import Laser
from .ops import (Ops, SetPowerCommand, SetCutSpeedCommand,
                  SetTravelSpeedCommand, EnableAirAssistCommand,
                  DisableAirAssistCommand)
from blinker import Signal


logger = logging.getLogger(__name__)

DEBUG_OPTIMIZE = getflag('DEBUG_OPTIMIZE')
DEBUG_SMOOTH = getflag('DEBUG_SMOOTH')
DEBUG_ARCWELD = getflag('DEBUG_ARCWELD')


class WorkStep(ABC):
    """
    A WorkStep is a set of Modifiers that operate on a set of
    WorkPieces. It normally generates a Ops in the end, but
    may also include modifiers that manipulate the input image.
    """
    typelabel = None

    def __init__(self, opsproducer: OpsProducer, name=None):
        if not self.typelabel:
            raise AttributeError('BUG: subclass must set typelabel attribute')

        self.workplan: Optional[WorkPlan] = None
        self.name: str = name or self.typelabel
        self.visible: bool = True
        self.modifiers: List[Modifier] = [
            MakeTransparent(),
            ToGrayscale(),
        ]
        self._modifier_ref_for_pyreverse: Modifier
        self.opsproducer: OpsProducer = opsproducer
        self.opstransformers: List[OpsTransformer] = []
        self._opstransformer_ref_for_pyreverse: OpsTransformer

        # Map WorkPieces to Ops and size
        self.workpiece_to_ops: Dict[
            WorkPiece,
            Tuple[Optional[Ops], Optional[Tuple[float, float]]]
        ] = {}
        self._workpiece_ref_for_pyreverse: WorkPiece
        self._ops_ref_for_pyreverse: Ops

        self.passes: int = 1
        self.pixels_per_mm = 50, 50

        self.changed = Signal()
        self.ops_generation_starting = Signal()  # (sender, workpiece)
        self.ops_chunk_available = Signal()      # (sender, workpiece, chunk)
        self.ops_generation_finished = Signal()  # (sender, workpiece)
        self.laser: Laser = Laser()
        self.set_laser(config.machine.heads[0])

        self.power: int = self.laser.max_power
        self.cut_speed: int = config.machine.max_cut_speed
        self.travel_speed: int = config.machine.max_travel_speed
        self.air_assist: bool = False

        if DEBUG_OPTIMIZE:
            self.opstransformers.append(Optimize())
        if DEBUG_SMOOTH:
            self.opstransformers.append(Smooth())
        if DEBUG_ARCWELD:
            self.opstransformers.append(ArcWeld())

    def set_passes(self, passes=True):
        self.passes = int(passes)
        self.update_all_workpieces()
        self.changed.send(self)

    def set_visible(self, visible=True):
        self.visible = visible
        self.changed.send(self)

    def set_laser(self, laser):
        if laser == self.laser:
            return
        if self.laser:
            self.laser.changed.disconnect(self._on_laser_changed)
        self.laser = laser
        laser.changed.connect(self._on_laser_changed)
        self.update_all_workpieces()
        self.changed.send(self)

    def _on_laser_changed(self, sender, **kwargs):
        self.update_all_workpieces()
        self.changed.send(self)

    def set_power(self, power):
        self.power = power
        self.update_all_workpieces()
        self.changed.send(self)

    def set_cut_speed(self, speed):
        """Sets the cut speed and triggers regeneration."""
        self.cut_speed = int(speed)
        self.update_all_workpieces()
        self.changed.send(self)

    def set_travel_speed(self, speed):
        """Sets the travel speed and triggers regeneration."""
        self.travel_speed = int(speed)
        self.update_all_workpieces()
        self.changed.send(self)

    def set_air_assist(self, enabled: bool):
        """Sets air assist state and triggers regeneration."""
        self.air_assist = bool(enabled)
        self.update_all_workpieces()
        self.changed.send(self)

    def set_workpieces(self, workpieces: List[WorkPiece]):
        for workpiece in list(self.workpiece_to_ops.keys()):
            if workpiece in workpieces:
                continue
            workpiece.size_changed.disconnect(self._on_workpiece_size_changed)
            del self.workpiece_to_ops[workpiece]
        for workpiece in workpieces:
            self.add_workpiece(workpiece)
        self.changed.send(self)

    def add_workpiece(self, workpiece: WorkPiece):
        if workpiece in self.workpiece_to_ops:
            return
        self.workpiece_to_ops[workpiece] = None, None
        workpiece.size_changed.connect(self._on_workpiece_size_changed)
        self.update_workpiece(workpiece)
        self.changed.send(self)

    def remove_workpiece(self, workpiece: WorkPiece):
        workpiece.size_changed.disconnect(self._on_workpiece_size_changed)
        del self.workpiece_to_ops[workpiece]
        self.changed.send(self)

    def _on_workpiece_size_changed(self, workpiece):
        logger.debug(
            f"WorkStep '{self.name}' size change for {workpiece.name}"
        )
        # Always update when workpiece size changes.
        # Scalable steps re-run execute() for the new scale factor.
        # Non-scalable steps (like raster) regenerate for the new size.
        self.update_workpiece(workpiece)

    def workpieces(self):
        return self.workpiece_to_ops.keys()

    def execute(self, workpiece: WorkPiece) -> Iterator[Ops]:
        """
        Generates Ops chunks for the given workpiece.

        Yields geometry Ops chunks based on rendering/tracing the workpiece.

        workpiece: the input workpiece to generate Ops for.
        """
        if not workpiece.size:
            logger.error(
                f"failed to render workpiece {workpiece.name}: missing size"
            )
            return

        # Yield state-setting commands once per step
        initial_ops = Ops()
        initial_ops.set_power(self.power)
        initial_ops.set_cut_speed(self.cut_speed)
        initial_ops.set_travel_speed(self.travel_speed)
        if self.air_assist:
            initial_ops.enable_air_assist()
        else:
            initial_ops.disable_air_assist()
        yield initial_ops

        if self.can_scale():
            # Vector output, render size doesn't matter much. Use small size.
            surface, _ = workpiece.render(
                *self.pixels_per_mm,
                size=(100, 100),
                force=True
            )
            if not surface:
                logger.error(
                    f"failed to render workpiece {workpiece.name} to surface"
                )
                return

            # Check actual rendered size
            width_px, height_px = surface.get_width(), surface.get_height()
            width_mm = width_px / self.pixels_per_mm[0]
            height_mm = height_px / self.pixels_per_mm[1]
            # Store the size the Ops corresponds to for later scaling
            size = width_mm, height_mm
            render_chunks = [(surface, (0, 0))]  # Single chunk only
        else:
            # Non-scalable (e.g., raster), render in chunks at actual size
            size = workpiece.size
            render_chunks = workpiece.render_chunk(*self.pixels_per_mm,
                                                   size=size,
                                                   force=True)

        # Process render chunks
        for surface, (x_offset_px, y_offset_px) in render_chunks:
            # Apply bitmap modifiers.
            for modifier in self.modifiers:
                modifier.run(surface)

            # Produce an Ops object from the resulting surface.
            chunk_ops = self.opsproducer.run(
                config.machine,
                self.laser,
                surface,
                self.pixels_per_mm
            )

            # Translate chunk ops based on its offset within the full workpiece
            # Calculate offsets in mm relative to workpiece bottom-left origin
            y_offset_px = (
                size[1] - (surface.get_height()+y_offset_px)
                / self.pixels_per_mm[1]
            )
            chunk_ops.translate(x_offset_px/self.pixels_per_mm[0], y_offset_px)

            if self.can_scale():
                chunk_ops.scale(workpiece.size[0] / size[0],
                                workpiece.size[1] / size[1])

            yield chunk_ops
            surface.flush()  # Free memory after use

    async def _stream_ops_and_cache(self, workpiece: WorkPiece):
        """
        Internal coroutine to run generation, emit signals, and cache result.
        """
        logger.debug(
            f"WorkStep '{self.name}': Coroutine started for {workpiece.name}."
            f" Emitting ops_generation_starting."
        )
        self.ops_generation_starting.send(self, workpiece=workpiece)
        accumulated_ops = Ops()
        # Use the size determined during execute (vector or actual)
        size_for_cache = getattr(
            self, '_render_size_for_cache', workpiece.size
        )

        # --- Initial Commands ---
        initial_ops = Ops()
        initial_ops.add(SetPowerCommand(self.power))
        initial_ops.add(SetCutSpeedCommand(self.cut_speed))
        initial_ops.add(SetTravelSpeedCommand(self.travel_speed))
        if self.air_assist:
            initial_ops.add(EnableAirAssistCommand())
        if initial_ops.commands:  # Only send if not empty
            self.ops_chunk_available.send(
                self, workpiece=workpiece, chunk=initial_ops
            )
            accumulated_ops += initial_ops

        try:
            # --- Geometry Generation ---
            for chunk_ops in self.execute(workpiece):
                # Yield control *before* processing the chunk to allow
                # cancellation exception to be raised promptly.
                await asyncio.sleep(0)

                # If sleep(0) didn't raise CancelledError, proceed.
                logger.debug(
                    f"Workplan {self.name}: Processing chunk for "
                    f"{workpiece.name}"
                )
                self.ops_chunk_available.send(
                    self, workpiece=workpiece, chunk=chunk_ops
                )
                accumulated_ops += chunk_ops

            # --- Apply Transformers (after all geometry) ---
            for transformer in self.opstransformers:
                transformer.run(accumulated_ops)  # Apply to accumulated ops

            # --- Final Commands ---
            final_ops = Ops()
            if self.air_assist:  # Only disable if it was enabled
                final_ops.add(DisableAirAssistCommand())
            if final_ops.commands:  # Only send if not empty
                self.ops_chunk_available.send(
                    self, workpiece=workpiece, chunk=final_ops
                )
                accumulated_ops += final_ops
            # --- Caching and Completion Signal (on success) ---
            self.workpiece_to_ops[workpiece] = accumulated_ops, size_for_cache
            self.ops_generation_finished.send(self, workpiece=workpiece)

        except CancelledError:
            logger.info(
                f"Workplan {self.name}: Ops generation for {workpiece.name} "
                f"cancelled."
            )
            # Clear partial cache entry if cancelled
            if workpiece in self.workpiece_to_ops:
                self.workpiece_to_ops[workpiece] = None, None
            # Re-raise so the Task wrapper in task.py sees the cancellation
            # and updates its status correctly.
            raise
        except Exception as e:
            # Log error, maybe send an error signal?
            print(f"Error during Ops generation for {workpiece}: {e}")
            # Clear cache entry on error
            if workpiece in self.workpiece_to_ops:
                self.workpiece_to_ops[workpiece] = None, None
            # Optionally send an error signal?
            return  # Stop processing

    def update_workpiece(self, workpiece):
        """Triggers the asynchronous generation and caching for a workpiece."""
        key = id(self), id(workpiece)
        logger.debug(
            f"WorkStep '{self.name}': Scheduling coroutine for"
            f" {workpiece.name} with key {key}"
        )
        # Cancellation of existing task with the same key is handled
        # internally by task_mgr.add_coroutine / add_task.
        # No need to call cancel explicitly here.
        # Add the new coroutine
        task_mgr.add_coroutine(
            self._stream_ops_and_cache(workpiece),
            # No 'when_done' needed here, signals handle updates
            key=key
        )

    def update_all_workpieces(self):
        for workpiece in self.workpiece_to_ops.keys():
            self.update_workpiece(workpiece)

    # _on_ops_created is removed as signals now handle completion/updates

    def get_ops(self, workpiece):
        """
        Returns Ops for the given workpiece, scaled to the size of
        the workpiece.
        Returns None if no Ops were made yet.
        """
        if not workpiece.size:
            logger.error(
                f"failed to render ops for workpiece {workpiece.name}: "
                "missing size"
            )
            return
        ops, size = self.workpiece_to_ops.get(workpiece, (None, None))
        if ops is None or size is None:
            return None
        orig_width_mm, orig_height_mm = size
        width_mm, height_mm = workpiece.size
        ops = deepcopy(ops)
        ops.scale(width_mm/orig_width_mm, height_mm/orig_height_mm)
        return ops

    def get_summary(self):
        power = int(self.power/self.laser.max_power*100)
        speed = int(self.cut_speed)
        return f"{power}% power, {speed} mm/min"

    def can_scale(self):
        return self.opsproducer.can_scale()

    def dump(self, indent=0):
        print("  "*indent, self.name)
        for workpiece in self.workpieces():
            workpiece.dump(1)


class Outline(WorkStep):
    typelabel = "External Outline"

    def __init__(self, name=None, **kwargs):
        super().__init__(OutlineTracer(), name, **kwargs)


class Contour(WorkStep):
    typelabel = "Contour"

    def __init__(self, name=None, **kwargs):
        super().__init__(EdgeTracer(), name, **kwargs)


class Rasterize(WorkStep):
    typelabel = "Raster Engrave"

    def __init__(self, name=None, **kwargs):
        super().__init__(Rasterizer(), name, **kwargs)


class WorkPlan:
    """
    Represents a sequence of worksteps.
    """
    def __init__(self, doc, name):
        self.doc = doc
        self.name: str = name
        self.worksteps: List[WorkStep] = []
        self._workstep_ref_for_pyreverse: WorkStep
        self.changed = Signal()
        self.add_workstep(Contour())

    def __iter__(self):
        return iter(self.worksteps)

    def set_workpieces(self, workpieces):
        for step in self.worksteps:
            step.set_workpieces(workpieces)

    def add_workstep(self, step):
        step.workplan = self
        self.worksteps.append(step)
        step.set_workpieces(self.doc.workpieces)
        self.changed.send(self)

    def remove_workstep(self, workstep):
        self.worksteps.remove(workstep)
        workstep.workplan = None
        self.changed.send(self)

    def set_worksteps(self, worksteps):
        """
        Replace all worksteps.
        """
        self.worksteps = worksteps
        for step in worksteps:
            step.workplan = self
        self.changed.send(self)

    def has_steps(self):
        return len(self.worksteps) > 0

    def execute(self, optimize=True) -> Ops:
        """
        Executes all visible worksteps and returns the final, combined Ops.

        This method synchronously generates, collects, transforms, and
        optimizes operations from all steps for all workpieces.
        It consumes the WorkStep.execute generators internally.

        Args:
            optimize: Whether to apply path optimization.

        Returns:
            A single Ops object containing the fully processed operations.
        """
        final_ops = Ops()
        optimizer = Optimize() if optimize else None

        for step in self.worksteps:
            if not step.visible:
                continue

            for workpiece in step.workpieces():
                if not workpiece.pos or not workpiece.size:
                    continue  # workpiece is not added to canvas

                # Consume the generator for this step/workpiece
                step_ops_for_workpiece = Ops()
                # The step.execute generator yields Ops relative to
                # workpiece (0,0). We need to translate it here for the
                # final machine coordinates.
                generator = step.execute(workpiece=workpiece)
                for ops_chunk in generator:
                    step_ops_for_workpiece += ops_chunk

                # Apply workpiece translation to machine coordinates
                if step_ops_for_workpiece:
                    step_ops_for_workpiece.translate(*workpiece.pos)

                # Apply optimization if enabled, after collecting and
                # translating
                if optimizer and step_ops_for_workpiece:
                    optimizer.run(step_ops_for_workpiece)  # Optimize in-place

                # Apply passes and accumulate
                if step_ops_for_workpiece:
                    final_ops += (step_ops_for_workpiece * step.passes)

        return final_ops
