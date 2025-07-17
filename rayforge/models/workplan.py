from __future__ import annotations
import logging
import asyncio
from abc import ABC
from typing import List, Dict, AsyncIterator, Tuple, Optional
from copy import deepcopy
from gi.repository import GLib
from ..task import task_mgr, CancelledError
from ..config import config, getflag
from ..modifier import Modifier, MakeTransparent, ToGrayscale
from ..opsproducer import OpsProducer, OutlineTracer, EdgeTracer, Rasterizer
from ..opstransformer import OpsTransformer, Optimize, Smooth, ArcWeld
from .workpiece import WorkPiece
from .machine import Laser
from .ops import Ops, DisableAirAssistCommand
from blinker import Signal


logger = logging.getLogger(__name__)

DEBUG_OPTIMIZE = getflag("DEBUG_OPTIMIZE")
DEBUG_SMOOTH = getflag("DEBUG_SMOOTH")
DEBUG_ARCWELD = getflag("DEBUG_ARCWELD")

DEBOUNCE_DELAY_MS = 100  # Delay in milliseconds for ops regeneration


class WorkStep(ABC):
    """
    A set of modifiers and an OpsProducer that operate on WorkPieces.

    A WorkStep generates laser operations (Ops) based on its configuration
    and the WorkPieces assigned to it.
    """

    typelabel: str

    def __init__(self, opsproducer: OpsProducer, name: Optional[str] = None):
        if not self.typelabel:
            raise AttributeError("Subclass must set a typelabel attribute.")

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
            WorkPiece, Tuple[Optional[Ops], Optional[Tuple[float, float]]]
        ] = {}
        self._workpiece_ref_for_pyreverse: WorkPiece
        self._ops_ref_for_pyreverse: Ops

        # A dictionary to hold debounce timers
        self._workpiece_update_timers: Dict[WorkPiece, int] = {}

        self.passes: int = 1
        self.pixels_per_mm = 50, 50

        self.changed = Signal()
        self.ops_generation_starting = Signal()
        self.ops_chunk_available = Signal()
        self.ops_generation_finished = Signal()

        self.laser: Laser = Laser()
        self.set_laser(config.machine.heads[0])

        self.power: int = self.laser.max_power
        self.cut_speed: int = config.machine.max_cut_speed
        self.travel_speed: int = config.machine.max_travel_speed
        self.air_assist: bool = False

        if DEBUG_OPTIMIZE:
            logger.info("Travel time optimization debugging enabled")
            self.opstransformers.append(Optimize())
        if DEBUG_SMOOTH:
            logger.info("Smoothing enabled")
            self.opstransformers.append(Smooth())
        if DEBUG_ARCWELD:
            logger.info("Arcweld enabled")
            self.opstransformers.append(ArcWeld())

    def set_passes(self, passes: bool = True):
        self.passes = int(passes)
        self.update_all_workpieces()
        self.changed.send(self)

    def set_visible(self, visible: bool = True):
        self.visible = visible
        self.changed.send(self)

    def set_laser(self, laser: Laser):
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

    def set_power(self, power: int):
        self.power = power
        self.update_all_workpieces()
        self.changed.send(self)

    def set_cut_speed(self, speed: int):
        """Sets the cut speed and triggers regeneration."""
        self.cut_speed = int(speed)
        self.update_all_workpieces()
        self.changed.send(self)

    def set_travel_speed(self, speed: int):
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
            # Ensure timer is cleaned up
            if workpiece in self._workpiece_update_timers:
                GLib.source_remove(
                    self._workpiece_update_timers.pop(workpiece)
                )
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
        # Ensure timer is cleaned up on removal
        if workpiece in self._workpiece_update_timers:
            GLib.source_remove(self._workpiece_update_timers.pop(workpiece))
        workpiece.size_changed.disconnect(self._on_workpiece_size_changed)
        del self.workpiece_to_ops[workpiece]
        self.changed.send(self)

    def _on_workpiece_size_changed(self, workpiece: WorkPiece):
        # If a timer is already running for this workpiece, cancel it.
        if workpiece in self._workpiece_update_timers:
            GLib.source_remove(self._workpiece_update_timers[workpiece])

        def _update_callback():
            """This function will be called after the delay."""
            logger.debug(
                f"Debounced update triggered for {workpiece.name}"
            )
            # Check if workpiece still exists before updating
            if workpiece in self.workpiece_to_ops:
                self.update_workpiece(workpiece)
            # Remove the timer ID from the dictionary
            if workpiece in self._workpiece_update_timers:
                del self._workpiece_update_timers[workpiece]
            return GLib.SOURCE_REMOVE  # Ensures the timer runs only once

        # Schedule the update to run after a short delay.
        timer_id = GLib.timeout_add(DEBOUNCE_DELAY_MS, _update_callback)
        self._workpiece_update_timers[workpiece] = timer_id

    def workpieces(self) -> List[WorkPiece]:
        return list(self.workpiece_to_ops.keys())

    async def execute(
        self, workpiece: WorkPiece
    ) -> AsyncIterator[Tuple[Ops, Optional[Tuple[int, int]]]]:
        """
        Asynchronously generates Ops chunks for a given WorkPiece.

        For scalable ops, it yields a single un-scaled geometry chunk along
        with the pixel dimensions of the surface it was traced from. For
        non-scalable ops (e.g., raster), it yields scaled chunks with no
        pixel dimensions.
        """
        if not workpiece.size:
            logger.error(f"Cannot render {workpiece.name}: missing size.")
            return

        def _blocking_trace_and_modify(surface, scaler):
            for modifier in self.modifiers:
                modifier.run(surface)
            return self.opsproducer.run(
                config.machine, self.laser, surface, scaler
            )

        if self.can_scale():
            TRACE_RESOLUTION_PX = 2400
            aspect_ratio = workpiece.get_current_aspect_ratio() or 1.0

            target_width = (
                TRACE_RESOLUTION_PX
                if aspect_ratio >= 1
                else int(TRACE_RESOLUTION_PX * aspect_ratio)
            )
            target_height = (
                int(TRACE_RESOLUTION_PX / aspect_ratio)
                if aspect_ratio > 1
                else TRACE_RESOLUTION_PX
            )

            surface = await asyncio.to_thread(
                workpiece.renderer.render_to_pixels,
                width=target_width,
                height=target_height,
            )

            if not surface:
                logger.error(
                    f"failed to render workpiece {workpiece.name} to surface"
                )
                return

            pixel_scaler = (1.0, 1.0)
            geometry_ops = await asyncio.to_thread(
                _blocking_trace_and_modify, surface, pixel_scaler
            )

            yield geometry_ops, (surface.get_width(), surface.get_height())
            surface.flush()

        else:  # Raster
            size = workpiece.size
            for surface, (x_offset_px, y_offset_px) in workpiece.render_chunk(
                *self.pixels_per_mm, size=size, chunk_height=20
            ):
                chunk_ops = await asyncio.to_thread(
                    _blocking_trace_and_modify, surface, self.pixels_per_mm
                )
                y_offset_mm = (
                    size[1] * self.pixels_per_mm[1]
                    - (surface.get_height() + y_offset_px)
                ) / self.pixels_per_mm[1]
                x_offset_mm = x_offset_px / self.pixels_per_mm[0]
                chunk_ops.translate(x_offset_mm, y_offset_mm)
                yield chunk_ops, None
                surface.flush()

    async def _stream_ops_and_cache(self, workpiece: WorkPiece):
        """
        Internal coroutine to run generation, emit signals, and cache
        the final result.
        """
        logger.debug(
            f"WorkStep '{self.name}': Coroutine started for {workpiece.name}."
        )
        self.ops_generation_starting.send(self, workpiece=workpiece)

        final_ops = Ops()
        cached_pixel_size = None

        try:
            initial_ops = Ops()
            initial_ops.set_power(self.power)
            initial_ops.set_cut_speed(self.cut_speed)
            initial_ops.set_travel_speed(self.travel_speed)
            initial_ops.enable_air_assist(self.air_assist)
            if initial_ops.commands:
                self.ops_chunk_available.send(
                    self, workpiece=workpiece, chunk=initial_ops
                )

            final_ops += initial_ops

            async for chunk, px_size in self.execute(workpiece):
                if px_size:
                    cached_pixel_size = px_size

                if self.can_scale() and cached_pixel_size:
                    display_ops = deepcopy(chunk)
                    final_mm = workpiece.get_current_size()
                    if final_mm is not None and None not in final_mm:
                        scale_x = final_mm[0] / cached_pixel_size[0]
                        scale_y = final_mm[1] / cached_pixel_size[1]
                    else:
                        scale_x = 1
                        scale_y = 1
                    display_ops.scale(scale_x, scale_y)
                    self.ops_chunk_available.send(
                        self, workpiece=workpiece, chunk=display_ops
                    )
                else:
                    self.ops_chunk_available.send(
                        self, workpiece=workpiece, chunk=chunk
                    )
                final_ops += chunk

            for transformer in self.opstransformers:
                name = transformer.__class__.__name__
                logger.debug(f"Running transformer {name}")
                transformer.run(final_ops)

            if self.air_assist:
                final_ops.add(DisableAirAssistCommand())

            self.workpiece_to_ops[workpiece] = final_ops, cached_pixel_size
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
            logger.error(
                f"Error during Ops generation for {workpiece}: {e}",
                exc_info=True,
            )
            if workpiece in self.workpiece_to_ops:
                self.workpiece_to_ops[workpiece] = None, None
            return

    def update_workpiece(self, workpiece: WorkPiece):
        """Triggers the asynchronous generation and caching for a workpiece."""
        key = id(self), id(workpiece)
        logger.debug(
            f"WorkStep '{self.name}': Scheduling coroutine for"
            f" {workpiece.name} with key {key}"
        )
        task_mgr.add_coroutine(self._stream_ops_and_cache(workpiece), key=key)

    def update_all_workpieces(self):
        for workpiece in self.workpiece_to_ops.keys():
            self.update_workpiece(workpiece)

    def get_ops(self, workpiece: WorkPiece) -> Optional[Ops]:
        """
        Returns final, scaled Ops for a workpiece from the cache.
        """
        if not workpiece.size:
            logger.error(
                f"failed to render ops for workpiece {workpiece.name}: "
                "missing size"
            )
            return

        raw_ops, pixel_size = self.workpiece_to_ops.get(
            workpiece, (None, None)
        )
        if raw_ops is None:
            return None

        ops = deepcopy(raw_ops)

        if pixel_size:
            traced_width_px, traced_height_px = pixel_size
            final_width_mm, final_height_mm = workpiece.size
            if traced_width_px > 0 and traced_height_px > 0:
                scale_x = final_width_mm / traced_width_px
                scale_y = final_height_mm / traced_height_px
                ops.scale(scale_x, scale_y)

        return ops

    def get_summary(self) -> str:
        power = int(self.power / self.laser.max_power * 100)
        speed = int(self.cut_speed)
        return f"{power}% power, {speed} mm/min"

    def can_scale(self) -> bool:
        return self.opsproducer.can_scale()

    def dump(self, indent: int = 0):
        print("  " * indent, self.name)
        for workpiece in self.workpieces():
            workpiece.dump(1)


class Outline(WorkStep):
    typelabel = _("External Outline")

    def __init__(self, name=None, **kwargs):
        super().__init__(OutlineTracer(), name=name, **kwargs)


class Contour(WorkStep):
    typelabel = _("Contour")

    def __init__(self, name=None, **kwargs):
        super().__init__(EdgeTracer(), name=name, **kwargs)


class Rasterize(WorkStep):
    typelabel = _("Raster Engrave")

    def __init__(self, name=None, **kwargs):
        super().__init__(Rasterizer(), name=name, **kwargs)


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
        self.add_workstep(Outline())

    def __iter__(self):
        return iter(self.worksteps)

    def set_workpieces(self, workpieces: List[WorkPiece]):
        for step in self.worksteps:
            step.set_workpieces(workpieces)

    def add_workstep(self, step: WorkStep):
        step.workplan = self
        self.worksteps.append(step)
        step.set_workpieces(self.doc.workpieces)
        self.changed.send(self)

    def remove_workstep(self, workstep: WorkStep):
        self.worksteps.remove(workstep)
        workstep.workplan = None
        self.changed.send(self)

    def set_worksteps(self, worksteps: List[WorkStep]):
        """
        Replace all worksteps.
        """
        self.worksteps = worksteps
        for step in worksteps:
            step.workplan = self
        self.changed.send(self)

    def has_steps(self) -> bool:
        return len(self.worksteps) > 0

    def execute(self, optimize: bool = True) -> Ops:
        """
        Executes all visible worksteps and returns the final, combined Ops.

        This method synchronously generates, collects, transforms, and
        optimizes operations from all steps for all workpieces.

        Args:
            optimize: Whether to apply path optimization.

        Returns:
            A single Ops object containing the fully processed operations.
        """
        final_ops = Ops()
        optimizer = Optimize() if optimize else None
        machine_width, machine_height = config.machine.dimensions
        clip_rect = 0, 0, machine_width, machine_height

        for step in self.worksteps:
            if not step.visible:
                continue

            for workpiece in step.workpieces():
                if not workpiece.pos or not workpiece.size:
                    continue  # workpiece is not added to canvas

                step_ops = step.get_ops(workpiece)
                if step_ops:
                    step_ops.translate(*workpiece.pos)
                    # Clip the translated ops to the machine's work area.
                    clipped_ops = step_ops.clip(clip_rect)

                    # Apply optimization if enabled, after collecting and
                    # clipping
                    if optimizer:
                        optimizer.run(clipped_ops)
                    final_ops += clipped_ops * step.passes

        return final_ops
