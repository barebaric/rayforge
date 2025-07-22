from __future__ import annotations
import logging
import asyncio
from abc import ABC
from typing import List, Dict, AsyncIterator, Tuple, Optional, Callable
from copy import deepcopy
from gi.repository import GLib
from ..task import task_mgr, CancelledError, ExecutionContext
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

        self._generation_id = 0  # Cancellation token

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
            logger.debug(f"Debounced update triggered for {workpiece.name}")
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

    def _trace_and_modify_surface(self, surface, scaler):
        """Applies modifiers and runs the OpsProducer on a surface."""
        for modifier in self.modifiers:
            modifier.run(surface)
        return self.opsproducer.run(
            config.machine, self.laser, surface, scaler
        )

    async def _execute_vector(
        self, workpiece: WorkPiece, check_cancelled: Callable[[], bool]
    ):
        """Handles Ops generation for scalable (vector) operations."""
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

        if check_cancelled():
            return
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

        if check_cancelled():
            return

        pixel_scaler = (1.0, 1.0)
        geometry_ops = await asyncio.to_thread(
            self._trace_and_modify_surface, surface, pixel_scaler
        )

        yield geometry_ops, (surface.get_width(), surface.get_height()), 1.0
        surface.flush()

    async def _execute_raster(
        self, workpiece: WorkPiece, check_cancelled: Callable[[], bool]
    ):
        """Handles Ops generation for non-scalable (raster) operations."""
        size = workpiece.get_current_size()
        if not size:
            return
        total_height_px = size[1] * self.pixels_per_mm[1]

        chunk_iter = iter(
            workpiece.render_chunk(
                *self.pixels_per_mm,
                size=size,
                max_memory_size=10 * 1024 * 1024,
            )
        )

        def get_next_chunk():
            try:
                return next(chunk_iter)
            except StopIteration:
                return None

        if check_cancelled():
            return
        render_task = asyncio.to_thread(get_next_chunk)

        while True:
            if check_cancelled():
                return
            current_chunk_data = await render_task
            if current_chunk_data is None:
                break

            render_task = asyncio.to_thread(get_next_chunk)

            surface, (x_offset_px, y_offset_px) = current_chunk_data

            progress = 0.0
            if total_height_px > 0:
                processed_height_px = y_offset_px + surface.get_height()
                progress = min(1.0, processed_height_px / total_height_px)

            if check_cancelled():
                return
            chunk_ops = await asyncio.to_thread(
                self._trace_and_modify_surface, surface, self.pixels_per_mm
            )

            y_offset_mm = (
                size[1] * self.pixels_per_mm[1]
                - (surface.get_height() + y_offset_px)
            ) / self.pixels_per_mm[1]
            x_offset_mm = x_offset_px / self.pixels_per_mm[0]
            chunk_ops.translate(x_offset_mm, y_offset_mm)

            yield chunk_ops, None, progress
            surface.flush()

    async def execute(
        self, workpiece: WorkPiece, check_cancelled: Callable[[], bool]
    ) -> AsyncIterator[Tuple[Ops, Optional[Tuple[int, int]], float]]:
        """
        Asynchronously generates Ops chunks for a given WorkPiece by
        dispatching to the appropriate vector or raster method.
        """
        if not workpiece.size:
            logger.error(f"Cannot render {workpiece.name}: missing size.")
            return

        if self.can_scale():
            async for item in self._execute_vector(workpiece, check_cancelled):
                yield item
        else:
            async for item in self._execute_raster(workpiece, check_cancelled):
                yield item

    def _create_initial_ops(self):
        """Creates and configures the initial Ops object."""
        initial_ops = Ops()
        initial_ops.set_power(self.power)
        initial_ops.set_cut_speed(self.cut_speed)
        initial_ops.set_travel_speed(self.travel_speed)
        initial_ops.enable_air_assist(self.air_assist)
        return initial_ops

    def _get_display_ops(
        self,
        chunk: Ops,
        pixel_size: Optional[Tuple[int, int]],
        workpiece: WorkPiece,
    ) -> Ops:
        """Returns a scaled version of the ops for display."""
        if self.can_scale() and pixel_size:
            display_ops = deepcopy(chunk)
            final_mm = workpiece.get_current_size()
            if final_mm is not None and None not in final_mm:
                scale_x = final_mm[0] / pixel_size[0]
                scale_y = final_mm[1] / pixel_size[1]
            else:
                scale_x = 1
                scale_y = 1
            display_ops.scale(scale_x, scale_y)
            return display_ops
        return chunk

    async def _stream_ops_and_cache(
        self,
        context: ExecutionContext,
        workpiece: WorkPiece,
        generation_id: int,
    ):
        """
        Internal coroutine to run generation, emit signals, and cache
        the final result. Progress is reported via the context.
        """
        self.ops_generation_starting.send(self, workpiece=workpiece)
        final_ops = Ops()
        cached_pixel_size = None

        execute_weight = 0.20
        transform_weight = 1.0 - execute_weight

        def check_cancelled():
            return (
                self._generation_id != generation_id or context.is_cancelled()
            )

        try:
            # The root context's total is 1.0 by default.
            context.set_message(
                _("Generating path for '{name}'").format(name=workpiece.name)
            )
            initial_ops = self._create_initial_ops()
            final_ops += initial_ops

            # Execute Phase (0% -> 20% of total progress)
            # Create a sub-context for just this phase.
            execute_ctx = context.sub_context(
                base_progress=0.0,
                progress_range=execute_weight,
                check_cancelled=check_cancelled,
            )

            async for chunk, px_size, execute_progress in self.execute(
                workpiece, check_cancelled
            ):
                if check_cancelled():
                    raise CancelledError()
                # execute_progress is 0.0-1.0, and execute_ctx.total is 1.0,
                # so this works perfectly.
                execute_ctx.set_progress(execute_progress)
                if px_size:
                    cached_pixel_size = px_size
                display_ops = self._get_display_ops(chunk, px_size, workpiece)
                self.ops_chunk_available.send(
                    self, workpiece=workpiece, chunk=display_ops
                )
                final_ops += chunk

            # Transform Phase (20% -> 100% of total progress)
            if self.opstransformers:
                num_transformers = len(self.opstransformers)

                # Create a single sub-context for the entire transformation
                # block, giving it the total number of steps directly.
                transform_context = context.sub_context(
                    base_progress=execute_weight,
                    progress_range=transform_weight,
                    total=num_transformers,
                    check_cancelled=check_cancelled,
                )

                for i, transformer in enumerate(self.opstransformers):
                    if check_cancelled():
                        raise CancelledError()

                    context.set_message(
                        _("Applying '{transformer}' on '{workpiece}'").format(
                            transformer=transformer.__class__.__name__,
                            workpiece=workpiece.name,
                        )
                    )

                    # Create a context for the transformer's own internal
                    # progress.
                    # Its slice is one "step" of the parent transform_context.
                    transformer_run_ctx = transform_context.sub_context(
                        base_progress=i,  # Raw step number
                        progress_range=1,  # This slice is 1 step wide
                        check_cancelled=check_cancelled,
                    )

                    await asyncio.to_thread(
                        transformer.run, final_ops, context=transformer_run_ctx
                    )

                    # Mark the step as complete in the transform_context.
                    transform_context.set_progress(i + 1)

            if check_cancelled():
                raise CancelledError()

            if self.air_assist:
                final_ops.add(DisableAirAssistCommand())

            context.set_message(
                _("Finalizing '{workpiece}'").format(workpiece=workpiece.name)
            )
            context.set_progress(1.0)
            self.workpiece_to_ops[workpiece] = final_ops, cached_pixel_size
            self.ops_generation_finished.send(self, workpiece=workpiece)
        except CancelledError:
            logger.info(
                f"Workplan {self.name}: Ops generation for {workpiece.name} "
                f"(gen {generation_id}) cancelled."
            )
            if (
                self._generation_id == generation_id
                and workpiece in self.workpiece_to_ops
            ):
                self.workpiece_to_ops[workpiece] = None, None
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
        self._generation_id += 1
        current_generation_id = self._generation_id

        key = id(self), id(workpiece)
        logger.debug(
            f"WorkStep '{self.name}': Scheduling coroutine for "
            f"{workpiece.name} with gen_id {current_generation_id}"
        )

        task_mgr.add_coroutine(
            self._stream_ops_and_cache,
            workpiece,
            current_generation_id,
            key=key,
        )

    def update_all_workpieces(self):
        for workpiece in self.workpiece_to_ops.keys():
            self.update_workpiece(workpiece)

    def get_ops(self, workpiece: WorkPiece) -> Optional[Ops]:
        """
        Returns final, scaled Ops for a workpiece from the cache.
        """
        if not workpiece.size:
            logger.error(f"missing size for workpiece {workpiece.name}")
            return None

        raw_ops, pixel_size = self.workpiece_to_ops.get(
            workpiece, (None, None)
        )
        if raw_ops is None:
            return None

        ops = deepcopy(raw_ops)

        if pixel_size:
            traced_width_px, traced_height_px = pixel_size
            size = workpiece.get_current_size() or (0, 0)
            final_width_mm, final_height_mm = size
            if (
                final_width_mm is not None
                and final_height_mm is not None
                and traced_width_px > 0
                and traced_height_px > 0
            ):
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

            step_ops = await asyncio.to_thread(step.get_ops, workpiece)
            if step_ops:
                step_ops.translate(*workpiece.pos)
                # Clip the translated ops to the machine's work area.
                clipped_ops = step_ops.clip(clip_rect)

                # Apply transformers after clipping.
                for transformer in step.opstransformers:
                    await asyncio.to_thread(transformer.run, clipped_ops)
                final_ops += clipped_ops * step.passes

        if context:
            context.set_progress(1.0)
            context.flush()
        return final_ops
