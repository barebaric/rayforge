from __future__ import annotations
import logging
from abc import ABC
from typing import Any, List, Dict, Tuple, Optional, Iterator
from copy import deepcopy
from blinker import Signal
from gi.repository import GLib
from ..tasker import task_mgr, Task, ExecutionContextProxy
from ..modifier import Modifier, MakeTransparent, ToGrayscale
from ..opsproducer import OpsProducer, OutlineTracer, EdgeTracer, Rasterizer
from ..opstransformer import OpsTransformer, Optimize, Smooth
from .workpiece import WorkPiece
from .laser import Laser
from .ops import Ops, DisableAirAssistCommand


logger = logging.getLogger(__name__)

MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024
DEBOUNCE_DELAY_MS = 250  # Delay in milliseconds for ops regeneration


# This top-level function contains the core logic for generating Ops.
# It is designed to be run in a separate process by the TaskManager.
def _execute_workstep_in_subprocess(
    proxy: ExecutionContextProxy,
    # Pass all required state. Assume these are pickleable.
    workpiece: WorkPiece,
    opsproducer: OpsProducer,
    modifiers: List[Modifier],
    opstransformers: List[OpsTransformer],
    laser: Laser,
    settings: dict,
):
    # Helper functions
    def _trace_and_modify_surface(surface, scaler):
        """Applies modifiers and runs the OpsProducer on a surface."""
        for modifier in modifiers:
            modifier.run(surface)
        return opsproducer.run(laser, surface, scaler)

    def _execute_vector() -> Iterator[Tuple[Ops, Tuple[int, int], float]]:
        """
        Handles Ops generation for scalable (vector) operations.
        This is a synchronous version of the original async method.
        """
        size_mm = workpiece.get_current_size()

        if not size_mm or None in size_mm:
            logger.warning(
                f"Cannot generate vector ops for '{workpiece.name}' "
                "without a defined size. Skipping."
            )
            return

        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]
        target_width = int(size_mm[0] * px_per_mm_x)
        target_height = int(size_mm[1] * px_per_mm_y)

        # Cap resolution
        num_pixels = target_width * target_height
        if num_pixels > MAX_VECTOR_TRACE_PIXELS:
            scale_factor = (MAX_VECTOR_TRACE_PIXELS / num_pixels) ** 0.5
            target_width = int(target_width * scale_factor)
            target_height = int(target_height * scale_factor)

        # This is now a blocking call, which is fine in a subprocess.
        surface = workpiece.renderer.render_to_pixels(
            width=target_width, height=target_height
        )
        if not surface:
            return

        pixel_scaler = 1.0, 1.0
        geometry_ops = _trace_and_modify_surface(surface, pixel_scaler)
        yield geometry_ops, (surface.get_width(), surface.get_height()), 1.0
        surface.flush()

    def _execute_raster() -> Iterator[Tuple[Ops, None, float]]:
        """
        Handles Ops generation for non-scalable (raster) operations.
        This is a synchronous version of the original async method.
        """
        size = workpiece.get_current_size()

        if not size or None in size:
            logger.warning(
                f"Cannot generate raster ops for '{workpiece.name}' "
                "without a defined size. Skipping."
            )
            return

        total_height_px = size[1] * settings["pixels_per_mm"][1]
        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

        # This iterator is now synchronous.
        chunk_iter = workpiece.render_chunk(
            px_per_mm_x,
            px_per_mm_y,
            size=size,
            max_memory_size=10 * 1024 * 1024,
        )

        for surface, (x_offset_px, y_offset_px) in chunk_iter:
            progress = 0.0
            if total_height_px > 0:
                processed_height_px = y_offset_px + surface.get_height()
                progress = min(1.0, processed_height_px / total_height_px)

            chunk_ops = _trace_and_modify_surface(
                surface, (px_per_mm_x, px_per_mm_y)
            )

            y_offset_mm = (
                size[1] * px_per_mm_y - (surface.get_height() + y_offset_px)
            ) / px_per_mm_y
            x_offset_mm = x_offset_px / px_per_mm_x
            chunk_ops.translate(x_offset_mm, y_offset_mm)

            yield chunk_ops, None, progress
            surface.flush()

    def _create_initial_ops():
        """Creates and new_configsures the initial Ops object."""
        initial_ops = Ops()
        initial_ops.set_power(settings["power"])
        initial_ops.set_cut_speed(settings["cut_speed"])
        initial_ops.set_travel_speed(settings["travel_speed"])
        initial_ops.enable_air_assist(settings["air_assist"])
        return initial_ops

    # === Main execution logic for the subprocess ===

    proxy.set_message(
        _("Generating path for '{name}'").format(name=workpiece.name)
    )
    final_ops = _create_initial_ops()
    cached_pixel_size = None

    execute_weight = 0.20
    transform_weight = 1.0 - execute_weight

    # --- Path generation phase ---
    execute_ctx = proxy.sub_context(
        base_progress=0.0, progress_range=execute_weight
    )
    execute_iterator = (
        _execute_vector() if opsproducer.can_scale() else _execute_raster()
    )

    for chunk, px_size, execute_progress in execute_iterator:
        execute_ctx.set_progress(execute_progress)
        if px_size:
            cached_pixel_size = px_size
        final_ops += chunk

    # Ensure path generation is marked as 100% complete before continuing.
    execute_ctx.set_progress(1.0)

    # --- Transform phase ---
    if opstransformers:
        enabled_transformers = [t for t in opstransformers if t.enabled]
        num_transformers = len(enabled_transformers)

        if num_transformers > 0:
            transform_context = proxy.sub_context(
                base_progress=execute_weight, progress_range=transform_weight
            )
        else:
            transform_context = None

        for i, transformer in enumerate(enabled_transformers):
            proxy.set_message(
                _("Applying '{transformer}' on '{workpiece}'").format(
                    transformer=transformer.__class__.__name__,
                    workpiece=workpiece.name,
                )
            )
            # Create a proxy for this transformer's slice of the progress bar
            transformer_run_proxy = transform_context.sub_context(
                base_progress=(i / num_transformers),
                progress_range=(1 / num_transformers),
            )
            # transformer.run now runs synchronously and may use the proxy
            # to report its own fine-grained progress.
            transformer.run(final_ops, context=transformer_run_proxy)

            # Ensure this step's progress is marked as 100% complete before
            # moving to the next one. This prevents progress from appearing
            # to jump or stall if a transformer doesn't report its own
            # completion.
            transformer_run_proxy.set_progress(1.0)

    if settings["air_assist"]:
        final_ops.add(DisableAirAssistCommand())

    proxy.set_message(
        _("Finalizing '{workpiece}'").format(workpiece=workpiece.name)
    )
    proxy.set_progress(1.0)

    # The final result is returned and sent back by the _process_target_wrapper
    return final_ops, cached_pixel_size


class WorkStep(ABC):
    """
    A set of modifiers and an OpsProducer that operate on WorkPieces.

    A WorkStep generates laser operations (Ops) based on its configuration
    and the WorkPieces assigned to it.
    """

    typelabel: str

    def __init__(
        self,
        opsproducer: OpsProducer,
        laser: Laser,
        max_cut_speed: int,
        max_travel_speed: int,
        name: Optional[str] = None,
    ):
        if not self.typelabel:
            raise AttributeError("Subclass must set a typelabel attribute.")

        self.name = name or self.typelabel
        self.visible = True
        self.modifiers = [
            MakeTransparent(),
            ToGrayscale(),
        ]
        self._modifier_ref_for_pyreverse: Modifier
        self.opsproducer = opsproducer
        self.opstransformers: List[OpsTransformer] = []

        # Maps UID to workpiece.
        self._workpieces: Dict[Any, WorkPiece] = {}
        self._ops_cache: Dict[
            Any, Tuple[Optional[Ops], Optional[Tuple[int, int]]]
        ] = {}
        self._workpiece_update_timers: Dict[Any, int] = {}

        self._generation_id_map: Dict[Any, int] = {}

        self.passes: int = 1
        self.pixels_per_mm = 50, 50

        self.changed = Signal()
        self.ops_generation_starting = Signal()
        self.ops_chunk_available = Signal()
        self.ops_generation_finished = Signal()

        self.laser = laser
        self.set_laser(laser)

        self.power = self.laser.max_power
        self.cut_speed = max_cut_speed
        self.travel_speed = max_travel_speed
        self.air_assist = False

    def update_workpiece(self, workpiece: WorkPiece):
        uid = workpiece.uid
        size = workpiece.get_current_size()
        if not size or None in size:
            logger.debug(
                f"Skipping update for '{workpiece.name}'; "
                "size is not yet available."
            )
            return

        self._generation_id_map[uid] = self._generation_id_map.get(uid, 0) + 1
        current_generation_id = self._generation_id_map[uid]
        key = (id(self), uid)

        self.ops_generation_starting.send(self, workpiece=workpiece)
        self._ops_cache[uid] = (None, None)

        settings = {
            "power": self.power,
            "cut_speed": self.cut_speed,
            "travel_speed": self.travel_speed,
            "air_assist": self.air_assist,
            "pixels_per_mm": self.pixels_per_mm,
        }

        try:
            (
                workpiece_copy,
                laser_copy,
                opsproducer_copy,
                modifiers_copy,
                opstransformers_copy,
            ) = deepcopy(
                (
                    workpiece,
                    self.laser,
                    self.opsproducer,
                    self.modifiers,
                    self.opstransformers,
                )
            )
        except Exception as e:
            logger.error(
                f"Could not prepare data for subprocess: {e}", exc_info=True
            )
            self.ops_generation_finished.send(self, workpiece=workpiece)
            return

        def when_done_callback(task: Task):
            self._on_generation_complete(
                task, uid, current_generation_id
            )

        task_mgr.run_process(
            _execute_workstep_in_subprocess,
            workpiece_copy,
            opsproducer_copy,
            modifiers_copy,
            opstransformers_copy,
            laser_copy,
            settings,
            key=key,
            when_done=when_done_callback,
        )

    def _on_generation_complete(
        self, task: Task, uid: Any, task_generation_id: int
    ):
        if (
            uid not in self._generation_id_map
            or self._generation_id_map[uid] != task_generation_id
        ):
            return
        if uid not in self._workpieces:
            return

        workpiece = self._workpieces[uid]
        status = task.get_status()
        if status == "completed":
            try:
                result = task.result()
                if result is None:
                    self._ops_cache[uid] = (None, None)
                else:
                    self._ops_cache[uid] = result
                    logger.info(
                        f"WorkStep {self.name}: Successfully generated "
                        f"ops for {workpiece.name}."
                    )
            except Exception as e:
                logger.error(
                    f"WorkStep {self.name}: Error generating ops for "
                    f"{workpiece.name}: {e}",
                    exc_info=True,
                )
                self._ops_cache[uid] = (None, None)
        else:
            self._ops_cache[uid] = (None, None)

        self.ops_generation_finished.send(self, workpiece=workpiece)
        self.changed.send(self)

    def get_ops(self, workpiece: WorkPiece) -> Optional[Ops]:
        uid = workpiece.uid
        if not workpiece.size:
            return None

        raw_ops, pixel_size = self._ops_cache.get(uid, (None, None))
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
                ops.scale(
                    final_width_mm / traced_width_px,
                    final_height_mm / traced_height_px,
                )
        return ops

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
        current_uids = {wp.uid for wp in workpieces}
        existing_uids = set(self._workpieces.keys())
        for uid in existing_uids - current_uids:
            self._cleanup_workpiece(uid)
        for workpiece in workpieces:
            self.add_workpiece(workpiece)
        self.changed.send(self)

    def add_workpiece(self, workpiece: WorkPiece):
        uid = workpiece.uid
        if uid in self._workpieces:
            return
        self._workpieces[uid] = workpiece
        self._ops_cache[uid] = (None, None)
        self._generation_id_map[uid] = 0
        workpiece.size_changed.connect(self._request_workpiece_update)
        self.update_workpiece(workpiece)
        self.changed.send(self)

    def remove_workpiece(self, workpiece: WorkPiece):
        self._cleanup_workpiece(workpiece.uid)
        self.changed.send(self)

    def _cleanup_workpiece(self, uid: Any):
        if uid in self._workpiece_update_timers:
            GLib.source_remove(self._workpiece_update_timers.pop(uid))
        if uid in self._workpieces:
            workpiece = self._workpieces.pop(uid)
            workpiece.size_changed.disconnect(self._request_workpiece_update)
        if uid in self._ops_cache:
            del self._ops_cache[uid]
        if uid in self._generation_id_map:
            del self._generation_id_map[uid]

    def _request_workpiece_update(self, workpiece: WorkPiece):
        uid = workpiece.uid
        if uid in self._workpiece_update_timers:
            GLib.source_remove(self._workpiece_update_timers[uid])

        def _update_callback():
            if uid in self._workpieces:
                self.update_workpiece(self._workpieces[uid])
            if uid in self._workpiece_update_timers:
                del self._workpiece_update_timers[uid]
            return GLib.SOURCE_REMOVE

        timer_id = GLib.timeout_add(DEBOUNCE_DELAY_MS, _update_callback)
        self._workpiece_update_timers[uid] = timer_id

    def workpieces(self) -> List[WorkPiece]:
        return list(self._workpieces.values())

    def update_all_workpieces(self):
        for workpiece in self._workpieces.values():
            self.update_workpiece(workpiece)

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
        self.opstransformers = [
            Smooth(enabled=False, amount=20),
            Optimize(enabled=True),
        ]


class Contour(WorkStep):
    typelabel = _("Contour")

    def __init__(self, name=None, **kwargs):
        super().__init__(EdgeTracer(), name=name, **kwargs)
        self.opstransformers = [
            Smooth(enabled=False, amount=20),
            Optimize(enabled=True),
        ]


class Rasterize(WorkStep):
    typelabel = _("Raster Engrave")

    def __init__(self, name=None, **kwargs):
        super().__init__(Rasterizer(), name=name, **kwargs)
        self.opstransformers = [
            Optimize(enabled=True),
        ]
