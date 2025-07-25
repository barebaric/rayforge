from __future__ import annotations
import logging
from abc import ABC
from typing import Any, List, Dict, Tuple, Optional
from copy import deepcopy
from blinker import Signal
from gi.repository import GLib
from ..config import task_mgr
from ..tasker.task import Task
from ..modifier import Modifier, MakeTransparent, ToGrayscale
from ..opsproducer import OpsProducer, OutlineTracer, EdgeTracer, Rasterizer
from ..opstransformer import OpsTransformer, Optimize, Smooth
from .workpiece import WorkPiece
from .laser import Laser
from .ops import Ops
from .worksteprunner import run_workstep_in_subprocess


logger = logging.getLogger(__name__)

MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024
DEBOUNCE_DELAY_MS = 250  # Delay in milliseconds for ops regeneration


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

        self.ops_generation_starting.send(
            self, workpiece=workpiece, generation_id=current_generation_id
        )
        self._ops_cache[uid] = (None, None)

        settings = {
            "power": self.power,
            "cut_speed": self.cut_speed,
            "travel_speed": self.travel_speed,
            "air_assist": self.air_assist,
            "pixels_per_mm": self.pixels_per_mm,
        }

        def when_done_callback(task: Task):
            self._on_generation_complete(
                task, uid, current_generation_id
            )

        task_mgr.run_process(
            run_workstep_in_subprocess,
            workpiece.to_dict(),
            self.opsproducer.to_dict(),
            [m.to_dict() for m in self.modifiers],
            [o.to_dict() for o in self.opstransformers],
            self.laser.to_dict(),
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

        self.ops_generation_finished.send(
            self, workpiece=workpiece, generation_id=task_generation_id
        )
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
