import asyncio
import logging
import multiprocessing
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np
from blinker import Signal

from rayforge.core.ops.commands import MovingCommand

from ...camera.models.camera import Camera
from ...context import RayforgeContext, get_context
from ...pipeline.encoder.gcode import MachineCodeOpMap
from ...shared.tasker import task_mgr
from ..driver.driver import (
    Axis,
    DeviceState,
)
from ..transport import TransportStatus
from .dialect import GcodeDialect, get_dialect
from .laser import Laser
from .machine_hours import MachineHours
from .macro import Macro, MacroTrigger


if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.ops import Ops
    from ...core.varset import VarSet
    from ..driver.driver import Driver
    from .controller import MachineController


class Origin(Enum):
    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"


class JogDirection(Enum):
    """Visual direction for jog operations."""

    EAST = "east"
    WEST = "west"
    NORTH = "north"
    SOUTH = "south"
    UP = "up"
    DOWN = "down"


logger = logging.getLogger(__name__)

MACHINE_SPACE_WCS = "MACHINE"


def _raise_error(*args, **kwargs):
    raise RuntimeError("Cannot schedule from worker process")


class Machine:
    def __init__(self, context: RayforgeContext):
        logger.debug("Machine.__init__")
        self.id = str(uuid.uuid4())
        self.name: str = _("Default Machine")
        self.context = context

        if multiprocessing.current_process().daemon:
            # This is the worker process, do not allow scheduling signals.
            self._scheduler = _raise_error
        else:
            # This is the main process, use the real scheduler.
            self._scheduler = task_mgr.schedule_on_main_thread

        # Signals
        self.changed = Signal()
        self.settings_error = Signal()
        self.settings_updated = Signal()
        self.setting_applied = Signal()
        self.connection_status_changed = Signal()
        self.state_changed = Signal()
        self.job_finished = Signal()
        self.command_status_changed = Signal()
        self.wcs_updated = Signal()

        self.connection_status: TransportStatus = TransportStatus.DISCONNECTED
        self.device_state: DeviceState = DeviceState()

        self.driver_name: Optional[str] = None
        self.driver_args: Dict[str, Any] = {}
        self.precheck_error: Optional[str] = None

        self.auto_connect: bool = True
        self.home_on_start: bool = False
        self.clear_alarm_on_connect: bool = False
        self.single_axis_homing_enabled: bool = True
        self.dialect_uid: str = "grbl"
        self._hydrated_dialect: Optional[GcodeDialect] = None
        self.gcode_precision: int = 3
        self.supports_arcs: bool = True
        self.arc_tolerance: float = 0.03
        self.hookmacros: Dict[MacroTrigger, Macro] = {}
        self.macros: Dict[str, Macro] = {}
        self.heads: List[Laser] = []
        self._heads_ref_for_pyreverse: Laser
        self.cameras: List[Camera] = []
        self._cameras_ref_for_pyreverse: Camera
        self.max_travel_speed: int = 3000  # in mm/min
        self.max_cut_speed: int = 1000  # in mm/min
        self.acceleration: int = 1000  # in mm/s²
        self._axis_extents: Tuple[float, float] = 200.0, 200.0
        self._work_margins: Tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._soft_limits: Optional[Tuple[float, float, float, float]] = None
        self.origin: Origin = Origin.BOTTOM_LEFT
        self.reverse_x_axis: bool = False
        self.reverse_y_axis: bool = False
        self.reverse_z_axis: bool = False
        self.soft_limits_enabled: bool = True
        self.wcs_origin_is_workarea_origin: bool = False
        self._settings_lock = asyncio.Lock()

        # Work Coordinate System (WCS) State
        # We default to standard G-code names for convenience, but the logic
        # is agnostic. Any key in wcs_offsets is considered a mutable WCS.
        # Any key NOT in wcs_offsets is considered an immutable/absolute system
        # with (0,0,0) offset.
        self.active_wcs: str = "G54"
        self.wcs_offsets: Dict[str, Tuple[float, float, float]] = {
            "G54": (0.0, 0.0, 0.0),
            "G55": (0.0, 0.0, 0.0),
            "G56": (0.0, 0.0, 0.0),
            "G57": (0.0, 0.0, 0.0),
            "G58": (0.0, 0.0, 0.0),
            "G59": (0.0, 0.0, 0.0),
        }

        self.machine_hours: MachineHours = MachineHours()
        self.machine_hours.changed.connect(self._on_machine_hours_changed)

        # Connect to dialect manager to detect dialect changes
        self.context.dialect_mgr.dialects_changed.connect(
            self._on_dialects_changed
        )

        self.add_head(Laser())

    @property
    def controller(self) -> "MachineController":
        """
        Dynamically retrieves the controller for this machine from the
        MachineManager. This enables lazy instantiation.
        """
        return self.context.machine_mgr.get_controller(self.id)

    @property
    def driver(self) -> "Driver":
        """Property to access the driver through the controller."""
        return self.controller.driver

    def _connect_controller_signals(self, controller: "MachineController"):
        """
        Connects this machine's signal proxies to the controller's signals.
        This is now called by the MachineManager when the controller is
        created.
        """
        controller.connection_status_changed.connect(
            self.connection_status_changed.send
        )
        controller.state_changed.connect(self.state_changed.send)
        controller.job_finished.connect(self.job_finished.send)
        controller.command_status_changed.connect(
            self.command_status_changed.send
        )
        controller.wcs_updated.connect(self.wcs_updated.send)

    @property
    def supported_wcs(self) -> List[str]:
        """
        Returns a sorted list of supported mutable Work Coordinate Systems.
        """
        return sorted(list(self.wcs_offsets.keys()))

    @property
    def machine_space_wcs(self) -> str:
        """
        Returns the identifier for the machine space coordinate system.
        Delegates to the controller's driver property.
        """
        return self.controller.machine_space_wcs

    @property
    def machine_space_wcs_display_name(self) -> str:
        """
        Returns the display name for the machine space coordinate system.
        Delegates to the controller's driver property.
        """
        return self.controller.machine_space_wcs_display_name

    async def connect(self):
        """Public method to connect the driver."""
        await self.controller.connect()

    async def disconnect(self):
        """Public method to disconnect the driver."""
        await self.controller.disconnect()

    async def shutdown(self):
        """
        Gracefully shuts down the machine's active driver and resources.
        """
        logger.info(f"Shutting down machine '{self.name}' (id:{self.id})")
        # We only shut down the controller if it exists to avoid creating
        # it during shutdown if it wasn't used.
        try:
            # Check for existence via manager without triggering creation
            # if possible, or just call the manager's shutdown for this ID.
            # But simpler here is to let the manager handle bulk shutdown.
            # If we are shutting down a specific machine instance:
            if self.id in self.context.machine_mgr.controllers:
                await self.controller.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down controller: {e}")

        self.context.dialect_mgr.dialects_changed.disconnect(
            self._on_dialects_changed
        )

    def _on_dialects_changed(self, sender=None, **kwargs):
        """
        Callback when dialects are updated.
        Sends machine's changed signal to trigger recalculation.
        """
        self.changed.send(self)

    def is_connected(self) -> bool:
        """
        Checks if the machine's driver is currently connected to the device.

        Returns:
            True if connected, False otherwise.
        """
        return self.connection_status == TransportStatus.CONNECTED

    async def select_tool(self, index: int):
        """Sends a command to the driver to select a tool."""
        await self.controller.select_tool(index)

    def set_name(self, name: str):
        self.name = str(name)
        self.changed.send(self)

    def set_driver(self, driver_cls: Type["Driver"], args=None):
        new_driver_name = driver_cls.__name__
        new_args = args or {}
        if (
            self.driver_name == new_driver_name
            and self.driver_args == new_args
        ):
            return

        self.driver_name = new_driver_name
        self.driver_args = new_args
        task_mgr.add_coroutine(
            self.controller._rebuild_driver_instance,
            key=(self.id, "rebuild-driver"),
        )

    def set_driver_args(self, args=None):
        new_args = args or {}
        if self.driver_args == new_args:
            return

        self.driver_args = new_args
        task_mgr.add_coroutine(
            self.controller._rebuild_driver_instance,
            key=(self.id, "rebuild-driver"),
        )

    @property
    def dialect(self) -> "GcodeDialect":
        """Get the current dialect instance for this machine."""
        if self._hydrated_dialect:
            return self._hydrated_dialect
        return get_dialect(self.dialect_uid)

    def hydrate(self):
        """
        Fetches the current dialect from the registry and stores it internally.
        This ensures that when serialized, the machine carries the full
        dialect definition.
        """
        self._hydrated_dialect = get_dialect(self.dialect_uid)

    def set_dialect_uid(self, dialect_uid: str):
        if self.dialect_uid == dialect_uid:
            return
        self.dialect_uid = dialect_uid
        self.changed.send(self)

    def set_gcode_precision(self, precision: int):
        if self.gcode_precision == precision:
            return
        self.gcode_precision = precision
        self.changed.send(self)

    def set_arc_tolerance(self, tolerance: float):
        if self.arc_tolerance == tolerance:
            return
        self.arc_tolerance = tolerance
        self.changed.send(self)

    def set_home_on_start(self, home_on_start: bool = True):
        self.home_on_start = home_on_start
        self.changed.send(self)

    def set_clear_alarm_on_connect(self, clear_alarm: bool = True):
        if self.clear_alarm_on_connect == clear_alarm:
            return
        self.clear_alarm_on_connect = clear_alarm
        self.changed.send(self)

    def set_single_axis_homing_enabled(self, enabled: bool = True):
        if self.single_axis_homing_enabled == enabled:
            return
        self.single_axis_homing_enabled = enabled
        self.changed.send(self)

    def set_max_travel_speed(self, speed: int):
        self.max_travel_speed = speed
        self.changed.send(self)

    def set_max_cut_speed(self, speed: int):
        self.max_cut_speed = speed
        self.changed.send(self)

    def set_acceleration(self, acceleration: int):
        self.acceleration = acceleration
        self.changed.send(self)

    @property
    def axis_extents(self) -> Tuple[float, float]:
        """The full range of machine axis movement (width, height)."""
        return self._axis_extents

    def _clamp_soft_limits(self):
        """Clamp soft limits to axis extents. Returns True if clamped."""
        if self._soft_limits is None:
            return False
        w, h = self._axis_extents
        x_min, y_min, x_max, y_max = self._soft_limits
        clamped = (
            max(0.0, min(x_min, w)),
            max(0.0, min(y_min, h)),
            max(0.0, min(x_max, w)),
            max(0.0, min(y_max, h)),
        )
        if clamped != self._soft_limits:
            self._soft_limits = clamped
            return True
        return False

    def set_axis_extents(self, width: float, height: float):
        if self._axis_extents == (width, height):
            return
        self._axis_extents = (width, height)
        ml, mt, mr, mb = self._work_margins
        ml = min(ml, width - 1)
        mr = min(mr, width - ml - 1)
        mt = min(mt, height - 1)
        mb = min(mb, height - mt - 1)
        self._work_margins = (ml, mt, mr, mb)
        self._clamp_soft_limits()
        self.changed.send(self)

    def set_dimensionssss(self, width: float, height: float):
        self.set_axis_extents(width, height)

    @property
    def work_margins(self) -> Tuple[float, float, float, float]:
        """
        The margins around the work area (left, top, right, bottom).
        These are positive distances from the axis extents edges.
        """
        return self._work_margins

    def set_work_margins(
        self, left: float, top: float, right: float, bottom: float
    ):
        new_margins = (left, top, right, bottom)
        if self._work_margins == new_margins:
            return
        self._work_margins = new_margins
        self._soft_limits = None
        self.changed.send(self)

    @property
    def work_area(self) -> Tuple[float, float, float, float]:
        """
        The usable work area within the axis extents (x, y, w, h).
        Computed from axis_extents and work_margins.
        """
        ml, mt, mr, mb = self._work_margins
        w = self._axis_extents[0] - ml - mr
        h = self._axis_extents[1] - mt - mb
        return (ml, mt, max(1.0, w), max(1.0, h))

    @property
    def soft_limits(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Configurable safety bounds for jogging (x_min, y_min, x_max, y_max).
        None means use work_area bounds.
        """
        return self._soft_limits

    def set_soft_limits(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ):
        w, h = self._axis_extents
        clamped = (
            max(0.0, min(x_min, w)),
            max(0.0, min(y_min, h)),
            max(0.0, min(x_max, w)),
            max(0.0, min(y_max, h)),
        )
        if self._soft_limits == clamped:
            return
        self._soft_limits = clamped
        self.changed.send(self)

    def set_origin(self, origin: Origin):
        self.origin = origin
        self.changed.send(self)

    def set_reverse_x_axis(self, is_reversed: bool):
        """Sets if the X-axis coordinate display is inverted."""
        if self.reverse_x_axis == is_reversed:
            return
        self.reverse_x_axis = is_reversed
        self.changed.send(self)

    def set_reverse_y_axis(self, is_reversed: bool):
        """Sets if the Y-axis coordinate display is inverted."""
        if self.reverse_y_axis == is_reversed:
            return
        self.reverse_y_axis = is_reversed
        self.changed.send(self)

    def set_reverse_z_axis(self, is_reversed: bool):
        """Sets if the Z-axis direction is reversed."""
        if self.reverse_z_axis == is_reversed:
            return
        self.reverse_z_axis = is_reversed
        self.changed.send(self)

    def set_wcs_origin_is_workarea_origin(self, value: bool):
        """Sets if the workarea origin should be treated as coordinate zero."""
        if self.wcs_origin_is_workarea_origin == value:
            return
        self.wcs_origin_is_workarea_origin = value
        self.changed.send(self)

    @property
    def y_axis_down(self) -> bool:
        """
        True if the Y coordinate decreases as the head moves away from the
        user (i.e., origin is at the top). Used for G-code generation.
        """
        return self.origin in (Origin.TOP_LEFT, Origin.TOP_RIGHT)

    @property
    def x_axis_right(self) -> bool:
        """
        True if the X coordinate decreases as the head moves left
        (i.e., origin is on the right). Used for G-code generation.
        """
        return self.origin in (Origin.TOP_RIGHT, Origin.BOTTOM_RIGHT)

    def calculate_jog(self, direction: JogDirection, distance: float) -> float:
        """
        Calculate the signed coordinate delta for a jog operation based on a
        visual direction.

        Args:
            direction: The visual direction for the jog.
            distance: The positive distance for the jog.

        Returns:
            The signed delta for the specified direction, taking into account
            origin position and reverse axis settings.
        """
        if direction == JogDirection.EAST:
            delta = -distance if self.x_axis_right else distance
            return -delta if self.reverse_x_axis else delta
        if direction == JogDirection.WEST:
            delta = distance if self.x_axis_right else -distance
            return -delta if self.reverse_x_axis else delta
        if direction == JogDirection.NORTH:
            delta = -distance if self.y_axis_down else distance
            return -delta if self.reverse_y_axis else delta
        if direction == JogDirection.SOUTH:
            delta = distance if self.y_axis_down else -distance
            return -delta if self.reverse_y_axis else delta
        if direction == JogDirection.UP:
            return -distance if self.reverse_z_axis else distance
        if direction == JogDirection.DOWN:
            return distance if self.reverse_z_axis else -distance
        return 0.0

    def set_soft_limits_enabled(self, enabled: bool):
        """Enable or disable soft limits for jog operations."""
        self.soft_limits_enabled = enabled
        self.changed.send(self)

    def get_current_position(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get the current work position of the machine."""
        return self.device_state.work_pos

    def get_soft_limits(self) -> Tuple[float, float, float, float]:
        """Get the soft limits as (x_min, y_min, x_max, y_max)."""
        if self._soft_limits is not None:
            x_min, y_min, x_max, y_max = self._soft_limits
            if self.reverse_x_axis:
                x_min, x_max = -x_max, -x_min
            if self.reverse_y_axis:
                y_min, y_max = -y_max, -y_min
            return (float(x_min), float(y_min), float(x_max), float(y_max))

        w, h = float(self._axis_extents[0]), float(self._axis_extents[1])

        x_min = -w if self.reverse_x_axis else 0.0
        x_max = 0.0 if self.reverse_x_axis else w
        y_min = -h if self.reverse_y_axis else 0.0
        y_max = 0.0 if self.reverse_y_axis else h

        return (x_min, y_min, x_max, y_max)

    def would_jog_exceed_limits(self, axis: Axis, distance: float) -> bool:
        """
        Check if a jog operation would exceed soft limits.

        Note: The `distance` argument must be the final, signed coordinate
        delta that will be sent to the machine.
        """
        if not self.soft_limits_enabled:
            return False

        current_pos = self.device_state.machine_pos
        x_pos, y_pos, z_pos = current_pos
        x_min, y_min, x_max, y_max = self.get_soft_limits()

        # Check X axis
        if axis & Axis.X:
            if x_pos is None:
                return False  # Cannot check limits if position is unknown
            new_x = x_pos + distance
            if new_x < x_min or new_x > x_max:
                return True

        # Check Y axis
        if axis & Axis.Y:
            if y_pos is None:
                return False  # Cannot check limits if position is unknown
            new_y = y_pos + distance
            if new_y < y_min or new_y > y_max:
                return True

        # Note: Z-axis soft limits are not currently implemented

        return False

    def _adjust_jog_distance_for_limits(
        self, axis: Axis, distance: float
    ) -> float:
        """Adjust jog distance to stay within soft limits."""
        if not self.soft_limits_enabled:
            return distance

        current_pos = self.device_state.machine_pos
        x_pos, y_pos, z_pos = current_pos
        x_min, y_min, x_max, y_max = self.get_soft_limits()
        adjusted_distance = distance

        # Check X axis
        if axis & Axis.X:
            if x_pos is None:
                return distance  # Cannot adjust if position is unknown
            new_x = x_pos + distance
            if new_x < x_min:
                adjusted_distance = x_min - x_pos
            elif new_x > x_max:
                adjusted_distance = x_max - x_pos

        # Check Y axis
        if axis & Axis.Y:
            if y_pos is None:
                return distance  # Cannot adjust if position is unknown
            new_y = y_pos + distance
            if new_y < y_min:
                adjusted_distance = y_min - y_pos
            elif new_y > y_max:
                adjusted_distance = y_max - y_pos

        return adjusted_distance

    @property
    def reports_granular_progress(self) -> bool:
        """Check if the machine's driver reports granular progress."""
        return self.controller.reports_granular_progress

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        """Check if the machine's driver supports homing for the given axis."""
        return self.controller.can_home(axis)

    async def home(self, axes=None):
        """Homes the specified axes or all axes if none specified."""
        await self.controller.home(axes)

    async def jog(self, deltas: Dict[Axis, float], speed: int):
        """
        Jogs the machine along specified axes.

        Args:
            deltas: Dictionary mapping Axis enum members to distances in mm.
            speed: Speed in mm/min.
        """
        await self.controller.jog(deltas, speed)

    async def run_raw(self, gcode: str):
        """Executes a raw G-code string on the machine."""
        await self.controller.run_raw(gcode)

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """Check if machine's supports jogging for the given axis."""
        return self.controller.can_jog(axis)

    def add_head(self, head: Laser):
        self.heads.append(head)
        head.changed.connect(self._on_head_changed)
        self.changed.send(self)

    def get_head_by_uid(self, uid: str) -> Optional[Laser]:
        for head in self.heads:
            if head.uid == uid:
                return head
        return None

    def get_default_head(self) -> Laser:
        """Returns the first laser head, or raises an error if none exist."""
        if not self.heads:
            raise ValueError("Machine has no laser heads configured.")
        return self.heads[0]

    def remove_head(self, head: Laser):
        head.changed.disconnect(self._on_head_changed)
        self.heads.remove(head)
        self.changed.send(self)

    def _on_head_changed(self, head, *args):
        self.changed.send(self)

    def add_camera(self, camera: Camera):
        self.cameras.append(camera)
        camera.changed.connect(self._on_camera_changed)
        self.changed.send(self)

    def remove_camera(self, camera: Camera):
        camera.changed.disconnect(self._on_camera_changed)
        self.cameras.remove(camera)
        self.changed.send(self)

    def _on_camera_changed(self, camera, *args):
        self.changed.send(self)

    def _on_machine_hours_changed(self, machine_hours, *args):
        """
        Handle machine hours changes and propagate to machine changed
        signal.
        """
        self._scheduler(self.changed.send, self)

    def add_machine_hours(self, hours: float) -> None:
        """
        Add hours to the machine's total hours and all counters.

        Args:
            hours: Hours to add (can be fractional).
        """
        self.machine_hours.add_hours(hours)

    def get_machine_hours(self) -> MachineHours:
        """Get the machine hours tracker."""
        return self.machine_hours

    def add_macro(self, macro: Macro):
        """Adds a macro and notifies listeners."""
        if macro.uid in self.macros:
            return
        self.macros[macro.uid] = macro
        self.changed.send(self)

    def remove_macro(self, macro_uid: str):
        """Removes a macro and notifies listeners."""
        if macro_uid not in self.macros:
            return
        del self.macros[macro_uid]
        self.changed.send(self)

    def can_frame(self):
        for head in self.heads:
            if head.frame_power_percent:
                return True
        return False

    def can_focus(self):
        for head in self.heads:
            if head.focus_power_percent:
                return True
        return False

    def validate_driver_setup(self) -> Tuple[bool, Optional[str]]:
        """
        Validates the machine's driver arguments against the driver's setup
        VarSet. Delegates to the controller.

        Returns:
            A tuple of (is_valid, error_message).
        """
        return self.controller.validate_driver_setup()

    async def set_power(
        self, head: Optional["Laser"] = None, percent: float = 0.0
    ) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control. If None, uses the default head.
            percent: Power percentage (0-1.0). 0 disables power.
        """
        await self.controller.set_power(head, percent)

    def get_active_wcs_offset(self) -> Tuple[float, float, float]:
        """
        Returns the (x, y, z) offset for the currently active WCS.
        If the active_wcs is not in the known offsets dictionary, it assumes
        an absolute coordinate system with zero offset.
        """
        return self.wcs_offsets.get(self.active_wcs, (0.0, 0.0, 0.0))

    def get_workarea_origin_offset(self) -> Tuple[float, float]:
        """
        Returns the position of the workarea origin in WORLD space.

        The workarea origin is at the corner specified by the machine's origin
        setting. This is used to convert from MACHINE coordinates to
        workarea-relative coordinates.

        Margins are: (left, top, right, bottom)

        Returns:
            Tuple of (x, y) in WORLD space.
        """
        ml, mt, mr, mb = self._work_margins
        width, height = self._axis_extents

        if self.origin == Origin.BOTTOM_LEFT:
            return (ml, mb)
        elif self.origin == Origin.TOP_LEFT:
            return (ml, height - mt)
        elif self.origin == Origin.BOTTOM_RIGHT:
            return (width - mr, mb)
        else:  # TOP_RIGHT
            return (width - mr, height - mt)

    def get_reference_offset(self) -> Tuple[float, float, float]:
        """
        Returns the offset for converting from MACHINE to REFERENCE coords.

        REFERENCE coordinates are what the user sees in the UI. When
        wcs_origin_is_workarea_origin is False, this returns the active WCS
        offset. When True, this returns the workarea origin offset.

        Returns:
            Tuple of (x, y, z) offset in MACHINE space.
        """
        if self.wcs_origin_is_workarea_origin:
            x, y = self.get_workarea_origin_offset()
            return (x, y, 0.0)
        else:
            return self.get_active_wcs_offset()

    def get_visual_wcs_offset(self) -> Tuple[float, float]:
        """
        Returns the WCS offset transformed to visual coordinates.

        Applies axis reversal to the WCS offset. The caller (UI layer)
        handles origin corner transformation based on canvas dimensions.

        Returns:
            Tuple of (x, y) with axis reversal applied.
        """
        wcs_x, wcs_y, _ = self.get_active_wcs_offset()

        if self.reverse_x_axis:
            wcs_x = -wcs_x
        if self.reverse_y_axis:
            wcs_y = -wcs_y

        return (wcs_x, wcs_y)

    def get_visual_extent_frame(self) -> Tuple[float, float, float, float]:
        """
        Returns the extent frame rectangle (x, y, width, height) in visual
        coordinates relative to the work area origin.

        The work area is at (0, 0) in its own coordinate system.
        The extent frame is positioned at (-margin_left, -margin_bottom)
        relative to the work area origin.

        Returns:
            Tuple of (x, y, width, height) where x,y is the frame position
            relative to the work area origin (0,0).
        """
        ml, mb = self._work_margins[0], self._work_margins[3]
        extent_w, extent_h = self._axis_extents
        return (float(-ml), float(-mb), float(extent_w), float(extent_h))

    def has_custom_work_area(self) -> bool:
        """
        Returns True if any margin is non-zero.
        """
        ml, mt, mr, mb = self._work_margins
        return ml != 0 or mt != 0 or mr != 0 or mb != 0

    def set_active_wcs(self, wcs: str):
        """Sets the active WCS and notifies listeners."""
        if wcs != self.active_wcs:
            self.active_wcs = wcs
            self.changed.send(self)

    async def set_work_origin(
        self, x: float, y: float, z: float, wcs_slot: Optional[str] = None
    ):
        """
        Sets the work origin at the specified machine coordinates.

        Args:
            x: X-coordinate in machine space.
            y: Y-coordinate in machine space.
            z: Z-coordinate in machine space.
            wcs_slot: The WCS slot to update (e.g. "G54"). Defaults to active.
        """
        await self.controller.set_work_origin(x, y, z, wcs_slot)

    async def set_work_origin_here(
        self, axes: Axis, wcs_slot: Optional[str] = None
    ):
        """
        Sets the work origin for the specified axes to the current machine
        position.

        Args:
            axes: Flag combination of axes to set (e.g. Axis.X | Axis.Y).
            wcs_slot: The WCS slot to update (e.g. "G54"). Defaults to active.
        """
        await self.controller.set_work_origin_here(axes, wcs_slot)

    async def sync_wcs_from_device(self):
        """Queries the device for current WCS offsets and updates state."""
        await self.controller.sync_wcs_from_device()

    async def sync_active_wcs_from_device(self):
        """Queries the device for its active WCS and updates state."""
        await self.controller.sync_active_wcs_from_device()

    def _prepare_ops_for_encoding(self, ops: "Ops") -> "Ops":
        """
        Prepares an Ops object for encoding by applying machine-specific
        coordinate transformations.

        The pipeline produces ops in machine coordinates. This method
        converts them to command coordinates for the G-code output.

        Args:
            ops: The Ops object to prepare.

        Returns:
            A transformed Ops object ready for encoding.
        """
        ops_for_encoder = ops.copy()

        # First, apply origin transform to convert from world coords
        # (bottom-left origin, Y-up) to machine coords.
        # If Origin is BOTTOM_LEFT and axes are not reversed, the internal
        # coordinate system matches the machine's.
        needs_transform = (
            self.origin != Origin.BOTTOM_LEFT
            or self.reverse_x_axis
            or self.reverse_y_axis
        )

        if needs_transform:
            width, height = self._axis_extents

            transform = np.identity(4)

            # --- Y-Axis Transformation ---
            if self.y_axis_down:  # Origin is TOP_LEFT or TOP_RIGHT
                if self.reverse_y_axis:
                    transform[1, 3] = -float(height)
                else:
                    transform[1, 1] = -1.0
                    transform[1, 3] = float(height)
            elif self.reverse_y_axis:
                transform[1, 1] = -1.0

            # --- X-Axis Transformation ---
            if self.x_axis_right:  # Origin is TOP_RIGHT or BOTTOM_RIGHT
                if self.reverse_x_axis:
                    transform[0, 3] = -float(width)
                else:
                    transform[0, 0] = -1.0
                    transform[0, 3] = float(width)
            elif self.reverse_x_axis:
                transform[0, 0] = -1.0

            ops_for_encoder.transform(transform)

        # Now ops are in machine coords. Convert to command coords.
        if self.wcs_origin_is_workarea_origin:
            # Workarea origin is treated as coordinate zero.
            # Convert from machine coords to workarea-relative coords.
            ml, mt, mr, mb = self._work_margins
            width, height = self._axis_extents

            # Calculate workarea origin position in machine coords
            # For reversed axes, the transform flips the sign
            if self.x_axis_right:
                if self.reverse_x_axis:
                    x_offset = -mr
                else:
                    x_offset = mr
            else:
                if self.reverse_x_axis:
                    x_offset = -ml
                else:
                    x_offset = ml

            if self.y_axis_down:
                if self.reverse_y_axis:
                    y_offset = -mt
                else:
                    y_offset = mt
            else:
                if self.reverse_y_axis:
                    y_offset = -mb
                else:
                    y_offset = mb

            for command in ops_for_encoder.commands:
                if isinstance(command, MovingCommand):
                    base_end = command.end or (0, 0, 0)
                    result_x = base_end[0] - x_offset
                    result_y = base_end[1] - y_offset
                    result_z = base_end[2]
                    # For right/top origins, positive axis points toward
                    # smaller machine coords, so negate the result
                    command.end = (result_x, result_y, result_z)
        else:
            # Standard WCS mode: subtract WCS offset to convert from
            # machine coordinates to command coordinates.
            wcs_x, wcs_y, wcs_z = self.get_active_wcs_offset()
            for command in ops_for_encoder.commands:
                if isinstance(command, MovingCommand):
                    base_end = command.end or (0, 0, 0)
                    command.end = (
                        base_end[0] - wcs_x,
                        base_end[1] - wcs_y,
                        base_end[2] - wcs_z,
                    )

        return ops_for_encoder

    def encode_ops(
        self, ops: "Ops", doc: "Doc"
    ) -> Tuple[str, "MachineCodeOpMap"]:
        """
        Encodes an Ops object into machine code (G-code) and a corresponding
        operation map. This method is safe to run in a worker process as it
        uses static driver instantiation to get the encoder.

        Args:
            ops: The Ops object to encode.
            doc: The document context for the job.

        Returns:
            A tuple containing:
            - A string of machine code (G-code).
            - A MachineCodeOpMap object.
        """
        # 1. Prepare ops (pure math)
        ops_for_encoder = self._prepare_ops_for_encoding(ops)

        # 2. Instantiate the correct encoder via the driver factory
        from ..driver import get_driver_cls
        from ..driver.dummy import NoDeviceDriver

        if self.driver_name:
            try:
                driver_cls = get_driver_cls(self.driver_name)
            except (ValueError, ImportError):
                # Fallback if driver class is missing
                driver_cls = NoDeviceDriver
        else:
            driver_cls = NoDeviceDriver

        encoder = driver_cls.create_encoder(self)

        # 3. Perform encoding
        return encoder.encode(ops_for_encoder, self, doc)

    def refresh_settings(self):
        """Public API for the UI to request a settings refresh."""
        task_mgr.add_coroutine(
            lambda ctx: self.controller._read_from_device(),
            key=(self.id, "device-settings-read"),
        )

    def apply_setting(self, key: str, value: Any):
        """Public API for the UI to apply a single setting."""
        task_mgr.add_coroutine(
            lambda ctx: self.controller._write_setting_to_device(key, value),
            key=(
                self.id,
                "device-settings-write",
                key,
            ),  # Key includes setting key for uniqueness
        )

    def get_setting_vars(self) -> List["VarSet"]:
        """
        Gets the setting definitions from the machine's active driver
        as a VarSet.
        """
        return self.controller.get_setting_vars()

    def to_dict(self, include_frozen_dialect: bool = True) -> Dict[str, Any]:
        data = {
            "machine": {
                "name": self.name,
                "driver": self.driver_name,
                "driver_args": self.driver_args,
                "auto_connect": self.auto_connect,
                "clear_alarm_on_connect": self.clear_alarm_on_connect,
                "home_on_start": self.home_on_start,
                "single_axis_homing_enabled": self.single_axis_homing_enabled,
                "dialect_uid": self.dialect_uid,
                "active_wcs": self.active_wcs,
                "wcs_offsets": self.wcs_offsets,
                "supports_arcs": self.supports_arcs,
                "arc_tolerance": self.arc_tolerance,
                "axis_extents": list(self._axis_extents),
                "work_margins": list(self._work_margins),
                "soft_limits": list(self._soft_limits)
                if self._soft_limits
                else None,
                "origin": self.origin.value,
                "reverse_x_axis": self.reverse_x_axis,
                "reverse_y_axis": self.reverse_y_axis,
                "reverse_z_axis": self.reverse_z_axis,
                "wcs_origin_is_workarea_origin": (
                    self.wcs_origin_is_workarea_origin
                ),
                "heads": [head.to_dict() for head in self.heads],
                "cameras": [camera.to_dict() for camera in self.cameras],
                "hookmacros": {
                    trigger.name: macro.to_dict()
                    for trigger, macro in self.hookmacros.items()
                },
                "macros": {
                    uid: macro.to_dict() for uid, macro in self.macros.items()
                },
                "speeds": {
                    "max_cut_speed": self.max_cut_speed,
                    "max_travel_speed": self.max_travel_speed,
                    "acceleration": self.acceleration,
                },
                "gcode": {
                    "gcode_precision": self.gcode_precision,
                },
                "machine_hours": self.machine_hours.to_dict(),
            }
        }
        if include_frozen_dialect and self._hydrated_dialect:
            data["machine"]["frozen_dialect"] = (
                self._hydrated_dialect.to_dict()
            )
        return data

    @staticmethod
    def _migrate_legacy_hooks_to_dialect(
        hook_data: Dict[str, Any],
        current_dialect_uid: str,
        machine_name: str,
        context: RayforgeContext,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Checks for legacy JOB_START/JOB_END hooks and migrates them to a
        new custom dialect.

        Returns:
            A tuple containing the (potentially new) dialect UID and the
            cleaned hook_data dictionary.
        """
        job_start_hook_data = hook_data.get("JOB_START")
        job_end_hook_data = hook_data.get("JOB_END")

        if not job_start_hook_data and not job_end_hook_data:
            # No migration needed
            return current_dialect_uid, hook_data

        logger.info(
            f"Migrating JOB_START/JOB_END hooks to a new custom dialect "
            f"for machine '{machine_name}'."
        )

        try:
            base_dialect = get_dialect(current_dialect_uid)
        except ValueError:
            logger.warning(
                f"Could not find base dialect '{current_dialect_uid}' for "
                f"migration. Using 'grbl' as a fallback."
            )
            base_dialect = get_dialect("grbl")

        new_label = _("{label} (for {machine_name})").format(
            label=base_dialect.label,
            machine_name=machine_name,
        )
        new_dialect = base_dialect.copy_as_custom(new_label=new_label)

        if job_start_hook_data:
            new_dialect.preamble = job_start_hook_data.get("code", [])
        if job_end_hook_data:
            new_dialect.postscript = job_end_hook_data.get("code", [])

        # Add the new dialect to the manager (registers and saves it)
        context.dialect_mgr.add_dialect(new_dialect)

        # Clean up the old hook data so it isn't loaded or re-saved
        new_hook_data = hook_data.copy()
        new_hook_data.pop("JOB_START", None)
        new_hook_data.pop("JOB_END", None)

        # Return the new dialect's UID and the cleaned hook data
        return new_dialect.uid, new_hook_data

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], is_inert: bool = False
    ) -> "Machine":
        context = get_context()
        ma = cls(context)
        ma_data = data.get("machine", {})
        ma.id = ma_data.get("id", ma.id)
        ma.name = ma_data.get("name", ma.name)
        ma.driver_name = ma_data.get("driver")
        ma.driver_args = ma_data.get("driver_args", {})
        ma.auto_connect = ma_data.get("auto_connect", ma.auto_connect)
        ma.clear_alarm_on_connect = ma_data.get(
            "clear_alarm_on_connect",
            ma.clear_alarm_on_connect,
        )
        ma.home_on_start = ma_data.get("home_on_start", ma.home_on_start)
        ma.single_axis_homing_enabled = ma_data.get(
            "single_axis_homing_enabled",
            ma.single_axis_homing_enabled,
        )

        dialect_uid = ma_data.get("dialect_uid")
        if not dialect_uid:  # backward compatibility
            dialect_uid = ma_data.get("dialect", "grbl").lower()

        hook_data = ma_data.get("hookmacros", {})

        # Run the migration logic, which may update the dialect_uid and
        # hook_data
        dialect_uid, hook_data = cls._migrate_legacy_hooks_to_dialect(
            hook_data, dialect_uid, ma.name, context
        )

        ma.dialect_uid = dialect_uid
        ma.active_wcs = ma_data.get("active_wcs", ma.active_wcs)
        if "wcs_offsets" in ma_data:
            ma.wcs_offsets = ma_data["wcs_offsets"]

        ma._axis_extents = tuple(ma_data.get("dimensions", ma._axis_extents))
        if "axis_extents" in ma_data:
            ma._axis_extents = tuple(ma_data["axis_extents"])

        if "work_margins" in ma_data:
            ma._work_margins = tuple(ma_data["work_margins"])
        elif "offsets" in ma_data:
            ox, oy = ma_data["offsets"]
            ma._work_margins = (ox, 0, 0, oy)

        if "soft_limits" in ma_data and ma_data["soft_limits"] is not None:
            ma._soft_limits = tuple(ma_data["soft_limits"])

        origin_value = ma_data.get("origin", None)
        if origin_value is not None:
            ma.origin = Origin(origin_value)
        else:  # Legacy support for y_axis_down
            ma.origin = (
                Origin.BOTTOM_LEFT
                if ma_data.get("y_axis_down", False) is False
                else Origin.TOP_LEFT
            )

        # Load new reverse axis settings if they exist
        ma.reverse_x_axis = ma_data.get("reverse_x_axis", False)
        ma.reverse_y_axis = ma_data.get("reverse_y_axis", False)
        ma.reverse_z_axis = ma_data.get("reverse_z_axis", False)

        # Migrate from old "negative" settings if present
        if "x_axis_negative" in ma_data:
            logger.info("Migrating legacy 'x_axis_negative' setting.")
            ma.reverse_x_axis = ma_data["x_axis_negative"]
        if "y_axis_negative" in ma_data:
            logger.info("Migrating legacy 'y_axis_negative' setting.")
            ma.reverse_y_axis = ma_data["y_axis_negative"]

        ma.soft_limits_enabled = ma_data.get(
            "soft_limits_enabled", ma.soft_limits_enabled
        )

        ma.wcs_origin_is_workarea_origin = ma_data.get(
            "wcs_origin_is_workarea_origin", False
        )

        # Deserialize remaining hookmacros from the (potentially cleaned) data
        for trigger_name, macro_data in hook_data.items():
            try:
                trigger = MacroTrigger[trigger_name]
                ma.hookmacros[trigger] = Macro.from_dict(macro_data)
            except KeyError:
                logger.warning(
                    f"Skipping unknown hook trigger '{trigger_name}'"
                )

        macro_data = ma_data.get("macros", {})
        for uid, macro_data in macro_data.items():
            macro_data["uid"] = uid  # Ensure UID is consistent with key
            ma.macros[uid] = Macro.from_dict(macro_data)

        ma.heads = []
        for obj in ma_data.get("heads", {}):
            ma.add_head(Laser.from_dict(obj))
        ma.cameras = []
        for obj in ma_data.get("cameras", {}):
            ma.add_camera(Camera.from_dict(obj))
        speeds = ma_data.get("speeds", {})
        ma.max_cut_speed = speeds.get("max_cut_speed", ma.max_cut_speed)
        ma.max_travel_speed = speeds.get(
            "max_travel_speed", ma.max_travel_speed
        )
        ma.acceleration = speeds.get("acceleration", ma.acceleration)
        gcode = ma_data.get("gcode", {})
        ma.gcode_precision = gcode.get("gcode_precision", ma.gcode_precision)
        ma.supports_arcs = ma_data.get("supports_arcs", ma.supports_arcs)
        ma.arc_tolerance = ma_data.get("arc_tolerance", ma.arc_tolerance)

        hours_data = ma_data.get("machine_hours", {})
        ma.machine_hours = MachineHours.from_dict(hours_data)
        ma.machine_hours.changed.connect(ma._on_machine_hours_changed)

        return ma

    def world_point_to_machine(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """
        Converts a point from WORLD space to MACHINE space.

        WORLD space: Bottom-Left (0,0), Y-up
        MACHINE space: Based on origin and axis reversal settings

        This is a pure coordinate transform - no size/box adjustment.

        Args:
            x: X coordinate in world space.
            y: Y coordinate in world space.

        Returns:
            (x, y) in machine space.
        """
        width, height = self._axis_extents

        # Origin corner transformation
        if self.x_axis_right:
            mx = width - x
        else:
            mx = x

        if self.y_axis_down:
            my = height - y
        else:
            my = y

        # Axis reversal
        if self.reverse_x_axis:
            mx = -mx
        if self.reverse_y_axis:
            my = -my

        return (mx, my)

    def machine_point_to_world(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """
        Converts a point from MACHINE space to WORLD space.

        Inverse of world_point_to_machine().

        Args:
            x: X coordinate in machine space.
            y: Y coordinate in machine space.

        Returns:
            (x, y) in world space.
        """
        width, height = self._axis_extents

        # Reverse axis reversal
        mx, my = x, y
        if self.reverse_x_axis:
            mx = -mx
        if self.reverse_y_axis:
            my = -my

        # Reverse origin corner transformation
        if self.x_axis_right:
            wx = width - mx
        else:
            wx = mx

        if self.y_axis_down:
            wy = height - my
        else:
            wy = my

        return (wx, wy)

    def world_to_machine(
        self,
        pos_world: Tuple[float, float],
        size_world: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Converts item position from WORLD space to MACHINE space.

        This handles the item's bounding box - for origins at right/bottom,
        the position refers to the top-left corner, which needs adjustment.

        For point coordinates, use world_point_to_machine() instead.

        Args:
            pos_world: (x, y) position in world coordinates (top-left corner
              of item).
            size_world: (width, height) of the item.

        Returns:
            (x, y) position in machine coordinates.
        """
        wx, wy = pos_world
        w, h = size_world
        width, height = self._axis_extents

        # X Calculation - origin corner + size adjustment
        # Size adjustment must happen BEFORE axis reversal
        if self.x_axis_right:
            mx = width - wx - w
        else:
            mx = wx

        # Y Calculation - origin corner + size adjustment
        if self.y_axis_down:
            my = height - wy - h
        else:
            my = wy

        # Axis reversal - flip coordinates if axes are reversed
        if self.reverse_x_axis:
            mx = -mx
        if self.reverse_y_axis:
            my = -my

        return (mx, my)

    def machine_to_world(
        self,
        pos_machine: Tuple[float, float],
        size_world: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Converts item position from MACHINE space to WORLD space.

        This handles the item's bounding box - for origins at right/bottom,
        the position refers to the top-left corner, which needs adjustment.

        For point coordinates, use machine_point_to_world() instead.

        Args:
            pos_machine: (x, y) position in machine coordinates.
            size_world: (width, height) of the item.

        Returns:
            (x, y) position in world coordinates.
        """
        mx, my = pos_machine
        w, h = size_world
        width, height = self._axis_extents

        # Reverse axis reversal first
        if self.reverse_x_axis:
            mx = -mx
        if self.reverse_y_axis:
            my = -my

        # Reverse origin corner transformation + size adjustment
        if self.x_axis_right:
            wx = width - mx - w
        else:
            wx = mx

        if self.y_axis_down:
            wy = height - my - h
        else:
            wy = my

        return (wx, wy)
