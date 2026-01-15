import asyncio
import logging
import multiprocessing
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np
import yaml
from blinker import Signal

from rayforge.core.ops.commands import MovingCommand

from ...camera.models.camera import Camera
from ...context import RayforgeContext, get_context
from ...core.varset import ValidationError
from ...pipeline.encoder.gcode import MachineCodeOpMap
from ...shared.tasker import task_mgr
from ..driver import get_driver_cls
from ..driver.driver import (
    Axis,
    DeviceConnectionError,
    DeviceState,
    DeviceStatus,
    Driver,
    DriverPrecheckError,
    ResourceBusyError,
)
from ..driver.dummy import NoDeviceDriver
from ..transport import TransportStatus
from .dialect import GcodeDialect, get_dialect
from .laser import Laser
from .machine_hours import MachineHours
from .macro import Macro, MacroTrigger


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


if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.ops import Ops
    from ...core.varset import VarSet
    from ...shared.tasker.context import ExecutionContext


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
        self.dimensions: Tuple[int, int] = 200, 200
        self.offsets: Tuple[int, int] = 0, 0
        self.origin: Origin = Origin.BOTTOM_LEFT
        self.reverse_x_axis: bool = False
        self.reverse_y_axis: bool = False
        self.reverse_z_axis: bool = False
        self.soft_limits_enabled: bool = True
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

        self.driver: Driver = NoDeviceDriver(context, self)
        self._connect_driver_signals()
        self.add_head(Laser())

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
        Delegates to the driver's machine_space_wcs property.
        """
        return self.driver.machine_space_wcs

    @property
    def machine_space_wcs_display_name(self) -> str:
        """
        Returns the display name for the machine space coordinate system.
        Delegates to the driver's machine_space_wcs_display_name property.
        """
        return self.driver.machine_space_wcs_display_name

    async def connect(self):
        """Public method to connect the driver."""
        if self.driver is not None:
            await self.driver.connect()

    async def disconnect(self):
        """Public method to disconnect the driver."""
        # Cancel any pending connection tasks for this driver
        task_mgr.cancel_task((self.id, "driver-connect"))
        if self.driver is not None:
            await self.driver.cleanup()
            # After cleanup, the driver might need to be rebuilt to reconnect
            task_mgr.add_coroutine(
                self._rebuild_driver_instance, key=(self.id, "rebuild-driver")
            )

    async def shutdown(self):
        """
        Gracefully shuts down the machine's active driver and resources.
        """
        logger.info(f"Shutting down machine '{self.name}' (id:{self.id})")
        # Cancel any pending connection tasks for this driver
        task_mgr.cancel_task((self.id, "driver-connect"))
        if self.driver is not None:
            await self.driver.cleanup()
        self._disconnect_driver_signals()
        self.context.dialect_mgr.dialects_changed.disconnect(
            self._on_dialects_changed
        )

    def _connect_driver_signals(self):
        if self.driver is None:
            return
        self.driver.connection_status_changed.connect(
            self._on_driver_connection_status_changed
        )
        self.driver.state_changed.connect(self._on_driver_state_changed)
        self.driver.command_status_changed.connect(
            self._on_driver_command_status_changed
        )
        self.driver.job_finished.connect(self._on_driver_job_finished)
        self.driver.wcs_updated.connect(self._on_driver_wcs_updated)
        self._on_driver_state_changed(self.driver, self.driver.state)
        self._reset_status()

    def _disconnect_driver_signals(self):
        if self.driver is None:
            return
        self.driver.connection_status_changed.disconnect(
            self._on_driver_connection_status_changed
        )
        self.driver.state_changed.disconnect(self._on_driver_state_changed)
        self.driver.command_status_changed.disconnect(
            self._on_driver_command_status_changed
        )
        self.driver.job_finished.disconnect(self._on_driver_job_finished)
        self.driver.wcs_updated.disconnect(self._on_driver_wcs_updated)

    def _on_dialects_changed(self, sender=None, **kwargs):
        """
        Callback when dialects are updated.
        Sends machine's changed signal to trigger recalculation.
        """
        self.changed.send(self)

    async def _rebuild_driver_instance(
        self, ctx: Optional["ExecutionContext"] = None
    ):
        """
        Instantiates and sets up the driver based on the machine's current
        configuration. It does NOT connect it.
        """
        logger.info(
            f"Machine '{self.name}' (id:{self.id}) rebuilding driver to "
            f"'{self.driver_name}'"
        )

        old_driver = self.driver
        self._disconnect_driver_signals()
        self.precheck_error = None

        if self.driver_name:
            driver_cls = get_driver_cls(self.driver_name)
        else:
            driver_cls = NoDeviceDriver

        # Run precheck before instantiation. This error is a non-fatal warning.
        try:
            driver_cls.precheck(**self.driver_args)
        except DriverPrecheckError as e:
            logger.warning(
                f"Precheck failed for driver {self.driver_name}: {e}"
            )
            self.precheck_error = str(e)

        new_driver = driver_cls(self.context, self)
        new_driver.setup(**self.driver_args)

        self.driver = new_driver
        self._connect_driver_signals()

        # Notify the UI of the change *after* the new driver is in place.
        self._scheduler(self.changed.send, self)

        # Now it is safe to clean up the old driver.
        if old_driver:
            await old_driver.cleanup()

    def _reset_status(self):
        """Resets status to a disconnected/unknown state and signals it."""
        state_actually_changed = (
            self.device_state.status != DeviceStatus.UNKNOWN
        )
        conn_actually_changed = (
            self.connection_status != TransportStatus.DISCONNECTED
        )

        self.device_state = DeviceState()  # Defaults to UNKNOWN
        self.connection_status = TransportStatus.DISCONNECTED

        if state_actually_changed:
            self._scheduler(
                self.state_changed.send, self, state=self.device_state
            )
        if conn_actually_changed:
            self._scheduler(
                self.connection_status_changed.send,
                self,
                status=self.connection_status,
                message="Driver inactive",
            )

    def _on_driver_connection_status_changed(
        self,
        driver: Driver,
        status: TransportStatus,
        message: Optional[str] = None,
    ):
        """Proxies the connection status signal from the active driver."""
        if self.connection_status != status:
            self.connection_status = status
            self._scheduler(
                self.connection_status_changed.send,
                self,
                status=status,
                message=message,
            )
            if status == TransportStatus.CONNECTED:
                # Sync all relevant state from device on connect
                task_mgr.add_coroutine(
                    lambda ctx: self.sync_wcs_from_device(),
                    key=(self.id, "sync-wcs-offsets"),
                )
                task_mgr.add_coroutine(
                    lambda ctx: self.sync_active_wcs_from_device(),
                    key=(self.id, "sync-active-wcs"),
                )

    def _on_driver_state_changed(self, driver: Driver, state: DeviceState):
        """Proxies the state changed signal from the active driver."""
        # Avoid redundant signals if state hasn't changed.
        if self.device_state != state:
            self.device_state = state
            self._scheduler(self.state_changed.send, self, state=state)

    def _on_driver_job_finished(self, driver: Driver):
        """Proxies the job finished signal from the active driver."""
        self._scheduler(self.job_finished.send, self)

    def _on_driver_command_status_changed(
        self,
        driver: Driver,
        status: TransportStatus,
        message: Optional[str] = None,
    ):
        """Proxies the command status changed signal from the active driver."""
        self._scheduler(
            self.command_status_changed.send,
            self,
            status=status,
            message=message,
        )

    def _on_driver_wcs_updated(
        self, driver: Driver, offsets: Dict[str, Tuple[float, float, float]]
    ):
        """Updates internal WCS state from driver updates."""
        self.wcs_offsets.update(offsets)
        self._scheduler(self.wcs_updated.send, self)
        # Also notify general change so views update
        self._scheduler(self.changed.send, self)

    def is_connected(self) -> bool:
        """
        Checks if the machine's driver is currently connected to the device.

        Returns:
            True if connected, False otherwise.
        """
        return self.connection_status == TransportStatus.CONNECTED

    async def select_tool(self, index: int):
        """Sends a command to the driver to select a tool."""
        if self.driver is None:
            return
        await self.driver.select_tool(index)

    def set_name(self, name: str):
        self.name = str(name)
        self.changed.send(self)

    def set_driver(self, driver_cls: Type[Driver], args=None):
        new_driver_name = driver_cls.__name__
        new_args = args or {}
        if (
            self.driver_name == new_driver_name
            and self.driver_args == new_args
        ):
            return

        self.driver_name = new_driver_name
        self.driver_args = new_args
        # Use a key to ensure only one rebuild task is pending per machine
        task_mgr.add_coroutine(
            self._rebuild_driver_instance, key=(self.id, "rebuild-driver")
        )

    def set_driver_args(self, args=None):
        new_args = args or {}
        if self.driver_args == new_args:
            return

        self.driver_args = new_args
        # Use a key to ensure only one rebuild task is pending per machine
        task_mgr.add_coroutine(
            self._rebuild_driver_instance, key=(self.id, "rebuild-driver")
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

    def set_dimensions(self, width: int, height: int):
        self.dimensions = (width, height)
        self.changed.send(self)

    def set_offsets(self, x_offset: int, y_offset: int):
        self.offsets = (x_offset, y_offset)
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
        w, h = float(self.dimensions[0]), float(self.dimensions[1])

        # If an axis is reversed, the workspace is in the negative domain.
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

    def can_g0_with_speed(self) -> bool:
        """Check if the machine's driver supports G0 with speed."""
        if self.driver is None:
            return False
        return self.driver.can_g0_with_speed()

    @property
    def reports_granular_progress(self) -> bool:
        """Check if the machine's driver reports granular progress."""
        if self.driver is None:
            return False
        return self.driver.reports_granular_progress

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        """Check if the machine's driver supports homing for the given axis."""
        if self.driver is None:
            return False
        return self.driver.can_home(axis)

    async def home(self, axes=None):
        """Homes the specified axes or all axes if none specified."""
        if self.driver is None:
            return
        await self.driver.home(axes)

    async def jog(self, deltas: Dict[Axis, float], speed: int):
        """
        Jogs the machine along specified axes.

        Args:
            deltas: Dictionary mapping Axis enum members to distances in mm.
            speed: Speed in mm/min.
        """
        if self.driver is None:
            return

        driver_kwargs = {}

        for axis, distance in deltas.items():
            if distance == 0:
                continue

            # Check limits if enabled
            if self.soft_limits_enabled:
                distance = self._adjust_jog_distance_for_limits(axis, distance)

            # Only add if there is movement and the axis has a name
            if distance != 0 and axis.name:
                driver_kwargs[axis.name.lower()] = distance

        if not driver_kwargs:
            return

        await self.driver.jog(speed=speed, **driver_kwargs)

    async def run_raw(self, gcode: str):
        """Executes a raw G-code string on the machine."""
        if self.driver is None:
            logger.warning("run_raw called but no driver is available.")
            return
        await self.driver.run_raw(gcode)

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """Check if machine's supports jogging for the given axis."""
        if self.driver is None:
            return False
        return self.driver.can_jog(axis)

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
        VarSet.

        Returns:
            A tuple of (is_valid, error_message).
        """
        if not self.driver_name:
            return False, _("No driver selected for this machine.")

        driver_cls = get_driver_cls(self.driver_name)
        if not driver_cls:
            return False, _("Driver '{driver}' not found.").format(
                driver=self.driver_name
            )

        try:
            setup_vars = driver_cls.get_setup_vars()
            setup_vars.set_values(self.driver_args)
            setup_vars.validate()
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            # Catch other potential errors during var setup
            return False, _(
                "An unexpected error occurred during validation: {error}"
            ).format(error=str(e))

        return True, None

    async def set_power(
        self, head: Optional["Laser"] = None, percent: float = 0.0
    ) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control. If None, uses the default head.
            percent: Power percentage (0-1.0). 0 disables power.
        """
        logger.debug(
            f"Head {head.uid if head else None} power to {percent * 100}%"
        )
        if not self.driver:
            raise ValueError("No driver configured for this machine.")

        # Use default head if none specified
        if head is None:
            head = self.get_default_head()

        await self.driver.set_power(head, percent)

    def get_active_wcs_offset(self) -> Tuple[float, float, float]:
        """
        Returns the (x, y, z) offset for the currently active WCS.
        If the active_wcs is not in the known offsets dictionary, it assumes
        an absolute coordinate system with zero offset.
        """
        return self.wcs_offsets.get(self.active_wcs, (0.0, 0.0, 0.0))

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
        slot = wcs_slot or self.active_wcs
        if slot not in self.wcs_offsets:
            logger.warning(
                f"Cannot set offset for immutable WCS '{slot}' "
                "(e.g. Machine Coordinates)."
            )
            return

        if not self.is_connected():
            # Update local state directly if offline
            self.wcs_offsets[slot] = (x, y, z)
            # Notify listeners of WCS update
            self._scheduler(self.wcs_updated.send, self)
            # Notify listeners of general machine state change
            self._scheduler(self.changed.send, self)
            return

        await self.driver.set_wcs_offset(slot, x, y, z)
        # Trigger read back to ensure state is synced
        await self.driver.read_wcs_offsets()

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
        if not self.is_connected():
            return

        slot = wcs_slot or self.active_wcs
        if slot not in self.wcs_offsets:
            logger.warning(
                f"Cannot set offset for immutable WCS '{slot}' "
                "(e.g. Machine Coordinates)."
            )
            return

        # Get current machine position
        m_pos = self.device_state.machine_pos
        # Need to handle None values in machine_pos (if not reported yet)
        if any(v is None for v in m_pos):
            logger.warning("Cannot set work origin: Unknown machine position.")
            return

        # Get current offsets to preserve unselected axes
        current_offsets = self.wcs_offsets.get(slot, (0.0, 0.0, 0.0))

        new_x, new_y, new_z = current_offsets

        # For "Set Zero Here", the new offset is exactly the current machine
        # position for that axis.
        # Work_Pos = Machine_Pos - Offset.
        # If Work_Pos = 0, Offset = Machine_Pos.
        if axes & Axis.X and m_pos[0] is not None:
            new_x = m_pos[0]
        if axes & Axis.Y and m_pos[1] is not None:
            new_y = m_pos[1]
        if axes & Axis.Z and m_pos[2] is not None:
            new_z = m_pos[2]

        await self.set_work_origin(new_x, new_y, new_z, slot)

    async def sync_wcs_from_device(self):
        """Queries the device for current WCS offsets and updates state."""
        if self.is_connected():
            await self.driver.read_wcs_offsets()

    async def sync_active_wcs_from_device(self):
        """Queries the device for its active WCS and updates state."""
        if self.is_connected():
            active_wcs = await self.driver.read_parser_state()
            if active_wcs:
                logger.info(f"Synced active WCS from device: '{active_wcs}'")
                self.set_active_wcs(active_wcs)

    def encode_ops(
        self, ops: "Ops", doc: "Doc"
    ) -> Tuple[str, "MachineCodeOpMap"]:
        """
        Encodes an Ops object into machine code (G-code) and a corresponding
        operation map, specific to this machine's configuration.
        This is the single source of truth for applying machine-specific
        coordinate transformations.

        Args:
            ops: The Ops object to encode.
            doc: The document context for the job.

        Returns:
            A tuple containing:
            - A string of machine code (G-code).
            - A MachineCodeOpMap object.
        """
        encoder = self.driver.get_encoder()

        # We operate on a copy to avoid modifying the original Ops object,
        # which is owned by the pipeline and may be reused.
        ops_for_encoder = ops.copy()

        # Apply offsets
        for command in ops_for_encoder.commands:
            if isinstance(command, MovingCommand):
                base_end = command.end or (0, 0, 0)
                command.end = (
                    base_end[0] + self.offsets[0],
                    base_end[1] + self.offsets[1],
                    base_end[2],
                )

        # If Origin is BOTTOM_LEFT and axes are not reversed, the internal
        # coordinate system matches the machine's. Any other configuration
        # requires transformation.
        needs_transform = (
            self.origin != Origin.BOTTOM_LEFT
            or self.reverse_x_axis
            or self.reverse_y_axis
        )

        if needs_transform:
            width, height = self.dimensions

            # Create the origin transformation matrix. This is complex because
            # it depends on both the origin corner and whether the machine
            # uses a positive or negative coordinate system for each axis.
            # The 'reverse_x_axis' and 'reverse_y_axis' flags indicate a
            # negative coordinate system.
            transform = np.identity(4)

            # --- Y-Axis Transformation ---
            if self.y_axis_down:  # Origin is TOP_LEFT or TOP_RIGHT
                if self.reverse_y_axis:
                    # Negative workspace: Machine Y is 0 at top,
                    # decreases down.
                    # World Y=height maps to Machine Y=0.
                    # Formula: y_m = y_w - height
                    transform[1, 3] = -float(height)
                else:
                    # Positive workspace: Machine Y is 0 at top,
                    # increases down.
                    # World Y=height maps to Y=0; World Y=0 maps to
                    # Y=height.
                    # Formula: y_m = height - y_w
                    transform[1, 1] = -1.0
                    transform[1, 3] = float(height)
            elif self.reverse_y_axis:
                # Origin is at bottom, but Y is negative (uncommon)
                # World Y=0 maps to Y=0, but Y increases negatively.
                # Formula: y_m = -y_w
                transform[1, 1] = -1.0

            # --- X-Axis Transformation ---
            if self.x_axis_right:  # Origin is TOP_RIGHT or BOTTOM_RIGHT
                if self.reverse_x_axis:
                    # Negative workspace: Machine X is 0 at right,
                    # decreases left.
                    # World X=width maps to Machine X=0.
                    # Formula: x_m = x_w - width
                    transform[0, 3] = -float(width)
                else:
                    # Positive workspace: Machine X is 0 at right,
                    # increases left.
                    # World X=width maps to X=0; World X=0 maps to
                    # X=width.
                    # Formula: x_m = width - x_w
                    transform[0, 0] = -1.0
                    transform[0, 3] = float(width)
            elif self.reverse_x_axis:
                # Origin is at left, but X is negative (uncommon)
                # World X=0 maps to X=0, but X increases negatively.
                # Formula: x_m = -x_w
                transform[0, 0] = -1.0

            ops_for_encoder.transform(transform)

        # Apply WCS Offset Logic
        # The document is drawn on a canvas representing the full machine bed.
        # ops_for_encoder is now in "Machine Coordinates".
        # We must subtract the WCS offset to get "Command Coordinates", so that
        # when the machine adds the offset back during execution, it lands on
        # the correct physical spot.
        # Cmd = Machine - Offset
        wcs_offset = self.get_active_wcs_offset()
        if wcs_offset != (0.0, 0.0, 0.0):
            wcs_transform = np.identity(4)
            wcs_transform[0, 3] = -wcs_offset[0]
            wcs_transform[1, 3] = -wcs_offset[1]
            wcs_transform[2, 3] = -wcs_offset[2]
            ops_for_encoder.transform(wcs_transform)

        gcode_str, op_map_obj = encoder.encode(ops_for_encoder, self, doc)

        return gcode_str, op_map_obj

    def refresh_settings(self):
        """Public API for the UI to request a settings refresh."""
        task_mgr.add_coroutine(
            lambda ctx: self._read_from_device(),
            key=(self.id, "device-settings-read"),
        )

    def apply_setting(self, key: str, value: Any):
        """Public API for the UI to apply a single setting."""
        task_mgr.add_coroutine(
            lambda ctx: self._write_setting_to_device(key, value),
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
        if self.driver is None:
            return []
        return self.driver.get_setting_vars()

    async def _read_from_device(self):
        """
        Task entry point for reading settings. This handles locking and
        all errors.
        """
        logger.debug("Machine._read_from_device: Acquiring lock.")
        async with self._settings_lock:
            logger.debug("_read_from_device: Lock acquired.")
            if self.driver is None:
                err = ConnectionError("No driver instance for this machine.")
                self.settings_error.send(self, error=err)
                return

            def on_settings_read(sender, settings: List["VarSet"]):
                logger.debug("on_settings_read: Handler called.")
                sender.settings_read.disconnect(on_settings_read)
                self._scheduler(
                    self.settings_updated.send, self, var_sets=settings
                )
                logger.debug("on_settings_read: Handler finished.")

            self.driver.settings_read.connect(on_settings_read)
            try:
                await self.driver.read_settings()
            except (DeviceConnectionError, ConnectionError) as e:
                logger.error(f"Failed to read settings from device: {e}")
                self.driver.settings_read.disconnect(on_settings_read)
                self._scheduler(self.settings_error.send, self, error=e)
            finally:
                logger.debug("_read_from_device: Read operation finished.")
        logger.debug("_read_from_device: Lock released.")

    async def _write_setting_to_device(self, key: str, value: Any):
        """
        Writes a single setting to the device and signals success or failure.
        It no longer triggers an automatic re-read.
        """
        logger.debug(f"_write_setting_to_device(key={key}): Acquiring lock.")
        if self.driver is None:
            err = ConnectionError("No driver instance for this machine.")
            self.settings_error.send(self, error=err)
            return

        try:
            async with self._settings_lock:
                logger.debug(
                    f"_write_setting_to_device(key={key}): Lock acquired."
                )
                await self.driver.write_setting(key, value)
                self._scheduler(self.setting_applied.send, self)
        except (DeviceConnectionError, ConnectionError) as e:
            logger.error(f"Failed to write setting to device: {e}")
            self._scheduler(self.settings_error.send, self, error=e)
        finally:
            logger.debug(f"_write_setting_to_device(key={key}): Done.")

    def to_dict(self, include_frozen_dialect: bool = True) -> Dict[str, Any]:
        data = {
            "machine": {
                "name": self.name,
                "driver": self.driver_name,
                "driver_args": self.driver_args,
                "auto_connect": self.auto_connect,
                "clear_alarm_on_connect": self.clear_alarm_on_connect,
                "home_on_start": self.home_on_start,
                "single_axis_homing_enabled": self.single_axis_homing_enabled,  # noqa: E501
                "dialect_uid": self.dialect_uid,
                "active_wcs": self.active_wcs,
                "wcs_offsets": self.wcs_offsets,
                "supports_arcs": self.supports_arcs,
                "arc_tolerance": self.arc_tolerance,
                "dimensions": list(self.dimensions),
                "offsets": list(self.offsets),
                "origin": self.origin.value,
                "reverse_x_axis": self.reverse_x_axis,
                "reverse_y_axis": self.reverse_y_axis,
                "reverse_z_axis": self.reverse_z_axis,
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

        ma.dimensions = tuple(ma_data.get("dimensions", ma.dimensions))
        ma.offsets = tuple(ma_data.get("offsets", ma.offsets))
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

    def world_to_machine(
        self,
        pos_world: Tuple[float, float],
        size_world: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Converts coordinates from internal World Space (Bottom-Left 0,0, Y-Up)
        to Machine Space (User-facing, based on Origin setting).

        Args:
            pos_world: (x, y) position in world coordinates (top-left corner
              of item).
            size_world: (width, height) of the item.

        Returns:
            (x, y) position in machine coordinates.
        """
        machine_width, machine_height = self.dimensions
        wx, wy = pos_world
        w, h = size_world

        # X Calculation
        if self.x_axis_right:
            # Origin is Right. World X=0 is Far Right in Machine Space?
            # No, Internal World 0,0 is always Bottom-Left.
            # If Machine Origin is Top-Right (X-Left, Y-Down):
            # Machine X=0 is Right edge. Machine X increases to the Left.
            # pos_machine_x = machine_width - pos_world_x - item_width
            mx = machine_width - wx - w
        else:
            # Origin is Left. Machine X increases to the Right (standard).
            mx = wx

        # Y Calculation
        if self.y_axis_down:
            # Origin is Top. Machine Y=0 is Top edge. Machine Y increases Down.
            # Internal World Y=0 is Bottom.
            # pos_machine_y = machine_height - pos_world_y - item_height
            my = machine_height - wy - h
        else:
            # Origin is Bottom. Machine Y increases Up (standard).
            my = wy

        return mx, my

    def machine_to_world(
        self,
        pos_machine: Tuple[float, float],
        size_world: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Converts coordinates from Machine Space (User-facing) back to
        internal World Space (Bottom-Left 0,0, Y-Up).

        Args:
            pos_machine: (x, y) position in machine coordinates.
            size_world: (width, height) of the item.

        Returns:
            (x, y) position in world coordinates.
        """
        machine_width, machine_height = self.dimensions
        mx, my = pos_machine
        w, h = size_world

        # The logic is symmetric to world_to_machine.

        # X Calculation
        if self.x_axis_right:
            # wx = machine_width - mx - w
            wx = machine_width - mx - w
        else:
            wx = mx

        # Y Calculation
        if self.y_axis_down:
            # wy = machine_height - my - h
            wy = machine_height - my - h
        else:
            wy = my

        return wx, wy


class MachineManager:
    def __init__(self, base_dir: Path):
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.machines: Dict[str, Machine] = dict()
        self._machine_ref_for_pyreverse: Machine
        self.machine_added = Signal()
        self.machine_removed = Signal()
        self.machine_updated = Signal()
        self.load()

    async def shutdown(self):
        """
        Shuts down all managed machines and their drivers gracefully.
        """
        logger.info("Shutting down all machines.")
        tasks = [machine.shutdown() for machine in self.machines.values()]
        if tasks:
            await asyncio.gather(*tasks)
        logger.info("All machines shut down.")

    def initialize_connections(self):
        """
        Initializes machine connections on startup by rebuilding all drivers
        and then connecting, prioritizing the active machine.
        """
        context = get_context()
        active_machine = context.config.machine

        # Define a lambda to use with add_coroutine that captures the machine
        def connect_task(m):
            return lambda ctx: self._rebuild_and_connect_machine(m)

        # First, schedule the task for the active machine
        if active_machine:
            task_mgr.add_coroutine(connect_task(active_machine))

        # Then, schedule tasks for the rest
        for machine in self.machines.values():
            if machine is not active_machine:
                task_mgr.add_coroutine(connect_task(machine))

    async def _rebuild_and_connect_machine(self, machine: "Machine"):
        """
        A single, sequenced task that rebuilds a machine's driver and then
        connects if auto_connect is on.
        """
        await machine._rebuild_driver_instance()
        if machine.auto_connect:
            await self._safe_connect(machine)

    async def _safe_connect(self, machine: "Machine"):
        """
        Attempts to connect a machine, suppressing ResourceBusyErrors.
        """
        try:
            await machine.connect()
        except ResourceBusyError:
            context = get_context()
            if machine is context.config.machine:
                logger.warning(
                    f"Active machine '{machine.name}' could not connect "
                    "because resource is busy."
                )
            else:
                logger.debug(
                    f"Inactive machine '{machine.name}' deferred connection: "
                    "resource busy."
                )
        except Exception as e:
            logger.error(
                f"Failed to auto-connect machine '{machine.name}': {e}"
            )

    def set_active_machine(self, new_machine: Machine):
        """
        Sets the active machine, handling the connection lifecycle for
        shared resources.
        """
        context = get_context()
        old_machine = context.config.machine

        if old_machine and old_machine.id == new_machine.id:
            return  # No change

        logger.info(f"Switching active machine to '{new_machine.name}'")

        async def switch_routine(ctx):
            # 1. Disconnect the old machine if it's connected
            if old_machine and old_machine.is_connected():
                logger.info(
                    f"Disconnecting previous machine '{old_machine.name}'"
                )
                await old_machine.disconnect()
                # Add a small delay for the OS to release the port
                await asyncio.sleep(0.2)

            # 2. Update the global config. This triggers UI updates.
            context.config.set_machine(new_machine)

            # 3. Connect the new machine if it's set to auto-connect
            if new_machine.auto_connect:
                logger.info(
                    f"Connecting to new active machine '{new_machine.name}'"
                )
                await self._safe_connect(new_machine)

        task_mgr.add_coroutine(switch_routine)

    def filename_from_id(self, machine_id: str) -> Path:
        return self.base_dir / f"{machine_id}.yaml"

    def add_machine(self, machine: Machine):
        if machine.id in self.machines:
            return
        self.machines[machine.id] = machine
        machine.changed.connect(self.on_machine_changed)
        self.save_machine(machine)
        self.machine_added.send(self, machine_id=machine.id)

    def remove_machine(self, machine_id: str):
        machine = self.machines.get(machine_id)
        if not machine:
            return

        machine.changed.disconnect(self.on_machine_changed)
        del self.machines[machine_id]

        machine_file = self.filename_from_id(machine_id)
        try:
            machine_file.unlink()
            logger.info(f"Removed machine file: {machine_file}")
        except OSError as e:
            logger.error(f"Error removing machine file {machine_file}: {e}")

        self.machine_removed.send(self, machine_id=machine_id)

    def get_machine_by_id(self, machine_id):
        return self.machines.get(machine_id)

    def get_machines(self) -> List["Machine"]:
        """Returns a list of all managed machines, sorted by name."""
        return sorted(list(self.machines.values()), key=lambda m: m.name)

    def create_default_machine(self):
        machine = Machine(get_context())
        self.add_machine(machine)
        return machine

    def save_machine(self, machine):
        logger.debug(f"Saving machine {machine.id}")
        machine_file = self.filename_from_id(machine.id)
        with open(machine_file, "w") as f:
            data = machine.to_dict(include_frozen_dialect=False)
            yaml.safe_dump(data, f)

    def load_machine(self, machine_id: str) -> Optional["Machine"]:
        machine_file = self.filename_from_id(machine_id)
        if not machine_file.exists():
            raise FileNotFoundError(f"Machine file {machine_file} not found")
        with open(machine_file, "r") as f:
            data = yaml.safe_load(f)
            if not data:
                msg = f"skipping invalid machine file {f.name}"
                logger.warning(msg)
                return None
        machine = Machine.from_dict(data)
        machine.id = machine_id
        self.machines[machine.id] = machine
        machine.changed.connect(self.on_machine_changed)
        return machine

    def on_machine_changed(self, machine, **kwargs):
        self.save_machine(machine)
        self.machine_updated.send(self, machine_id=machine.id)

    def load(self):
        for file in self.base_dir.glob("*.yaml"):
            try:
                self.load_machine(file.stem)
            except Exception as e:
                logger.error(f"Failed to load machine from {file}: {e}")
