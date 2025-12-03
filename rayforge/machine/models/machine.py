import yaml
import uuid
import logging
import asyncio
import multiprocessing
import json
import numpy as np
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Type
from pathlib import Path
from blinker import Signal
from ...camera.models.camera import Camera
from ...context import get_context, RayforgeContext
from ...core.varset import ValidationError
from ...shared.tasker import task_mgr
from ..transport import TransportStatus
from ..driver.driver import (
    Driver,
    DeviceConnectionError,
    DeviceState,
    DeviceStatus,
    DriverSetupError,
    DriverPrecheckError,
)
from ..driver.dummy import NoDeviceDriver
from ..driver import get_driver_cls
from ..driver.driver import Axis
from .laser import Laser
from .macro import Macro, MacroTrigger
from .dialect import get_dialect, GcodeDialect

if TYPE_CHECKING:
    from ...core.varset import VarSet
    from ...shared.tasker.context import ExecutionContext
    from ...core.ops import Ops
    from ...core.doc import Doc


logger = logging.getLogger(__name__)


def _raise_error(*args, **kwargs):
    raise RuntimeError("Cannot schedule from worker process")


class Machine:
    def __init__(self, context: RayforgeContext):
        logger.debug("Machine.__init__")
        self.id = str(uuid.uuid4())
        self.name: str = _("Default Machine")
        self.context = context

        if multiprocessing.current_process().daemon:
            # This is a worker process, do not allow scheduling signals.
            self._scheduler = _raise_error
        else:
            # This is the main process, use the real scheduler.
            self._scheduler = task_mgr.schedule_on_main_thread

        self.connection_status: TransportStatus = TransportStatus.DISCONNECTED
        self.device_state: DeviceState = DeviceState()

        self.driver_name: Optional[str] = None
        self.driver_args: Dict[str, Any] = {}
        self.precheck_error: Optional[str] = None

        self.driver: Driver = NoDeviceDriver(context, self)

        self.home_on_start: bool = False
        self.clear_alarm_on_connect: bool = False
        self.dialect_uid: str = "grbl"
        self.gcode_precision: int = 3
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
        self.y_axis_down: bool = False
        self.soft_limits_enabled: bool = True
        self._settings_lock = asyncio.Lock()

        # Signals
        self.changed = Signal()
        self.settings_error = Signal()
        self.settings_updated = Signal()
        self.setting_applied = Signal()
        self.connection_status_changed = Signal()
        self.state_changed = Signal()
        self.job_finished = Signal()
        self.command_status_changed = Signal()

        self._connect_driver_signals()
        self.add_head(Laser())

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

    async def _rebuild_driver_instance(
        self, ctx: Optional["ExecutionContext"] = None
    ):
        """
        Instantiates, sets up, and connects the driver based on the
        machine's current configuration. This is managed by the task manager.
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

        # Run setup. A setup error is considered fatal and prevents connection.
        try:
            new_driver.setup(**self.driver_args)
        except DriverSetupError as e:
            logger.error(f"Setup failed for driver {self.driver_name}: {e}")
            new_driver.setup_error = str(e)

        self.driver = new_driver

        self._connect_driver_signals()

        # A setup error prevents connection, but a precheck error does not.
        if self.driver is not None and not self.driver.setup_error:
            # Add the connect task with a key unique to this machine
            task_mgr.add_coroutine(
                lambda ctx: self.driver.connect(),
                key=(self.id, "driver-connect"),
            )
        else:
            logger.error(
                "Driver setup failed, connection will not be attempted."
            )

        # Notify the UI of the change *after* the new driver is in place.
        # This MUST be done on the main thread to prevent UI corruption.
        self._scheduler(self.changed.send, self)

        # Now it is safe to clean up the old driver.
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
        return get_dialect(self.dialect_uid)

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

    def set_home_on_start(self, home_on_start: bool = True):
        self.home_on_start = home_on_start
        self.changed.send(self)

    def set_clear_alarm_on_connect(self, clear_alarm: bool = True):
        if self.clear_alarm_on_connect == clear_alarm:
            return
        self.clear_alarm_on_connect = clear_alarm
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

    def set_y_axis_down(self, y_axis_down: bool):
        self.y_axis_down = y_axis_down
        self.changed.send(self)

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
        # Use machine dimensions as soft limits
        return (0.0, 0.0, float(self.dimensions[0]), float(self.dimensions[1]))

    def would_jog_exceed_limits(self, axis: Axis, distance: float) -> bool:
        """Check if a jog operation would exceed soft limits."""
        if not self.soft_limits_enabled:
            return False

        current_pos = self.get_current_position()
        x_pos, y_pos, z_pos = current_pos

        if x_pos is None or y_pos is None:
            return False

        x_min, y_min, x_max, y_max = self.get_soft_limits()

        # Check X axis
        if axis & Axis.X:
            new_x = x_pos + distance
            if new_x < x_min or new_x > x_max:
                return True

        # Check Y axis
        if axis & Axis.Y:
            new_y = y_pos + distance
            if new_y < y_min or new_y > y_max:
                return True

        return False

    def _adjust_jog_distance_for_limits(
        self, axis: Axis, distance: float
    ) -> float:
        """Adjust jog distance to stay within soft limits."""
        if not self.soft_limits_enabled:
            return distance

        current_pos = self.get_current_position()
        x_pos, y_pos, z_pos = current_pos

        if x_pos is None or y_pos is None:
            return distance

        x_min, y_min, x_max, y_max = self.get_soft_limits()
        adjusted_distance = distance

        # Check X axis
        if axis & Axis.X:
            new_x = x_pos + distance
            if new_x < x_min:
                adjusted_distance = x_min - x_pos
            elif new_x > x_max:
                adjusted_distance = x_max - x_pos

        # Check Y axis (only if not already adjusted for X)
        if axis & Axis.Y and adjusted_distance == distance:
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

    async def jog(self, axis: Axis, distance: float, speed: int):
        """Jogs the machine along a specific axis or combination of axes."""
        if self.driver is None:
            return

        # If soft limits are enabled, adjust distance to stay within limits
        if self.soft_limits_enabled:
            adjusted_distance = self._adjust_jog_distance_for_limits(
                axis, distance
            )
            if adjusted_distance != distance:
                logger.debug(
                    f"Adjusting jog distance from {distance} to "
                    f"{adjusted_distance} to stay within limits"
                )
                distance = adjusted_distance

        await self.driver.jog(axis, distance, speed)

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

    def encode_ops(
        self, ops: "Ops", doc: "Doc"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes an Ops object into machine code (G-code) and a corresponding
        operation map, specific to this machine's configuration.

        Args:
            ops: The Ops object to encode.
            doc: The document context for the job.

        Returns:
            A tuple containing:
            - A numpy array of machine code bytes (UTF-8 encoded G-code).
            - A numpy array of the operation map bytes (UTF-8 encoded JSON).
        """
        encoder = self.driver.get_encoder()

        # If the machine is Y-down, we must transform the ops coordinate
        # system (Y-Up Internal -> Y-Down Machine) before generating G-code.
        # The transform is Y_new = Height - Y_old.
        ops_for_encoder = ops
        if self.y_axis_down:
            ops_for_encoder = ops.copy()
            height = self.dimensions[1]
            # Create transform matrix: Translate(0, H) @ Scale(1, -1)
            # Result: y -> -y -> -y + H
            transform = np.identity(4)
            transform[1, 1] = -1.0
            transform[1, 3] = height
            ops_for_encoder.transform(transform)

        gcode_str, op_map_obj = encoder.encode(ops_for_encoder, self, doc)

        # Encode G-code and map to byte arrays
        machine_code_bytes = np.frombuffer(
            gcode_str.encode("utf-8"), dtype=np.uint8
        )
        op_map_str = json.dumps(asdict(op_map_obj))
        op_map_bytes = np.frombuffer(
            op_map_str.encode("utf-8"), dtype=np.uint8
        )

        return machine_code_bytes, op_map_bytes

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": {
                "name": self.name,
                "driver": self.driver_name,
                "driver_args": self.driver_args,
                "clear_alarm_on_connect": self.clear_alarm_on_connect,
                "home_on_start": self.home_on_start,
                "dialect_uid": self.dialect_uid,
                "dimensions": list(self.dimensions),
                "y_axis_down": self.y_axis_down,
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
            }
        }

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
            label=base_dialect.label, machine_name=machine_name
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
        ma.clear_alarm_on_connect = ma_data.get(
            "clear_alarm_on_connect", ma.clear_alarm_on_connect
        )
        ma.home_on_start = ma_data.get("home_on_start", ma.home_on_start)

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
        ma.dimensions = tuple(ma_data.get("dimensions", ma.dimensions))
        ma.y_axis_down = ma_data.get("y_axis_down", ma.y_axis_down)
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
        ma.gcode_precision = gcode.get("gcode_precision", 3)

        if not is_inert:
            task_mgr.add_coroutine(
                ma._rebuild_driver_instance, key=(ma.id, "rebuild-driver")
            )

        return ma


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
            data = machine.to_dict()
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
