import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from blinker import Signal

from ...core.varset import ValidationError
from ...shared.tasker import task_mgr
from ..driver import get_driver_cls
from ..driver.driver import (
    Axis,
    DeviceConnectionError,
    DeviceState,
    DeviceStatus,
    Driver,
    DriverPrecheckError,
)
from ..driver.dummy import NoDeviceDriver
from ..transport import TransportStatus


if TYPE_CHECKING:
    from ...context import RayforgeContext
    from ...core.varset import VarSet
    from ...shared.tasker.context import ExecutionContext
    from .laser import Laser
    from .machine import Machine


logger = logging.getLogger(__name__)


class MachineController:
    """
    Controller for machine logic and driver ownership.
    Manages the driver lifecycle and emits signals that the Machine
    re-emits to maintain backward compatibility.
    """

    def __init__(self, machine: "Machine", context: "RayforgeContext"):
        self.machine = machine
        self.context = context

        # Controller signals - Machine will connect to these and re-emit
        self.connection_status_changed = Signal()
        self.state_changed = Signal()
        self.job_finished = Signal()
        self.command_status_changed = Signal()
        self.wcs_updated = Signal()

        self.driver: Driver = NoDeviceDriver(context, machine)
        self._connect_driver_signals()

        # Track the last driver configuration to detect changes
        self._last_driver_name = self.machine.driver_name
        self._last_driver_args = self.machine.driver_args.copy()

        # Listen to machine's changed signal to rebuild driver when
        # driver configuration changes
        self.machine.changed.connect(self._on_machine_changed)

        # If the machine already has a driver_name configured, rebuild
        # the driver instance to match
        if self.machine.driver_name:
            task_mgr.add_coroutine(
                self._rebuild_driver_instance,
                key=(self.machine.id, "rebuild-driver-on-init"),
            )

    async def connect(self):
        """Public method to connect the driver."""
        if self.driver is not None:
            await self.driver.connect()

    async def disconnect(self):
        """Public method to disconnect the driver."""
        task_mgr.cancel_task((self.machine.id, "driver-connect"))
        if self.driver is not None:
            await self.driver.cleanup()
            task_mgr.add_coroutine(
                self._rebuild_driver_instance,
                key=(self.machine.id, "rebuild-driver"),
            )

    async def shutdown(self):
        """
        Gracefully shuts down the machine's active driver and resources.
        """
        logger.info(
            f"Shutting down controller for machine '{self.machine.name}' "
            f"(id:{self.machine.id})"
        )
        task_mgr.cancel_task((self.machine.id, "driver-connect"))
        if self.driver is not None:
            await self.driver.cleanup()
        self._disconnect_driver_signals()
        self.machine.changed.disconnect(self._on_machine_changed)
        self.context.dialect_mgr.dialects_changed.disconnect(
            self._on_dialects_changed
        )

    def _on_machine_changed(self, sender=None, **kwargs):
        """
        Callback when the machine's configuration changes.
        Triggers driver rebuild only if the driver configuration has changed.
        """
        current_driver_name = self.machine.driver_name
        current_driver_args = self.machine.driver_args

        if (current_driver_name != self._last_driver_name or
                current_driver_args != self._last_driver_args):
            self._last_driver_name = current_driver_name
            self._last_driver_args = current_driver_args.copy()
            task_mgr.add_coroutine(
                self._rebuild_driver_instance,
                key=(self.machine.id, "rebuild-driver-on-change"),
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
        self.machine.changed.send(self.machine)

    async def _rebuild_driver_instance(
        self, ctx: Optional["ExecutionContext"] = None
    ):
        """
        Instantiates and sets up the driver based on the machine's current
        configuration. Connects if auto_connect is enabled and the new driver
        is not NoDeviceDriver.
        """
        logger.info(
            f"Machine '{self.machine.name}' (id:{self.machine.id}) rebuilding "
            f"driver to '{self.machine.driver_name}'"
        )

        old_driver = self.driver
        self._disconnect_driver_signals()
        self.machine.precheck_error = None

        if self.machine.driver_name:
            driver_cls = get_driver_cls(self.machine.driver_name)
        else:
            driver_cls = NoDeviceDriver

        try:
            driver_cls.precheck(**self.machine.driver_args)
        except DriverPrecheckError as e:
            logger.warning(
                f"Precheck failed for driver {self.machine.driver_name}: {e}"
            )
            self.machine.precheck_error = str(e)

        new_driver = driver_cls(self.context, self.machine)
        new_driver.setup(**self.machine.driver_args)

        self.driver = new_driver
        self._connect_driver_signals()

        self.machine._scheduler(self.machine.changed.send, self.machine)

        if old_driver:
            await old_driver.cleanup()

        if self.machine.auto_connect and not isinstance(
            new_driver, NoDeviceDriver
        ):
            logger.info(
                f"Machine '{self.machine.name}' (id:{self.machine.id}) "
                f"connecting after driver rebuild"
            )
            await self.driver.connect()

    def _reset_status(self):
        """Resets status to a disconnected/unknown state and signals it."""
        state_actually_changed = (
            self.machine.device_state.status != DeviceStatus.UNKNOWN
        )
        conn_actually_changed = (
            self.machine.connection_status != TransportStatus.DISCONNECTED
        )

        self.machine.device_state = DeviceState()
        self.machine.connection_status = TransportStatus.DISCONNECTED

        if state_actually_changed:
            self.machine._scheduler(
                self.state_changed.send,
                self.machine,
                state=self.machine.device_state,
            )
        if conn_actually_changed:
            self.machine._scheduler(
                self.connection_status_changed.send,
                self.machine,
                status=self.machine.connection_status,
                message="Driver inactive",
            )

    def _on_driver_connection_status_changed(
        self,
        driver: Driver,
        status: TransportStatus,
        message: Optional[str] = None,
    ):
        """Proxies the connection status signal from the active driver."""
        if self.machine.connection_status != status:
            self.machine.connection_status = status
            self.machine._scheduler(
                self.connection_status_changed.send,
                self.machine,
                status=status,
                message=message,
            )
            if status == TransportStatus.CONNECTED:
                if self.machine.precheck_error:
                    self.machine.precheck_error = None
                task_mgr.add_coroutine(
                    lambda ctx: self.machine.sync_wcs_from_device(),
                    key=(self.machine.id, "sync-wcs-offsets"),
                )
                task_mgr.add_coroutine(
                    lambda ctx: self.machine.sync_active_wcs_from_device(),
                    key=(self.machine.id, "sync-active-wcs"),
                )

    def _on_driver_state_changed(self, driver: Driver, state: DeviceState):
        """Proxies the state changed signal from the active driver."""
        if self.machine.device_state != state:
            self.machine.device_state = state
            self.machine._scheduler(
                self.state_changed.send, self.machine, state=state
            )

    def _on_driver_job_finished(self, driver: Driver):
        """Proxies the job finished signal from the active driver."""
        self.machine._scheduler(self.job_finished.send, self.machine)

    def _on_driver_command_status_changed(
        self,
        driver: Driver,
        status: TransportStatus,
        message: Optional[str] = None,
    ):
        """Proxies the command status changed signal from the active driver."""
        self.machine._scheduler(
            self.command_status_changed.send,
            self.machine,
            status=status,
            message=message,
        )

    def _on_driver_wcs_updated(
        self, driver: Driver, offsets: Dict[str, Tuple[float, float, float]]
    ):
        """Updates internal WCS state from driver updates."""
        self.machine.wcs_offsets.update(offsets)
        logger.debug(
            f"MachineController: Emitting wcs_updated for machine "
            f"{self.machine.id}. Sender: {self.machine}"
        )
        self.machine._scheduler(self.wcs_updated.send, self.machine)
        self.machine._scheduler(self.machine.changed.send, self.machine)

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

            if self.machine.soft_limits_enabled:
                distance = self.machine._adjust_jog_distance_for_limits(
                    axis, distance
                )

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

    async def select_tool(self, index: int):
        """Sends a command to the driver to select a tool."""
        if self.driver is None:
            return
        await self.driver.select_tool(index)

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

        if head is None:
            head = self.machine.get_default_head()

        await self.driver.set_power(head, percent)

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
        slot = wcs_slot or self.machine.active_wcs
        if slot not in self.machine.wcs_offsets:
            logger.warning(
                f"Cannot set offset for immutable WCS '{slot}' "
                "(e.g. Machine Coordinates)."
            )
            return

        if not self.machine.is_connected():
            self.machine.wcs_offsets[slot] = (x, y, z)
            self.machine._scheduler(self.wcs_updated.send, self)
            self.machine._scheduler(self.machine.changed.send, self)
            return

        await self.driver.set_wcs_offset(slot, x, y, z)
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
        if not self.machine.is_connected():
            return

        slot = wcs_slot or self.machine.active_wcs
        if slot not in self.machine.wcs_offsets:
            logger.warning(
                f"Cannot set offset for immutable WCS '{slot}' "
                "(e.g. Machine Coordinates)."
            )
            return

        m_pos = self.machine.device_state.machine_pos
        if any(v is None for v in m_pos):
            logger.warning("Cannot set work origin: Unknown machine position.")
            return

        current_offsets = self.machine.wcs_offsets.get(slot, (0.0, 0.0, 0.0))

        new_x, new_y, new_z = current_offsets

        if axes & Axis.X and m_pos[0] is not None:
            new_x = m_pos[0]
        if axes & Axis.Y and m_pos[1] is not None:
            new_y = m_pos[1]
        if axes & Axis.Z and m_pos[2] is not None:
            new_z = m_pos[2]

        await self.set_work_origin(new_x, new_y, new_z, slot)

    async def sync_wcs_from_device(self):
        """Queries the device for current WCS offsets and updates state."""
        if self.machine.is_connected():
            try:
                await self.driver.read_wcs_offsets()
            except asyncio.TimeoutError:
                logger.error(
                    "Failed to sync WCS offsets: device timed out "
                    "while responding to $# command."
                )
            except asyncio.CancelledError:
                logger.debug("WCS offset sync cancelled by task manager.")
                raise
            except DeviceConnectionError as e:
                logger.error(
                    f"Failed to sync WCS offsets: connection error: {e}"
                )

    async def sync_active_wcs_from_device(self):
        """Queries the device for its active WCS and updates state."""
        if self.machine.is_connected():
            try:
                active_wcs = await self.driver.read_parser_state()
                if active_wcs:
                    logger.info(
                        f"Synced active WCS from device: '{active_wcs}'"
                    )
                    self.machine.set_active_wcs(active_wcs)
            except asyncio.TimeoutError:
                logger.error(
                    "Failed to sync active WCS: device timed out "
                    "while responding to $G command."
                )
            except asyncio.CancelledError:
                logger.debug("Active WCS sync cancelled by task manager.")
                raise
            except DeviceConnectionError as e:
                logger.error(
                    f"Failed to sync active WCS: connection error: {e}"
                )

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

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """Check if machine's supports jogging for the given axis."""
        if self.driver is None:
            return False
        return self.driver.can_jog(axis)

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
        async with self.machine._settings_lock:
            logger.debug("_read_from_device: Lock acquired.")
            if self.driver is None:
                err = ConnectionError("No driver instance for this machine.")
                self.machine.settings_error.send(self, error=err)
                return

            def on_settings_read(sender, settings: List["VarSet"]):
                logger.debug("on_settings_read: Handler called.")
                sender.settings_read.disconnect(on_settings_read)
                self.machine._scheduler(
                    self.machine.settings_updated.send,
                    self.machine,
                    var_sets=settings,
                )
                logger.debug("on_settings_read: Handler finished.")

            self.driver.settings_read.connect(on_settings_read)
            try:
                await self.driver.read_settings()
            except (DeviceConnectionError, ConnectionError) as e:
                logger.error(f"Failed to read settings from device: {e}")
                self.driver.settings_read.disconnect(on_settings_read)
                self.machine._scheduler(
                    self.machine.settings_error.send, self, error=e
                )
            finally:
                logger.debug("_read_from_device: Read operation finished.")
        logger.debug("_read_from_device: Lock released.")

    async def _write_setting_to_device(self, key: str, value: Any):
        """
        Writes a single setting to the device and signals success or failure.
        """
        logger.debug(f"_write_setting_to_device(key={key}): Acquiring lock.")
        if self.driver is None:
            err = ConnectionError("No driver instance for this machine.")
            self.machine.settings_error.send(self, error=err)
            return

        try:
            async with self.machine._settings_lock:
                logger.debug(
                    f"_write_setting_to_device(key={key}): Lock acquired."
                )
                await self.driver.write_setting(key, value)
                self.machine._scheduler(
                    self.machine.setting_applied.send, self
                )
        except (DeviceConnectionError, ConnectionError) as e:
            logger.error(f"Failed to write setting to device: {e}")
            self.machine._scheduler(
                self.machine.settings_error.send, self, error=e
            )
        finally:
            logger.debug(f"_write_setting_to_device(key={key}): Done.")

    def validate_driver_setup(self) -> Tuple[bool, Optional[str]]:
        """
        Validates the machine's driver arguments against the driver's setup
        VarSet.

        Returns:
            A tuple of (is_valid, error_message).
        """
        if not self.machine.driver_name:
            return False, _("No driver selected for this machine.")

        driver_cls = get_driver_cls(self.machine.driver_name)
        if not driver_cls:
            return False, _("Driver '{driver}' not found.").format(
                driver=self.machine.driver_name
            )

        try:
            setup_vars = driver_cls.get_setup_vars()
            setup_vars.set_values(self.machine.driver_args)
            setup_vars.validate()
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, _(
                "An unexpected error occurred during validation: {error}"
            ).format(error=str(e))

        return True, None
