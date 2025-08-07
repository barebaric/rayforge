import yaml
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Type
from pathlib import Path
from blinker import Signal
from ...shared.util.glib import idle_add
from ...shared.tasker import task_mgr
from ...shared.varset import ValidationError
from ...camera.models.camera import Camera
from ..transport import TransportStatus
from ..driver.driver import (
    Driver,
    DeviceConnectionError,
    DeviceState,
    DeviceStatus,
    DriverSetupError,
)
from ..driver.dummy import NoDeviceDriver
from ..driver import get_driver_cls
from .laser import Laser

if TYPE_CHECKING:
    from ...shared.varset import VarSet
    from ...shared.tasker.context import ExecutionContext


logger = logging.getLogger(__name__)


class Machine:
    def __init__(self):
        logger.debug("Machine.__init__")
        self.id = str(uuid.uuid4())
        self.name: str = _("Default Machine")

        self.connection_status: TransportStatus = TransportStatus.DISCONNECTED
        self.device_state: DeviceState = DeviceState()

        self.driver_name: Optional[str] = None
        self.driver_args: Dict[str, Any] = {}

        self.driver: Driver = NoDeviceDriver()
        self._connect_driver_signals()

        self.home_on_start: bool = False
        self.dialect_name: str = "GRBL"
        self.use_custom_preamble: bool = False
        self.preamble: List[str] = []
        self.use_custom_postscript: bool = False
        self.postscript: List[str] = []
        self.heads: List[Laser] = []
        self._heads_ref_for_pyreverse: Laser
        self.cameras: List[Camera] = []
        self._cameras_ref_for_pyreverse: Camera
        self.max_travel_speed: int = 3000  # in mm/min
        self.max_cut_speed: int = 1000  # in mm/min
        self.dimensions: Tuple[int, int] = 200, 200
        self.y_axis_down: bool = False
        self._settings_lock = asyncio.Lock()

        # Signals
        self.changed = Signal()
        self.settings_error = Signal()
        self.settings_updated = Signal()
        self.setting_applied = Signal()
        self.connection_status_changed = Signal()
        self.state_changed = Signal()

        self.add_head(Laser())

    def _connect_driver_signals(self):
        self.driver.connection_status_changed.connect(
            self._on_driver_connection_status_changed
        )
        self.driver.state_changed.connect(self._on_driver_state_changed)
        self._on_driver_state_changed(self.driver, self.driver.state)
        self._reset_status()

    def _disconnect_driver_signals(self):
        if not self.driver:
            return
        self.driver.connection_status_changed.disconnect(
            self._on_driver_connection_status_changed
        )
        self.driver.state_changed.disconnect(self._on_driver_state_changed)

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

        driver_cls = None
        if self.driver_name:
            driver_cls = get_driver_cls(self.driver_name)

        if driver_cls:
            new_driver = driver_cls()
        else:
            if self.driver_name:
                logger.warning(
                    f"Driver '{self.driver_name}' not found for machine "
                    f"'{self.name}'. Falling back to NoDeviceDriver."
                )
            new_driver = NoDeviceDriver()

        try:
            new_driver.setup(**self.driver_args)
        except DriverSetupError as e:
            logger.error(f"Setup failed for driver {self.driver_name}: {e}")
            new_driver.setup_error = str(e)

        self.driver = new_driver

        self._connect_driver_signals()
        if not self.driver.setup_error:
            # Add the connect task with a key unique to this machine
            task_mgr.add_coroutine(
                lambda ctx: self.driver.connect(),
                key=(self.id, "driver-connect"),
            )

        # Notify the UI of the change *after* the new driver is in place.
        # This MUST be done on the main thread to prevent UI corruption.
        idle_add(self.changed.send, self)

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
            idle_add(self.state_changed.send, self, state=self.device_state)
        if conn_actually_changed:
            idle_add(
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
            idle_add(
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
            idle_add(self.state_changed.send, self, state=state)

    def is_connected(self) -> bool:
        """
        Checks if the machine's driver is currently connected to the device.

        Returns:
            True if connected, False otherwise.
        """
        return self.connection_status == TransportStatus.CONNECTED

    def set_name(self, name: str):
        self.name = str(name)
        self.changed.send(self)

    def set_driver(self, driver_cls: Type[Driver], args=None):
        new_driver_name = driver_cls.__name__
        if self.driver_name == new_driver_name and self.driver_args == (
            args or {}
        ):
            return

        self.driver_name = new_driver_name
        self.driver_args = args or {}
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

    def set_dialect_name(self, dialect_name: str):
        if self.dialect_name == dialect_name:
            return
        self.dialect_name = dialect_name
        self.changed.send(self)

    def set_home_on_start(self, home_on_start: bool = True):
        self.home_on_start = home_on_start
        self.changed.send(self)

    def set_use_custom_preamble(self, use: bool):
        if self.use_custom_preamble == use:
            return
        self.use_custom_preamble = use
        self.changed.send(self)

    def set_preamble(self, preamble: List[str]):
        self.preamble = preamble
        self.changed.send(self)

    def set_use_custom_postscript(self, use: bool):
        if self.use_custom_postscript == use:
            return
        self.use_custom_postscript = use
        self.changed.send(self)

    def set_postscript(self, postscript: List[str]):
        self.postscript = postscript
        self.changed.send(self)

    def set_max_travel_speed(self, speed: int):
        self.max_travel_speed = speed
        self.changed.send(self)

    def set_max_cut_speed(self, speed: int):
        self.max_cut_speed = speed
        self.changed.send(self)

    def set_dimensions(self, width: int, height: int):
        self.dimensions = (width, height)
        self.changed.send(self)

    def set_y_axis_down(self, y_axis_down: bool):
        self.y_axis_down = y_axis_down
        self.changed.send(self)

    def add_head(self, head: Laser):
        self.heads.append(head)
        head.changed.connect(self._on_head_changed)
        self.changed.send(self)

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

    def can_frame(self):
        for head in self.heads:
            if head.frame_power:
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
        return self.driver.get_setting_vars()

    async def _read_from_device(self):
        """
        Task entry point for reading settings. This handles locking and
        all errors.
        """
        logger.debug("Machine._read_from_device: Acquiring lock.")
        async with self._settings_lock:
            logger.debug("_read_from_device: Lock acquired.")
            if not self.driver:
                err = ConnectionError("No driver instance for this machine.")
                self.settings_error.send(self, error=err)
                return

            def on_settings_read(sender, settings: List["VarSet"]):
                logger.debug("on_settings_read: Handler called.")
                sender.settings_read.disconnect(on_settings_read)
                idle_add(self.settings_updated.send, self, var_sets=settings)
                logger.debug("on_settings_read: Handler finished.")

            self.driver.settings_read.connect(on_settings_read)
            try:
                await self.driver.read_settings()
            except (DeviceConnectionError, ConnectionError) as e:
                logger.error(f"Failed to read settings from device: {e}")
                self.driver.settings_read.disconnect(on_settings_read)
                idle_add(self.settings_error.send, self, error=e)
            finally:
                logger.debug("_read_from_device: Read operation finished.")
        logger.debug("_read_from_device: Lock released.")

    async def _write_setting_to_device(self, key: str, value: Any):
        """
        Writes a single setting to the device and signals success or failure.
        It no longer triggers an automatic re-read.
        """
        logger.debug(f"_write_setting_to_device(key={key}): Acquiring lock.")
        if not self.driver:
            err = ConnectionError("No driver instance for this machine.")
            self.settings_error.send(self, error=err)
            return

        try:
            async with self._settings_lock:
                logger.debug(
                    f"_write_setting_to_device(key={key}): Lock acquired."
                )
                await self.driver.write_setting(key, value)
                idle_add(self.setting_applied.send, self)
        except (DeviceConnectionError, ConnectionError) as e:
            logger.error(f"Failed to write setting to device: {e}")
            idle_add(self.settings_error.send, self, error=e)
        finally:
            logger.debug(f"_write_setting_to_device(key={key}): Done.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": {
                "name": self.name,
                "driver": self.driver_name,
                "driver_args": self.driver_args,
                "home_on_start": self.home_on_start,
                "dialect": self.dialect_name,
                "dimensions": list(self.dimensions),
                "y_axis_down": self.y_axis_down,
                "heads": [head.to_dict() for head in self.heads],
                "cameras": [camera.to_dict() for camera in self.cameras],
                "speeds": {
                    "max_cut_speed": self.max_cut_speed,
                    "max_travel_speed": self.max_travel_speed,
                },
                "gcode": {
                    "preamble": self.preamble,
                    "postscript": self.postscript,
                    "use_custom_preamble": self.use_custom_preamble,
                    "use_custom_postscript": self.use_custom_postscript,
                },
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Machine":
        ma = cls()
        ma_data = data.get("machine", {})
        ma.name = ma_data.get("name", ma.name)
        ma.driver_name = ma_data.get("driver")
        ma.driver_args = ma_data.get("driver_args", {})
        ma.home_on_start = ma_data.get("home_on_start", ma.home_on_start)
        ma.dialect_name = ma_data.get("dialect", "GRBL")
        ma.dimensions = tuple(ma_data.get("dimensions", ma.dimensions))
        ma.y_axis_down = ma_data.get("y_axis_down", ma.y_axis_down)
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
        gcode = ma_data.get("gcode", {})

        # Load preamble/postscript values. They might be None in old files.
        preamble = gcode.get("preamble")
        postscript = gcode.get("postscript")
        ma.preamble = preamble if preamble is not None else []
        ma.postscript = postscript if postscript is not None else []

        # Load override flags. If they don't exist (old file),
        # infer state from whether preamble/postscript were defined.
        ma.use_custom_preamble = gcode.get(
            "use_custom_preamble", preamble is not None
        )
        ma.use_custom_postscript = gcode.get(
            "use_custom_postscript", postscript is not None
        )

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

    def create_default_machine(self):
        machine = Machine()
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
