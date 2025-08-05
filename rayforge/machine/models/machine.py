import yaml
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Type
from pathlib import Path
from blinker import Signal
from ...shared.tasker import task_mgr
from ...shared.varset import ValidationError
from ...camera.models.camera import Camera
from .laser import Laser
from ..driver.driver import driver_mgr, Driver, DeviceConnectionError

if TYPE_CHECKING:
    from ...shared.varset import VarSet


logger = logging.getLogger(__name__)


class Machine:
    def __init__(self):
        logger.debug("Machine.__init__")
        self.id = str(uuid.uuid4())
        self.name: str = _("Default Machine")
        self.driver: Optional[str] = None
        self.driver_args: Dict[str, Any] = {}
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

        self.add_head(Laser())

    def set_name(self, name: str):
        self.name = str(name)
        self.changed.send(self)

    def set_driver(self, driver_cls: Type[Driver], args=None):
        self.driver = driver_cls.__name__
        self.driver_args = args or {}
        self.changed.send(self)

    def set_driver_args(self, args=None):
        self.driver_args = args or {}
        self.changed.send(self)

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
        # Local import to prevent circular dependency
        from ..driver import get_driver_cls

        if not self.driver:
            return False, _("No driver selected for this machine.")

        driver_cls = get_driver_cls(self.driver)
        if not driver_cls:
            return False, _("Driver '{driver}' not found.").format(
                driver=self.driver
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
            self._read_from_device, key="device-settings-read"
        )

    def apply_setting(self, key: str, value: Any):
        """Public API for the UI to apply a single setting."""
        task_mgr.add_coroutine(
            self._write_setting_to_device,
            key,
            value,
            key="device-settings-write",
        )

    def _get_active_driver(self) -> Optional[Driver]:
        """
        Helper to get the active driver instance, but only if it matches
        this machine's configured driver type.
        """
        driver = driver_mgr.driver
        if driver and driver.__class__.__name__ == self.driver:
            return driver
        return None

    def get_setting_vars(self) -> List["VarSet"]:
        """
        Gets the setting definitions from the machine's active driver
        as a VarSet.
        """
        driver = self._get_active_driver()
        if driver:
            return driver.get_setting_vars()
        return []

    async def _read_from_device(self, ctx):
        """
        Task entry point for reading settings. This handles locking and
        all errors.
        """
        logger.debug("Machine._read_from_device: Acquiring lock.")
        async with self._settings_lock:
            logger.debug("_read_from_device: Lock acquired.")
            driver = self._get_active_driver()
            if not driver:
                err = ConnectionError(
                    "No active driver for this machine to read settings from."
                )
                self.settings_error.send(self, error=err)
                return

            def on_settings_read(sender, settings: List["VarSet"]):
                logger.debug("on_settings_read: Handler called.")
                sender.settings_read.disconnect(on_settings_read)
                self.settings_updated.send(self, var_sets=settings)
                logger.debug("on_settings_read: Handler finished.")

            driver.settings_read.connect(on_settings_read)
            try:
                await driver.read_settings()
            except (DeviceConnectionError, ConnectionError) as e:
                logger.error(f"Failed to read settings from device: {e}")
                driver.settings_read.disconnect(on_settings_read)
                self.settings_error.send(self, error=e)
            finally:
                logger.debug("_read_from_device: Read operation finished.")
        logger.debug("_read_from_device: Lock released.")

    async def _write_setting_to_device(self, ctx, key: str, value: Any):
        """
        Writes a single setting to the device and signals success or failure.
        It no longer triggers an automatic re-read.
        """
        logger.debug(f"_write_setting_to_device(key={key}): Acquiring lock.")
        driver = self._get_active_driver()
        if not driver:
            err = ConnectionError(
                "No active driver for this machine to write settings to."
            )
            self.settings_error.send(self, error=err)
            return

        try:
            async with self._settings_lock:
                logger.debug(
                    f"_write_setting_to_device(key={key}): Lock acquired."
                )
                await driver.write_setting(key, value)
                self.setting_applied.send(self)
        except (DeviceConnectionError, ConnectionError) as e:
            logger.error(f"Failed to write setting to device: {e}")
            self.settings_error.send(self, error=e)
        finally:
            logger.debug(
                f"_write_setting_to_device(key={key}): Operation finished."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": {
                "name": self.name,
                "driver": self.driver,
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
        ma.driver = ma_data.get("driver")
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
