import yaml
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from blinker import Signal
from rayforge.models.camera import Camera


logger = logging.getLogger(__name__)


class Laser:
    def __init__(self):
        self.max_power: int = 1000  # Max power (0-1000 for GRBL)
        self.frame_power: int = 0  # 0 = framing not supported
        self.spot_size_mm: tuple[float, float] = 0.1, 0.1  # millimeters
        self.changed = Signal()

    def set_max_power(self, power):
        self.max_power = power
        self.changed.send(self)

    def set_frame_power(self, power):
        self.frame_power = power
        self.changed.send(self)

    def set_spot_size(self, spot_size_x_mm, spot_size_y_mm):
        self.spot_size_mm = spot_size_x_mm, spot_size_y_mm
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_power": self.max_power,
            "frame_power": self.frame_power,
            "spot_size_mm": self.spot_size_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Laser':
        lh = cls()
        lh.max_power = data.get("max_power", lh.max_power)
        lh.frame_power = data.get("frame_power", lh.frame_power)
        lh.spot_size_mm = data.get("spot_size_mm", lh.spot_size_mm)
        return lh


class Machine:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.name: str = 'Default Machine'
        self.driver: Optional[str] = None
        self.driver_args: Dict[str, Any] = {}
        self.home_on_start: bool = False
        self.preamble: List[str] = ["G21 ; Set units to mm",
                                    "G90 ; Absolute positioning"]
        self.postscript: List[str] = ["G0 X0 Y0 ; Return to origin"]
        self.air_assist_on = "M8 ; Enable air assist"
        self.air_assist_off = "M9 ; Disable air assist"
        self.heads: List[Laser] = []
        self._heads_ref_for_pyreverse: Laser
        self.cameras: List[Camera] = []
        self._cameras_ref_for_pyreverse: Camera
        self.max_travel_speed: int = 3000   # in mm/min
        self.max_cut_speed: int = 1000   # in mm/min
        self.dimensions: tuple[int, int] = 200, 200
        self.changed = Signal()
        self.add_head(Laser())

    def set_driver(self, driver_cls: type, args=None):
        self.driver = driver_cls.__name__
        self.driver_args = args or {}
        self.changed.send(self)

    def set_driver_args(self, args=None):
        self.driver_args = args or {}
        self.changed.send(self)

    def set_home_on_start(self, home_on_start: bool = True):
        self.home_on_start = home_on_start
        self.changed.send(self)

    def set_preamble(self, preamble: List[str]):
        self.preamble = preamble
        self.changed.send(self)

    def set_postscript(self, postscript: List[str]):
        self.postscript = postscript
        self.changed.send(self)

    def set_air_assist_on(self, gcode: Optional[str]):
        self.air_assist_on = gcode
        self.changed.send(self)

    def set_air_assist_off(self, gcode: Optional[str]):
        self.air_assist_off = gcode
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": {
                "name": self.name,
                "driver": self.driver,
                "driver_args": self.driver_args,
                "home_on_start": self.home_on_start,
                "dimensions": list(self.dimensions),
                "heads": [head.to_dict() for head in self.heads],
                "cameras": [camera.to_dict() for camera in self.cameras],
                "speeds": {
                    "max_cut_speed": self.max_cut_speed,
                    "max_travel_speed": self.max_travel_speed,
                },
                "gcode": {
                    "preamble": self.preamble,
                    "postscript": self.postscript,
                    "air_assist_on": self.air_assist_on,
                    "air_assist_off": self.air_assist_off,
                },
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Machine':
        ma = cls()
        ma_data = data.get("machine", {})
        ma.name = ma_data.get("name", ma.name)
        ma.driver = ma_data.get("driver")
        ma.driver_args = ma_data.get("driver_args", {})
        ma.home_on_start = ma_data.get("home_on_start", ma.home_on_start)
        ma.dimensions = tuple(ma_data.get("dimensions", ma.dimensions))
        ma.heads = []
        for obj in ma_data.get("heads", {}):
            ma.add_head(Laser.from_dict(obj))
        ma.cameras = []
        for obj in ma_data.get("cameras", {}):
            ma.add_camera(Camera.from_dict(obj))
        speeds = ma_data.get("speeds", {})
        ma.max_cut_speed = speeds.get("max_cut_speed", ma.max_cut_speed)
        ma.max_travel_speed = speeds.get("max_travel_speed",
                                         ma.max_travel_speed)
        gcode = ma_data.get("gcode", {})
        ma.preamble = gcode.get("preamble", ma.preamble)
        ma.postscript = gcode.get("postscript", ma.postscript)
        ma.air_assist_on = gcode.get("air_assist_on", ma.air_assist_on)
        ma.air_assist_off = gcode.get("air_assist_off", ma.air_assist_off)
        return ma


class MachineManager:
    def __init__(self, base_dir):
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.machines: Dict[str, Machine] = dict()
        self._machine_ref_for_pyreverse: Machine
        self.load()

    def filename_from_id(self, machine_id: str) -> Path:
        return self.base_dir / f"{machine_id}.yaml"

    def add_machine(self, machine):
        if machine.id in self.machines:
            return
        self.machines[machine.id] = machine
        machine.changed.connect(self.on_machine_changed)

    def get_machine_by_id(self, machine_id):
        return self.machines.get(machine_id)

    def create_default_machine(self):
        machine = Machine()
        self.add_machine(machine)
        self.save_machine(machine)
        return machine

    def save_machine(self, machine):
        machine_file = self.filename_from_id(machine.id)
        with open(machine_file, 'w') as f:
            data = machine.to_dict()
            yaml.safe_dump(data, f)

    def load_machine(self, machine_id: str) -> Optional['Machine']:
        machine_file = self.filename_from_id(machine_id)
        if not machine_file.exists():
            raise FileNotFoundError(f"Machine file {machine_file} not found")
        with open(machine_file, 'r') as f:
            data = yaml.safe_load(f)
            if not data:
                msg = f"skipping invalid machine file {f.name}"
                logger.warning(msg)
                return None
        machine = Machine.from_dict(data)
        machine.id = machine_id
        self.add_machine(machine)
        return machine

    def on_machine_changed(self, machine, **kwargs):
        self.save_machine(machine)

    def load(self):
        for file in self.base_dir.glob("*.yaml"):
            machine = self.load_machine(file.stem)
            if machine:
                self.add_machine(machine)
