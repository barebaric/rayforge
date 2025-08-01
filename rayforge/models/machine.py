import yaml
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from blinker import Signal
from .camera import Camera
from .laser import Laser


logger = logging.getLogger(__name__)


class Machine:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.name: str = _('Default Machine')
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
        self.dimensions: Tuple[int, int] = 200, 200
        self.changed = Signal()
        self.y_axis_down: bool = False
        self.add_head(Laser())

    def set_name(self, name: str):
        self.name = str(name)
        self.changed.send(self)

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": {
                "name": self.name,
                "driver": self.driver,
                "driver_args": self.driver_args,
                "home_on_start": self.home_on_start,
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
        ma.y_axis_down = ma_data.get("y_axis_down", ma.y_axis_down)
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
    def __init__(self, base_dir: Path):
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
        logger.debug("Saving machine")
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
