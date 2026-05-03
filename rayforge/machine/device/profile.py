import logging
import shutil
from dataclasses import asdict, dataclass, fields as dc_fields
from gettext import gettext as _
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import yaml

from ...core.model import Model
from ...machine.driver import get_driver_cls
from ...machine.models.dialect import GcodeDialect
from ...machine.models.laser import Laser
from ...machine.models.machine import Machine, Origin
from ...machine.models.macro import Macro, MacroTrigger
from ...machine.models.rotary_module import RotaryModule
from ...machine.models.zone import Zone

if TYPE_CHECKING:
    from ...context import RayforgeContext
    from ...core.model_manager import ModelManager

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "device.yaml"
DIALECT_FILENAME = "dialect.yaml"

CURRENT_API_VERSION = 1


@dataclass
class DeviceMeta:
    name: str
    vendor: str = ""
    model: str = ""
    description: str = ""
    api_version: int = CURRENT_API_VERSION


_DEVICE_META_FIELDS = frozenset(f.name for f in dc_fields(DeviceMeta))


def parse_meta(data: Dict, manifest_path: Path) -> DeviceMeta:
    api_version = data.get("api_version", 1)
    if api_version != CURRENT_API_VERSION:
        raise ValueError(
            f"Unsupported api_version {api_version} in "
            f"{manifest_path} (expected {CURRENT_API_VERSION})"
        )

    device_data = data.get("device", {})
    if not isinstance(device_data, dict):
        raise ValueError(f"Invalid 'device' section in {manifest_path}")

    if not device_data.get("name"):
        raise ValueError(f"Missing 'device.name' in {manifest_path}")

    filtered = {
        k: v for k, v in device_data.items() if k in _DEVICE_META_FIELDS
    }
    filtered.setdefault("api_version", api_version)
    return DeviceMeta(**filtered)


_TUPLE_FIELDS = frozenset({"axis_extents", "work_margins", "soft_limits"})

_VALID_ORIGINS = sorted(o.value for o in Origin)


def parse_machine_config(data: Dict, manifest_path: Path) -> "MachineConfig":
    raw = data.get("machine", {})
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid 'machine' section in {manifest_path}")

    _validate_machine_config(raw, manifest_path)

    kwargs = {}
    for key in _MACHINE_CONFIG_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key == "origin":
            try:
                value = Origin(value)
            except ValueError:
                raise ValueError(
                    f"Invalid origin '{raw['origin']}' in "
                    f"{manifest_path}. Must be one of: "
                    f"{_VALID_ORIGINS}"
                )
        elif key in _TUPLE_FIELDS:
            value = tuple(value)
        kwargs[key] = value

    return MachineConfig(**kwargs)


def _validate_machine_config(config: Dict[str, Any], manifest_path: Path):
    for key in config:
        if key not in _MACHINE_CONFIG_KEYS:
            logger.warning(
                f"Unknown machine config key '{key}' in {manifest_path}"
            )

    if "axis_extents" in config:
        ae = config["axis_extents"]
        if (
            not isinstance(ae, list)
            or len(ae) != 2
            or not all(isinstance(v, (int, float)) for v in ae)
        ):
            raise ValueError(
                f"'axis_extents' must be a list of two numbers "
                f"in {manifest_path}"
            )

    if "work_margins" in config:
        wm = config["work_margins"]
        if (
            not isinstance(wm, list)
            or len(wm) != 4
            or not all(isinstance(v, (int, float)) for v in wm)
        ):
            raise ValueError(
                f"'work_margins' must be a list of four numbers "
                f"in {manifest_path}"
            )

    if "soft_limits" in config:
        sl = config["soft_limits"]
        if (
            not isinstance(sl, list)
            or len(sl) != 4
            or not all(isinstance(v, (int, float)) for v in sl)
        ):
            raise ValueError(
                f"'soft_limits' must be a list of four numbers "
                f"in {manifest_path}"
            )

    if "heads" in config:
        if not isinstance(config["heads"], list):
            raise ValueError(f"'heads' must be a list in {manifest_path}")

    if "hookmacros" in config:
        if not isinstance(config["hookmacros"], list):
            raise ValueError(f"'hookmacros' must be a list in {manifest_path}")

    if "rotary_modules" in config:
        if not isinstance(config["rotary_modules"], list):
            raise ValueError(
                f"'rotary_modules' must be a list in {manifest_path}"
            )

    if "nogo_zones" in config:
        if not isinstance(config["nogo_zones"], list):
            raise ValueError(f"'nogo_zones' must be a list in {manifest_path}")


def _resolve_and_copy_models(
    device_data: dict,
    dest_dir: Path,
    model_mgr: "ModelManager",
):
    machine_section = device_data["machine"]
    models_dir = dest_dir / "models"
    copied: set = set()

    for head_data in machine_section.get("heads", []):
        _copy_model_ref(head_data, models_dir, model_mgr, copied)

    for rm_data in machine_section.get("rotary_modules", []):
        _copy_model_ref(rm_data, models_dir, model_mgr, copied)


def _copy_model_ref(
    data: dict,
    models_dir: Path,
    model_mgr: "ModelManager",
    copied: set,
):
    mp = data.get("model_path")
    if not mp:
        return
    resolved = model_mgr.resolve(Model.from_path(Path(mp)))
    if resolved is None or not resolved.is_file():
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    filename = resolved.name
    if filename not in copied:
        shutil.copy2(resolved, models_dir / filename)
        copied.add(filename)
    data["model_path"] = filename


@dataclass
class MachineConfig:
    driver: Optional[str] = None
    driver_args: Optional[Dict[str, Any]] = None
    driver_config: Optional[Dict[str, Any]] = None
    gcode_precision: Optional[int] = None
    supports_arcs: Optional[bool] = None
    supports_curves: Optional[bool] = None
    axis_extents: Optional[Tuple[float, float]] = None
    work_margins: Optional[Tuple[float, float, float, float]] = None
    soft_limits: Optional[Tuple[float, float, float, float]] = None
    origin: Optional[Origin] = None
    max_travel_speed: Optional[int] = None
    max_cut_speed: Optional[int] = None
    home_on_start: Optional[bool] = None
    rotary_enabled_default: Optional[bool] = None
    heads: Optional[List[Dict[str, Any]]] = None
    hookmacros: Optional[List[Dict[str, Any]]] = None
    rotary_modules: Optional[List[Dict[str, Any]]] = None
    nogo_zones: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            if isinstance(value, tuple):
                result[key] = list(value)
            elif isinstance(value, Origin):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @classmethod
    def from_machine(cls, machine: Machine) -> "MachineConfig":
        heads = None
        if machine.heads:
            heads = []
            for head in machine.heads:
                d = head.to_dict()
                d.pop("uid", None)
                heads.append(d)

        rotary_modules = None
        if machine.rotary_modules:
            rotary_modules = []
            for rm in machine.rotary_modules.values():
                d = rm.to_dict()
                d.pop("uid", None)
                rotary_modules.append(d)

        hookmacros = None
        if machine.hookmacros:
            hookmacros = []
            for trigger, macro in machine.hookmacros.items():
                d = macro.to_dict()
                d["trigger"] = trigger.name
                hookmacros.append(d)

        nogo_zones = None
        if machine.nogo_zones:
            nogo_zones = [z.to_dict() for z in machine.nogo_zones.values()]

        work_margins = None
        if machine.work_margins != (0, 0, 0, 0):
            work_margins = machine.work_margins

        driver_config = None
        if machine.driver_config:
            driver_config = machine.driver_config.copy()

        return cls(
            driver=machine.driver_name or None,
            driver_args=machine.driver_args or None,
            driver_config=driver_config,
            gcode_precision=machine.gcode_precision,
            supports_arcs=machine.supports_arcs,
            supports_curves=machine.supports_curves,
            axis_extents=machine.axis_extents,
            work_margins=work_margins,
            soft_limits=machine.soft_limits,
            origin=machine.origin,
            max_travel_speed=machine.max_travel_speed,
            max_cut_speed=machine.max_cut_speed,
            home_on_start=machine.home_on_start,
            rotary_enabled_default=machine.rotary_enabled_default,
            heads=heads,
            hookmacros=hookmacros,
            rotary_modules=rotary_modules,
            nogo_zones=nogo_zones,
        )


_MACHINE_CONFIG_KEYS = frozenset(f.name for f in dc_fields(MachineConfig))


@dataclass
class DeviceProfile:
    meta: DeviceMeta
    machine_config: MachineConfig
    dialect_config: Dict[str, Any]
    source_dir: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.meta.name

    @classmethod
    def from_path(cls, path: Path) -> "DeviceProfile":
        """
        Load a device profile from a directory containing a
        ``device.yaml`` manifest. A ``dialect.yaml`` file is required
        for G-code drivers but optional for non-G-code drivers.

        Args:
            path: Path to the directory containing device.yaml.

        Returns:
            A DeviceProfile instance.

        Raises:
            FileNotFoundError: If device.yaml is not found.
            FileNotFoundError: If dialect.yaml is not found and the
                driver uses G-code.
            ValueError: If the manifest or dialect is invalid.
        """
        manifest_path = path / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Device manifest not found: {manifest_path}"
            )

        with open(manifest_path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid device manifest: {manifest_path} is not a mapping"
            )

        meta = parse_meta(data, manifest_path)
        machine_config = parse_machine_config(data, manifest_path)

        driver_uses_gcode = True
        if machine_config.driver:
            driver_cls = get_driver_cls(machine_config.driver)
            driver_uses_gcode = driver_cls.uses_gcode

        dialect_config: Dict[str, Any] = {}
        dialect_path = path / DIALECT_FILENAME
        if dialect_path.exists():
            with open(dialect_path, "r") as f:
                dialect_config = yaml.safe_load(f)
            GcodeDialect.validate_template_dict(
                dialect_config, str(dialect_path)
            )
        elif driver_uses_gcode:
            raise FileNotFoundError(f"Dialect file not found: {dialect_path}")

        return cls(
            meta=meta,
            machine_config=machine_config,
            dialect_config=dialect_config,
            source_dir=path,
        )

    def create_machine(self, context: "RayforgeContext") -> Machine:
        m = Machine(context)
        m.name = self.meta.name
        cfg = self.machine_config

        context.machine_mgr.add_machine(m)

        driver_uses_gcode = True
        if cfg.driver:
            driver_cls = get_driver_cls(cfg.driver)
            driver_uses_gcode = driver_cls.uses_gcode
            try:
                m.set_driver(driver_cls, cfg.driver_args)
            except Exception as exc:
                logger.error(
                    f"Failed to create driver {cfg.driver} "
                    f"for device '{self.name}': {exc}"
                )

        if cfg.driver_config is not None:
            m.driver_config = cfg.driver_config.copy()

        if driver_uses_gcode and self.dialect_config:
            new_label = _("{name} (device dialect)").format(name=self.name)
            dialect = GcodeDialect.from_template_dict(
                self.dialect_config,
                label=new_label,
                description="",
                is_custom=True,
            )
            context.dialect_mgr.add_dialect(dialect)
            m.dialect_uid = dialect.uid
        elif not driver_uses_gcode:
            m.dialect_uid = None

        if cfg.gcode_precision is not None:
            m.gcode_precision = cfg.gcode_precision
        if cfg.supports_arcs is not None:
            m.supports_arcs = cfg.supports_arcs
        if cfg.supports_curves is not None:
            m.supports_curves = cfg.supports_curves
        if cfg.axis_extents is not None:
            m.set_axis_extents(*cfg.axis_extents)
        if cfg.work_margins is not None:
            m.set_work_margins(*cfg.work_margins)
        if cfg.soft_limits is not None:
            m.set_soft_limits(*cfg.soft_limits)
        if cfg.origin is not None:
            m.origin = cfg.origin
        if cfg.max_travel_speed is not None:
            m.max_travel_speed = cfg.max_travel_speed
        if cfg.max_cut_speed is not None:
            m.max_cut_speed = cfg.max_cut_speed
        if cfg.home_on_start is not None:
            m.home_on_start = cfg.home_on_start
        if cfg.rotary_enabled_default is not None:
            m.rotary_enabled_default = cfg.rotary_enabled_default

        if cfg.hookmacros is not None:
            for s_data in cfg.hookmacros:
                try:
                    trigger = MacroTrigger[s_data["trigger"]]
                    m.hookmacros[trigger] = Macro.from_dict(s_data)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid hook in device: {e}")

        m.cameras = []

        if cfg.heads is not None:
            for head in m.heads[:]:
                m.remove_head(head)
            for head_data in cfg.heads:
                m.add_head(Laser.from_dict(head_data))

        if cfg.rotary_modules is not None:
            for rm_data in cfg.rotary_modules:
                m.add_rotary_module(RotaryModule.from_dict(rm_data))

        if cfg.nogo_zones is not None:
            for z_data in cfg.nogo_zones:
                m.add_nogo_zone(Zone.from_dict(z_data))

        return m


def export_machine_to_dir(
    machine: Machine,
    dest_dir: Path,
    model_mgr: Optional["ModelManager"] = None,
) -> DeviceProfile:
    dest_dir.mkdir(parents=True, exist_ok=True)

    mc = MachineConfig.from_machine(machine)
    device_data = {
        "api_version": CURRENT_API_VERSION,
        "device": {"name": machine.name},
        "machine": mc.to_dict(),
    }

    if model_mgr is not None:
        _resolve_and_copy_models(device_data, dest_dir, model_mgr)

    with open(dest_dir / MANIFEST_FILENAME, "w") as f:
        yaml.safe_dump(device_data, f, sort_keys=False)

    if machine.dialect is not None:
        with open(dest_dir / DIALECT_FILENAME, "w") as f:
            yaml.safe_dump(
                machine.dialect.to_template_dict(), f, sort_keys=False
            )

    return DeviceProfile.from_path(dest_dir)
