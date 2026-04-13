import logging
from dataclasses import dataclass, fields as dc_fields, MISSING
from gettext import gettext as _
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import yaml

from ...machine.driver import get_driver_cls
from ...machine.models.dialect import GcodeDialect
from ...machine.models.laser import Laser
from ...machine.models.machine import Machine, Origin
from ...machine.models.macro import Macro, MacroTrigger
from ...machine.models.rotary_module import RotaryModule
from ...machine.models.zone import Zone

if TYPE_CHECKING:
    from ...context import RayforgeContext

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "device.yaml"
DIALECT_FILENAME = "dialect.yaml"

CURRENT_API_VERSION = 1

_META_FIELDS = frozenset(
    {"label", "description", "uid", "is_custom", "parent_uid", "extra"}
)


def _dialect_template_fields() -> Tuple[frozenset, frozenset, frozenset]:
    required = set()
    optional = set()
    for f in dc_fields(GcodeDialect):
        if f.name in _META_FIELDS:
            continue
        if f.default is not MISSING:
            optional.add(f.name)
        else:
            required.add(f.name)
    return (
        frozenset(required),
        frozenset(optional),
        frozenset(required | optional),
    )


DIALECT_REQUIRED_FIELDS, DIALECT_OPTIONAL_FIELDS, DIALECT_ALL_FIELDS = (
    _dialect_template_fields()
)


@dataclass
class DeviceMeta:
    name: str
    vendor: str = ""
    model: str = ""
    description: str = ""
    api_version: int = CURRENT_API_VERSION


@dataclass
class MachineConfig:
    driver: Optional[str] = None
    driver_args: Optional[Dict[str, Any]] = None
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


MACHINE_CONFIG_KEYS = frozenset(f.name for f in dc_fields(MachineConfig))

ORIGIN_MAP = {o.value: o for o in Origin}


@dataclass
class DevicePackage:
    meta: DeviceMeta
    machine_config: MachineConfig
    dialect_config: Dict[str, Any]
    source_dir: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.meta.name

    def create_machine(self, context: "RayforgeContext") -> Machine:
        m = Machine(context)
        m.name = self.meta.name
        cfg = self.machine_config

        if cfg.driver:
            try:
                driver_cls = get_driver_cls(cfg.driver)
                m.set_driver(driver_cls, cfg.driver_args)
            except (ValueError, ImportError):
                logger.error(
                    f"Failed to create driver {cfg.driver} "
                    f"for device '{self.name}'"
                )

        context.machine_mgr.add_machine(m)

        new_label = _("{name} (device dialect)").format(name=self.name)
        dialect = GcodeDialect(
            label=new_label,
            description="",
            is_custom=True,
            **{
                k: v
                for k, v in self.dialect_config.items()
                if k in DIALECT_ALL_FIELDS
            },
        )
        context.dialect_mgr.add_dialect(dialect)
        m.dialect_uid = dialect.uid

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


def load(path: Path) -> DevicePackage:
    """
    Load a device package from a directory containing a device.yaml
    manifest and a dialect.yaml file.

    Args:
        path: Path to the directory containing device.yaml.

    Returns:
        A DevicePackage instance.

    Raises:
        FileNotFoundError: If device.yaml or dialect.yaml is not found.
        ValueError: If the manifest or dialect is invalid.
    """
    manifest_path = path / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Device manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid device manifest: {manifest_path} is not a mapping"
        )

    meta = _parse_meta(data, manifest_path)
    machine_config = _parse_machine_config(data, manifest_path)

    dialect_path = path / DIALECT_FILENAME
    if not dialect_path.exists():
        raise FileNotFoundError(f"Dialect file not found: {dialect_path}")

    dialect_config = _load_dialect(dialect_path)

    return DevicePackage(
        meta=meta,
        machine_config=machine_config,
        dialect_config=dialect_config,
        source_dir=path,
    )


def _parse_meta(data: Dict, manifest_path: Path) -> DeviceMeta:
    api_version = data.get("api_version", 1)
    if api_version != CURRENT_API_VERSION:
        raise ValueError(
            f"Unsupported api_version {api_version} in {manifest_path} "
            f"(expected {CURRENT_API_VERSION})"
        )

    device_data = data.get("device", {})
    if not isinstance(device_data, dict):
        raise ValueError(f"Invalid 'device' section in {manifest_path}")

    name = device_data.get("name")
    if not name:
        raise ValueError(f"Missing 'device.name' in {manifest_path}")

    return DeviceMeta(
        name=name,
        vendor=device_data.get("vendor", ""),
        model=device_data.get("model", ""),
        description=device_data.get("description", ""),
        api_version=api_version,
    )


def _parse_machine_config(data: Dict, manifest_path: Path) -> MachineConfig:
    raw = data.get("machine", {})
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid 'machine' section in {manifest_path}")

    _validate_machine_config(raw, manifest_path)

    origin = None
    if "origin" in raw:
        origin = ORIGIN_MAP.get(raw["origin"])
        if origin is None:
            raise ValueError(
                f"Invalid origin '{raw['origin']}' in {manifest_path}. "
                f"Must be one of: {sorted(ORIGIN_MAP.keys())}"
            )

    axis_extents = None
    if "axis_extents" in raw:
        axis_extents = tuple(raw["axis_extents"])

    work_margins = None
    if "work_margins" in raw:
        work_margins = tuple(raw["work_margins"])

    soft_limits = None
    if "soft_limits" in raw:
        soft_limits = tuple(raw["soft_limits"])

    return MachineConfig(
        driver=raw.get("driver"),
        driver_args=raw.get("driver_args"),
        gcode_precision=raw.get("gcode_precision"),
        supports_arcs=raw.get("supports_arcs"),
        supports_curves=raw.get("supports_curves"),
        axis_extents=axis_extents,
        work_margins=work_margins,
        soft_limits=soft_limits,
        origin=origin,
        max_travel_speed=raw.get("max_travel_speed"),
        max_cut_speed=raw.get("max_cut_speed"),
        home_on_start=raw.get("home_on_start"),
        rotary_enabled_default=raw.get("rotary_enabled_default"),
        heads=raw.get("heads"),
        hookmacros=raw.get("hookmacros"),
        rotary_modules=raw.get("rotary_modules"),
        nogo_zones=raw.get("nogo_zones"),
    )


def _validate_machine_config(config: Dict[str, Any], manifest_path: Path):
    for key in config:
        if key not in MACHINE_CONFIG_KEYS:
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


def _load_dialect(dialect_path: Path) -> Dict[str, Any]:
    with open(dialect_path, "r") as f:
        dialect_data = yaml.safe_load(f)

    if not isinstance(dialect_data, dict):
        raise ValueError(f"Invalid dialect file: {dialect_path}")

    missing = DIALECT_REQUIRED_FIELDS - set(dialect_data.keys())
    if missing:
        raise ValueError(
            f"Missing required dialect fields in {dialect_path}: "
            f"{sorted(missing)}"
        )

    for key in dialect_data:
        if key not in DIALECT_ALL_FIELDS:
            logger.warning(f"Unknown dialect field '{key}' in {dialect_path}")

    return dialect_data
