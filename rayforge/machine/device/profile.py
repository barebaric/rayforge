import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dc_fields
from gettext import gettext as _
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

from ...camera.models.camera import Camera
from ...camera.v4l import migrate_camera_data
from ...core.model import Model
from ...machine.driver import get_driver_cls
from ...machine.models.dialect import GcodeDialect
from ...machine.models.head import head_from_dict
from ...machine.models.machine import Machine, Origin
from ...machine.models.machine_panel import PanelOrientation
from ...machine.models.macro import Macro, MacroTrigger
from ...machine.models.rotary_module import RotaryModule
from ...machine.models.zone import Zone
from ...shared.units.system import UnitSystem

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
    id: str = ""
    usb_ids: list[tuple[int, int | None]] = field(default_factory=list)


_DEVICE_META_FIELDS = frozenset(f.name for f in dc_fields(DeviceMeta))


def _parse_usb_id(entry: str) -> tuple[int, int | None]:
    """
    Parse a ``"vid:pid"`` USB id string (hexadecimal) into an int
    pair. The pid part is optional ("vid" or "vid:*" declares any
    pid for the given vid).
    """
    text = entry.strip().lower()
    vid_text, sep, pid_text = text.partition(":")
    try:
        vid = int(vid_text, 16)
        pid: int | None = None
        if sep and pid_text.strip() != "*":
            pid = int(pid_text.strip(), 16)
    except ValueError:
        raise ValueError(f"Invalid USB id '{entry}'")
    return (vid, pid)


def _parse_usb_ids(
    raw: list[str], manifest_path: Path
) -> list[tuple[int, int | None]]:
    if not isinstance(raw, list):
        raise TypeError(f"'device.usb_ids' must be a list in {manifest_path}")
    try:
        return [_parse_usb_id(entry) for entry in raw]
    except (ValueError, AttributeError) as e:
        raise ValueError(f"{e} in {manifest_path}")


def format_usb_id(vid: int, pid: int | None) -> str:
    """
    Format a USB id pair as the canonical ``"vid:pid"`` yaml string
    (the inverse of :func:`_parse_usb_id`). A ``None`` pid yields the
    vid-only form.
    """
    if pid is None:
        return f"{vid:04x}"
    return f"{vid:04x}:{pid:04x}"


def parse_meta(data: dict, manifest_path: Path) -> DeviceMeta:
    api_version = data.get("api_version", 1)
    if api_version != CURRENT_API_VERSION:
        raise ValueError(
            f"Unsupported api_version {api_version} in "
            f"{manifest_path} (expected {CURRENT_API_VERSION})"
        )

    device_data = data.get("device", {})
    if not isinstance(device_data, dict):
        raise TypeError(f"Invalid 'device' section in {manifest_path}")

    if not device_data.get("name"):
        raise ValueError(f"Missing 'device.name' in {manifest_path}")

    filtered = {
        k: v for k, v in device_data.items() if k in _DEVICE_META_FIELDS
    }
    filtered.setdefault("api_version", api_version)
    if "usb_ids" in filtered:
        filtered["usb_ids"] = _parse_usb_ids(
            filtered["usb_ids"], manifest_path
        )
    return DeviceMeta(**filtered)


_TUPLE_FIELDS = frozenset({"axis_extents", "work_margins", "soft_limits"})

_VALID_ORIGINS = sorted(o.value for o in Origin)


def parse_machine_config(data: dict, manifest_path: Path) -> "MachineConfig":
    raw = data.get("machine", {})
    if not isinstance(raw, dict):
        raise TypeError(f"Invalid 'machine' section in {manifest_path}")

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
        elif key == "unit_system":
            try:
                value = UnitSystem(value)
            except ValueError:
                raise ValueError(
                    f"Invalid unit_system '{raw['unit_system']}' in "
                    f"{manifest_path}. Must be one of: "
                    f"{sorted(u.value for u in UnitSystem)}"
                )
        elif key == "panel_orientation":
            try:
                value = PanelOrientation(value)
            except ValueError:
                raise ValueError(
                    f"Invalid panel_orientation "
                    f"'{raw['panel_orientation']}' in {manifest_path}"
                )
        elif key in _TUPLE_FIELDS:
            value = tuple(value)
        kwargs[key] = value

    return MachineConfig(**kwargs)


def _validate_machine_config(config: dict[str, Any], manifest_path: Path):
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

    if "heads" in config and not isinstance(config["heads"], list):
        raise ValueError(f"'heads' must be a list in {manifest_path}")

    if "hookmacros" in config and not isinstance(config["hookmacros"], list):
        raise ValueError(f"'hookmacros' must be a list in {manifest_path}")

    if "rotary_modules" in config and not isinstance(
        config["rotary_modules"], list
    ):
        raise ValueError(f"'rotary_modules' must be a list in {manifest_path}")

    if "nogo_zones" in config and not isinstance(config["nogo_zones"], list):
        raise ValueError(f"'nogo_zones' must be a list in {manifest_path}")

    if "cameras" in config and not isinstance(config["cameras"], list):
        raise ValueError(f"'cameras' must be a list in {manifest_path}")


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


@dataclass(frozen=True)
class SettingBinding:
    """
    Describes how a :class:`MachineConfig` field maps onto a live
    :class:`Machine`, so profile updates can be diffed and applied.

    Declared per-field in ``MachineConfig`` via the ``setting`` field
    metadata; fields without a binding are not reviewable (connection
    info, ID-bearing objects, ...).
    """

    section: str
    label: str
    get: Callable[["Machine"], Any]
    apply: Callable[["Machine", Any], None]


def _binding(
    section: str,
    label: str,
    get: Callable[["Machine"], Any],
    apply: Callable[["Machine", Any], None],
) -> dict[str, Any]:
    return {"setting": SettingBinding(section, label, get, apply)}


def _get_capabilities(machine: "Machine") -> list[str] | None:
    if machine._explicit_capabilities is None:
        return None
    return sorted(c.value for c in machine._explicit_capabilities)


@dataclass
class MachineConfig:
    driver: str | None = None
    driver_args: dict[str, Any] | None = None
    driver_config: dict[str, Any] | None = None

    gcode_precision: int | None = field(
        default=None,
        metadata=_binding(
            _("G-code"),
            _("G-code Precision"),
            lambda m: m.gcode_precision,
            lambda m, v: setattr(m, "gcode_precision", v),
        ),
    )
    supports_arcs: bool | None = field(
        default=None,
        metadata=_binding(
            _("G-code"),
            _("Supports Arcs (G2/G3)"),
            lambda m: m.supports_arcs,
            lambda m, v: setattr(m, "supports_arcs", v),
        ),
    )
    supports_curves: bool | None = field(
        default=None,
        metadata=_binding(
            _("G-code"),
            _("Supports Curves"),
            lambda m: m.supports_curves,
            lambda m, v: setattr(m, "supports_curves", v),
        ),
    )

    axis_extents: tuple[float, float] | None = field(
        default=None,
        metadata=_binding(
            _("Hardware"),
            _("Work Area Size (X × Y)"),
            lambda m: m.axis_extents,
            lambda m, v: m.set_axis_extents(*v),
        ),
    )
    work_margins: tuple[float, float, float, float] | None = field(
        default=None,
        metadata=_binding(
            _("Hardware"),
            _("Work Margins"),
            lambda m: m.work_margins,
            lambda m, v: m.set_work_margins(*v),
        ),
    )
    soft_limits: tuple[float, float, float, float] | None = field(
        default=None,
        metadata=_binding(
            _("Hardware"),
            _("Soft Limits"),
            lambda m: m.soft_limits,
            lambda m, v: m.set_soft_limits(*v),
        ),
    )
    origin: Origin | None = field(
        default=None,
        metadata=_binding(
            _("Hardware"),
            _("Coordinate Origin"),
            lambda m: m.origin,
            lambda m, v: setattr(m, "origin", v),
        ),
    )
    panel_orientation: PanelOrientation | None = field(
        default=None,
        metadata=_binding(
            _("Hardware"),
            _("Panel Orientation"),
            lambda m: m.panel_orientation,
            lambda m, v: m.set_panel_orientation(v),
        ),
    )
    has_z: bool | None = field(
        default=None,
        metadata=_binding(
            _("Hardware"),
            _("Has Z-Axis"),
            lambda m: m.has_z_axis,
            lambda m, v: m.set_has_z_axis(v),
        ),
    )

    max_travel_speed: int | None = field(
        default=None,
        metadata=_binding(
            _("Speeds"),
            _("Max Travel Speed"),
            lambda m: m.max_travel_speed,
            lambda m, v: setattr(m, "max_travel_speed", v),
        ),
    )
    max_cut_speed: int | None = field(
        default=None,
        metadata=_binding(
            _("Speeds"),
            _("Max Cut Speed"),
            lambda m: m.max_cut_speed,
            lambda m, v: setattr(m, "max_cut_speed", v),
        ),
    )
    acceleration: int | None = field(
        default=None,
        metadata=_binding(
            _("Speeds"),
            _("Acceleration"),
            lambda m: m.acceleration,
            lambda m, v: setattr(m, "acceleration", v),
        ),
    )

    home_on_start: bool | None = field(
        default=None,
        metadata=_binding(
            _("Behavior"),
            _("Home on Start"),
            lambda m: m.home_on_start,
            lambda m, v: setattr(m, "home_on_start", v),
        ),
    )
    single_axis_homing_enabled: bool | None = field(
        default=None,
        metadata=_binding(
            _("Behavior"),
            _("Single-Axis Homing"),
            lambda m: m.single_axis_homing_enabled,
            lambda m, v: setattr(m, "single_axis_homing_enabled", v),
        ),
    )
    rotary_enabled_default: bool | None = field(
        default=None,
        metadata=_binding(
            _("Behavior"),
            _("Rotary Enabled by Default"),
            lambda m: m.rotary_enabled_default,
            lambda m, v: m.set_rotary_enabled_default(v),
        ),
    )

    unit_system: UnitSystem | None = field(
        default=None,
        metadata=_binding(
            _("Units"),
            _("Unit System"),
            lambda m: m.unit_system,
            lambda m, v: setattr(m, "unit_system", v),
        ),
    )

    heads: list[dict[str, Any]] | None = None
    capabilities: list[str] | None = field(
        default=None,
        metadata=_binding(
            _("General"),
            _("Capabilities"),
            _get_capabilities,
            lambda m, v: m.set_explicit_capabilities(
                Machine._parse_capabilities(v)
            ),
        ),
    )
    hookmacros: list[dict[str, Any]] | None = None
    rotary_modules: list[dict[str, Any]] | None = None
    nogo_zones: list[dict[str, Any]] | None = None
    cameras: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            if isinstance(value, tuple):
                result[key] = list(value)
            elif isinstance(value, (Origin, PanelOrientation, UnitSystem)):
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

        cameras = None
        if machine.cameras:
            cameras = [c.to_dict() for c in machine.cameras]

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
            panel_orientation=machine.panel.orientation,
            max_travel_speed=machine.max_travel_speed,
            max_cut_speed=machine.max_cut_speed,
            home_on_start=machine.home_on_start,
            acceleration=machine.acceleration,
            single_axis_homing_enabled=(machine.single_axis_homing_enabled),
            has_z=machine.has_z_axis,
            rotary_enabled_default=machine.rotary_enabled_default,
            unit_system=machine.unit_system,
            heads=heads,
            capabilities=(
                [
                    c.value
                    for c in sorted(
                        machine._explicit_capabilities, key=lambda c: c.value
                    )
                ]
                if machine._explicit_capabilities
                else None
            ),
            hookmacros=hookmacros,
            rotary_modules=rotary_modules,
            nogo_zones=nogo_zones,
            cameras=cameras,
        )


_MACHINE_CONFIG_KEYS = frozenset(f.name for f in dc_fields(MachineConfig))


@dataclass
class DeviceProfile:
    meta: DeviceMeta
    machine_config: MachineConfig
    dialect_config: dict[str, Any]
    source_dir: Path | None = None

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def id(self) -> str:
        """Stable identifier used to link machines to this profile."""
        return self.meta.id or self.meta.name

    def content_hash(self) -> str:
        """
        Returns a hash over the profile's machine and dialect settings.

        Machines store the hash current at their last review so profile
        updates can be detected without keeping a full snapshot.
        """
        payload = {
            "machine": self.machine_config.to_dict(),
            "dialect": self.dialect_config,
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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
            raise TypeError(
                f"Invalid device manifest: {manifest_path} is not a mapping"
            )

        meta = parse_meta(data, manifest_path)
        if not meta.id:
            meta.id = path.name
        machine_config = parse_machine_config(data, manifest_path)

        driver_uses_gcode = True
        if machine_config.driver:
            driver_cls = get_driver_cls(machine_config.driver)
            driver_uses_gcode = driver_cls.uses_gcode

        dialect_config: dict[str, Any] = {}
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
        # Imported here: schema_migration transitively imports this
        # module, so a module-level import would be circular.
        from .schema_migration import CURRENT_SCHEMA_VERSION

        m = Machine(context)
        m.name = self.meta.name
        cfg = self.machine_config
        m.source_profile_id = self.id
        m.reviewed_profile_hash = self.content_hash()
        # Fresh machines already carry current defaults; stamp them so
        # they never trigger the schema-migration review dialog.
        m.schema_version = CURRENT_SCHEMA_VERSION

        context.machine_mgr.add_machine(m)

        driver_uses_gcode = True
        if cfg.driver:
            driver_cls = get_driver_cls(cfg.driver)
            driver_uses_gcode = driver_cls.uses_gcode
            try:
                m.set_driver(driver_cls, cfg.driver_args)
            except (ValueError, TypeError, RuntimeError) as exc:
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
        if cfg.panel_orientation is not None:
            # Profile cameras replace the machine's cameras below and their
            # saved alignment already uses the profile's orientation.
            m.panel._orientation = cfg.panel_orientation
        if cfg.max_travel_speed is not None:
            m.max_travel_speed = cfg.max_travel_speed
        if cfg.max_cut_speed is not None:
            m.max_cut_speed = cfg.max_cut_speed
        if cfg.home_on_start is not None:
            m.home_on_start = cfg.home_on_start
        if cfg.acceleration is not None:
            m.acceleration = cfg.acceleration
        if cfg.single_axis_homing_enabled is not None:
            m.single_axis_homing_enabled = cfg.single_axis_homing_enabled
        if cfg.has_z is not None:
            m.set_has_z_axis(cfg.has_z)
        if cfg.rotary_enabled_default is not None:
            m.rotary_enabled_default = cfg.rotary_enabled_default
        if cfg.unit_system is not None:
            m.unit_system = cfg.unit_system

        if cfg.hookmacros is not None:
            for s_data in cfg.hookmacros:
                try:
                    trigger = MacroTrigger[s_data["trigger"]]
                    m.hookmacros[trigger] = Macro.from_dict(s_data)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid hook in device: {e}")

        m.cameras = []
        if cfg.cameras is not None:
            for cam_data in cfg.cameras:
                m.add_camera(Camera.from_dict(migrate_camera_data(cam_data)))

        if cfg.heads is not None:
            for head in m.heads[:]:
                m.remove_head(head)
            for head_data in cfg.heads:
                m.add_head(head_from_dict(head_data))

        if cfg.capabilities is not None:
            m.set_explicit_capabilities(
                Machine._parse_capabilities(cfg.capabilities)
            )

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
    usb_ids: list[tuple[int, int | None]] | None = None,
) -> DeviceProfile:
    dest_dir.mkdir(parents=True, exist_ok=True)

    mc = MachineConfig.from_machine(machine)
    device_section: dict[str, Any] = {"name": machine.name}
    if machine.source_profile_id:
        device_section["id"] = machine.source_profile_id

    ids: list[tuple[int, int | None]] = []
    if isinstance(machine.usb_vid, int):
        ids.append((machine.usb_vid, machine.usb_pid))
    for pair in usb_ids or []:
        if pair not in ids:
            ids.append(pair)
    if ids:
        device_section["usb_ids"] = [
            format_usb_id(vid, pid) for vid, pid in ids
        ]

    device_data = {
        "api_version": CURRENT_API_VERSION,
        "device": device_section,
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
