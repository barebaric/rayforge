import asyncio
import logging
import multiprocessing
import uuid
from enum import Enum
from gettext import gettext as _
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

from blinker import Signal
from raygeo.geo.types import Point3D, Rect
from raygeo.ops.axis import Axis

from ...camera.models.camera import Camera
from ...camera.v4l import migrate_camera_data
from ...context import RayforgeContext, get_context
from ...core.capability import MachineCapability
from ...core.layer import Layer
from ...core.model import Model
from ...shared.tasker import task_mgr
from ...shared.units.system import UnitSystem
from ..assembly import Assembly
from ..driver import get_driver_cls
from ..driver.driver import DeviceState, Pos, PWMParams, pwm_varset
from ..kinematics import HeadSpec, Kinematics, build_assembly
from ..models.axis import AxisConfig, AxisDirection, AxisSet, AxisType
from ..transport import TransportStatus
from .coordspace import MachineSpace
from .dialect import GcodeDialect
from .head import Head, head_from_dict
from .laser import Laser, LaserHead
from .machine_hours import MachineHours
from .macro import Macro, MacroTrigger
from .rotary_module import RotaryMode, RotaryModule
from .zone import Zone

if TYPE_CHECKING:
    from ...core.varset import VarSet
    from ..driver.driver import Driver
    from .controller import MachineController
from .coordinate_system import CoordinateSystem


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

        self.driver_name: str | None = None
        self.driver_args: dict[str, Any] = {}
        self.driver_config: dict[str, Any] = {}
        self.precheck_error: str | None = None

        self.auto_connect: bool = True
        self.home_on_start: bool = False
        self.clear_alarm_on_connect: bool = False
        self.single_axis_homing_enabled: bool = True
        self.dialect_uid: str | None = "grbl"
        self.dialect_migrated: bool = False
        self._hydrated_dialect: GcodeDialect | None = None
        self.gcode_precision: int = 3
        self.supports_arcs: bool = True
        self.supports_curves: bool = False
        self.arc_tolerance: float = 0.03
        self.unit_system: UnitSystem = UnitSystem.METRIC
        self.hookmacros: dict[MacroTrigger, Macro] = {}
        self.macros: dict[str, Macro] = {}
        self.heads: list[Head] = []
        self._explicit_capabilities: frozenset[MachineCapability] | None = None
        self.cameras: list[Camera] = []
        self.max_travel_speed: int = 3000  # in mm/min
        self.max_cut_speed: int = 1000  # in mm/min
        self.acceleration: int = 1000  # in mm/s²
        self.axes: AxisSet = AxisSet(
            [
                AxisConfig(
                    letter=Axis.X,
                    axis_type=AxisType.LINEAR,
                    extents=(0, 200),
                ),
                AxisConfig(
                    letter=Axis.Y,
                    axis_type=AxisType.LINEAR,
                    extents=(0, 200),
                ),
                AxisConfig(
                    letter=Axis.Z,
                    axis_type=AxisType.LINEAR,
                    extents=(-50, 50),
                ),
            ]
        )
        self._work_margins: Rect = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._soft_limits: Rect | None = None
        self.origin: Origin = Origin.BOTTOM_LEFT
        self.rotary_enabled_default: bool = False
        self.default_rotary_module_uid: str | None = None
        self.soft_limits_enabled: bool = True
        self.wcs_origin_is_workarea_origin: bool = False
        self._settings_lock = asyncio.Lock()

        # Work Coordinate System (WCS) State
        # We default to standard G-code names for convenience, but the logic
        # is agnostic. Any key in wcs_offsets is considered a mutable WCS.
        # Any key NOT in wcs_offsets is considered an immutable/absolute system
        # with (0,0,0) offset.
        self.active_wcs: str = "G54"
        self.coordinate_systems: dict[str, CoordinateSystem] = (
            CoordinateSystem.defaults()
        )

        self.machine_hours: MachineHours = MachineHours()
        self.machine_hours.changed.connect(self._on_machine_hours_changed)

        # Connect to dialect manager to detect dialect changes
        self.context.dialect_mgr.dialects_changed.connect(
            self._on_dialects_changed
        )

        self.add_head(LaserHead())

        self.rotary_modules: dict[str, RotaryModule] = {}
        self.nogo_zones: dict[str, Zone] = {}

        self._assembly: Assembly | None = None
        self._assembly_dirty: bool = True
        self._mounted_rotaries: list[RotaryModule] = []
        self._layer_configured: bool = False

    @property
    def controller(self) -> "MachineController":
        """
        Dynamically retrieves the controller for this machine from the
        MachineManager. This enables lazy instantiation.
        """
        return self.context.machine_mgr.get_controller(self.id)

    @property
    def has_controller(self) -> bool:
        """
        Returns whether a controller currently exists for this machine
        without lazily creating one. This is ``False`` once the machine has
        been removed and its controller torn down, allowing callers to skip
        controller access that would otherwise raise.
        """
        return self.context.machine_mgr.has_controller(self.id)

    @property
    def driver(self) -> "Driver":
        """Property to access the driver through the controller."""
        return self.controller.driver

    def supports_pwm(self, head: Head | None = None) -> bool:
        """Whether the machine's driver supports PWM for the given head."""
        if head is None:
            if not self.heads:
                return False
            head = self.heads[0]
        return bool(self.driver.supports_pwm(head))

    def get_pwm_params(self, head: Head | None = None) -> PWMParams | None:
        """
        Returns the driver-reported PWM parameters for the given head, or
        None when the driver reports no PWM support.
        """
        if head is None:
            if not self.heads:
                return None
            head = self.heads[0]
        return self.driver.get_pwm_params(head)

    def get_pwm_settings(self, head: Head | None = None) -> Optional["VarSet"]:
        """
        Returns the PWM settings VarSet for the given head, or None when
        the driver reports no PWM support.
        """
        params = self.get_pwm_params(head)
        if params is None:
            return None
        return pwm_varset(params)

    def get_capabilities(self) -> frozenset[MachineCapability]:
        """
        Returns the machine capabilities as the union of the explicitly
        declared capabilities, the capabilities inferred from the
        configured heads (e.g. a LaserHead implies LASER), and the
        capabilities inferred from the driver (e.g. PWM on Ruida CO2
        lasers).
        """
        caps: set = set(self._explicit_capabilities or ())
        for head in self.heads:
            if head.machine_capability:
                caps.add(head.machine_capability)
        if self.supports_pwm():
            caps.add(MachineCapability.PWM)
        if self.rotary_modules:
            caps.add(MachineCapability.ROTARY)
        return frozenset(caps)

    def set_explicit_capabilities(
        self, capabilities: frozenset[MachineCapability] | None
    ):
        """
        Sets the explicitly declared capabilities. ``None`` means
        "not set", so capabilities are inferred from the heads.
        """
        if self._explicit_capabilities != capabilities:
            self._explicit_capabilities = capabilities
            self.changed.send(self)

    def _connect_controller_signals(self, controller: "MachineController"):
        """
        Connects this machine's signal proxies to the controller's signals.
        This is now called by the MachineManager when the controller is
        created.
        """
        controller.connection_status_changed.connect(
            self.connection_status_changed.send
        )
        controller.state_changed.connect(self.state_changed.send)
        controller.job_finished.connect(self.job_finished.send)
        controller.command_status_changed.connect(
            self.command_status_changed.send
        )
        controller.wcs_updated.connect(self.wcs_updated.send)

    def set_device_state(self, state: DeviceState):
        self.device_state = state

    def set_connection_status(self, status: TransportStatus):
        self.connection_status = status

    def set_precheck_error(self, error: str | None):
        self.precheck_error = error

    def set_unit_system(self, unit_system: UnitSystem) -> None:
        """
        Updates the machine's unit system and emits ``changed`` when
        it actually changed. Set by the probe wizard during device
        setup or by the user from the machine settings.
        """
        if self.unit_system != unit_system:
            self.unit_system = unit_system
            self.changed.send(self)

    def update_wcs_offset(self, slot: str, offset: Point3D):
        cs = self.coordinate_systems.get(slot)
        if cs:
            cs.offset = offset

    def update_wcs_offsets_batch(self, offsets: dict[str, Point3D]) -> bool:
        new_systems = {
            name: CoordinateSystem(name=name, label="", offset=offset)
            for name, offset in offsets.items()
        }
        changed = False
        for name, cs in new_systems.items():
            old = self.coordinate_systems.get(name)
            if old is None or old.offset != cs.offset:
                changed = True
                break
        self.coordinate_systems = new_systems
        return changed

    def get_wcs_offset(self, name: str) -> Point3D:
        cs = self.coordinate_systems.get(name)
        return cs.offset if cs else (0.0, 0.0, 0.0)

    @property
    def supported_wcs(self) -> list[str]:
        """
        Returns the list of supported Work Coordinate Systems from the driver.
        """
        return sorted(self.coordinate_systems.keys())

    def get_wcs_list(self) -> list[CoordinateSystem]:
        """Returns a sorted list of CoordinateSystem objects."""
        return [self.coordinate_systems[k] for k in self.supported_wcs]

    def get_wcs_label(self, name: str) -> str:
        cs = self.coordinate_systems.get(name)
        return cs.label if cs else ""

    def set_wcs_label(self, name: str, label: str):
        cs = self.coordinate_systems.get(name)
        if not cs:
            return
        if cs.label == label:
            return
        cs.label = label
        self.changed.send(self)

    @property
    def kinematics(self) -> Kinematics:
        return Kinematics(self.assembly)

    @property
    def assembly(self) -> Assembly:
        if self._assembly_dirty or self._assembly is None:
            self._assembly = self._build_assembly()
            self._assembly_dirty = False
        return self._assembly

    def invalidate_assembly(self):
        self._assembly_dirty = True

    def configure_for_layer(self, layer: Optional["Layer"]) -> None:
        required_rotaries: list[RotaryModule] = []
        if layer and layer.rotary_enabled:
            module = self.get_rotary_module_for_layer(layer)
            if module:
                required_rotaries.append(module)
        if self._assembly_needs_rebuild(required_rotaries):
            self._mounted_rotaries = required_rotaries
            self._layer_configured = True
            self._assembly_dirty = True

    def _assembly_needs_rebuild(
        self,
        rotaries: list[RotaryModule],
    ) -> bool:
        if self._assembly_dirty:
            return True
        if len(rotaries) != len(self._mounted_rotaries):
            return True
        current_uids = {r.uid for r in self._mounted_rotaries}
        new_uids = {r.uid for r in rotaries}
        return current_uids != new_uids

    def get_head_specs(self) -> list[HeadSpec]:
        """Return the head specs used to build an assembly.

        Does not build or mutate anything.  Each spec is a ``(model,
        transform)`` pair with the focal distance folded into the
        transform's Z translation.
        """
        head_specs: list[HeadSpec] = []
        for h in self.heads:
            t = h.transform.copy()
            focal_distance = getattr(h, "focal_distance", 0.0)
            if focal_distance > 0:
                t[2, 3] += focal_distance
            model = (
                Model.from_path(Path(h.model_path)) if h.model_path else None
            )
            head_specs.append((model, t))
        return head_specs

    def build_assembly_for_rotary(
        self,
        rotary_modules: dict[str, RotaryModule] | None = None,
    ) -> Assembly:
        """Build a throwaway assembly for the given rotary modules.

        Unlike ``configure_for_layer`` + ``assembly``, this never mutates
        machine state.  When *rotary_modules* is None or empty, a flat
        assembly (no rotary) is built.
        """
        return build_assembly(
            axis_set=self.axes,
            head_specs=self.get_head_specs(),
            rotary_modules=rotary_modules or None,
        )

    def _build_assembly(self) -> Assembly:
        rotaries = self._mounted_rotaries
        if not rotaries and not self._layer_configured and self.rotary_modules:
            rotaries = list(self.rotary_modules.values())[:1]
        rotary_modules_for_build: dict[str, RotaryModule] = {}
        if rotaries:
            for r in rotaries:
                rotary_modules_for_build[r.uid] = r
        return self.build_assembly_for_rotary(rotary_modules_for_build or None)

    @property
    def machine_space_wcs(self) -> str:
        """
        Returns the identifier for the machine space coordinate system.
        Delegates to the controller's driver property.
        """
        return self.controller.machine_space_wcs

    @property
    def machine_space_wcs_display_name(self) -> str:
        """
        Returns the display name for the machine space coordinate system.
        Delegates to the controller's driver property.
        """
        return self.controller.machine_space_wcs_display_name

    async def connect(self):
        """Public method to connect the driver."""
        await self.controller.connect()

    async def disconnect(self):
        """Public method to disconnect the driver."""
        await self.controller.disconnect()

    async def shutdown(self):
        """
        Gracefully shuts down the machine's active driver and resources.
        """
        logger.info(f"Shutting down machine '{self.name}' (id:{self.id})")
        # We only shut down the controller if it exists to avoid creating
        # it during shutdown if it wasn't used.
        try:
            # Check for existence via manager without triggering creation
            # if possible, or just call the manager's shutdown for this ID.
            # But simpler here is to let the manager handle bulk shutdown.
            # If we are shutting down a specific machine instance:
            if self.id in self.context.machine_mgr.controllers:
                await self.controller.shutdown()
        except Exception as e:  # noqa: BLE001 - best-effort shutdown cleanup
            logger.warning(f"Error shutting down controller: {e}")

        self.context.dialect_mgr.dialects_changed.disconnect(
            self._on_dialects_changed
        )

    def _on_dialects_changed(self, sender=None, **kwargs):
        """
        Callback when dialects are updated.
        Sends machine's changed signal to trigger recalculation.
        """
        self._hydrated_dialect = None
        self.changed.send(self)

    def is_connected(self) -> bool:
        """
        Checks if the machine's driver is currently connected to the device.

        Returns:
            True if connected, False otherwise.
        """
        return self.connection_status == TransportStatus.CONNECTED

    async def select_tool(self, index: int):
        """Sends a command to the driver to select a tool."""
        await self.controller.select_tool(index)

    def set_name(self, name: str):
        self.name = str(name)
        self.changed.send(self)

    def set_driver(self, driver_cls: type["Driver"], args=None):
        new_driver_name = driver_cls.__name__
        new_args = args or {}
        if (
            self.driver_name == new_driver_name
            and self.driver_args == new_args
        ):
            return

        self.driver_name = new_driver_name
        self.driver_args = new_args
        self.changed.send(self)
        task_mgr.add_coroutine(
            self.controller.rebuild_driver,
            key=(self.id, "rebuild-driver"),
        )

    def set_driver_args(self, args=None):
        new_args = args or {}
        if self.driver_args == new_args:
            return

        self.driver_args = new_args
        self.changed.send(self)
        task_mgr.add_coroutine(
            self.controller.rebuild_driver,
            key=(self.id, "rebuild-driver"),
        )

    @property
    def dialect(self) -> Optional["GcodeDialect"]:
        """Get the current dialect instance for this machine."""
        if self._hydrated_dialect:
            return self._hydrated_dialect
        if self.dialect_uid is None:
            return None
        try:
            return self.context.dialect_mgr.get(self.dialect_uid)
        except ValueError:
            logger.warning(
                f"Dialect '{self.dialect_uid}' not found for machine "
                f"'{self.name}'. Falling back to 'grbl'."
            )
            self.dialect_uid = "grbl"
            return self.context.dialect_mgr.get("grbl")

    def hydrate(self):
        """
        Fetches the current dialect from the registry and stores it internally.
        This ensures that when serialized, the machine carries the full
        dialect definition.
        """
        if self.dialect_uid is None:
            return
        try:
            self._hydrated_dialect = self.context.dialect_mgr.get(
                self.dialect_uid
            )
        except ValueError:
            logger.warning(
                f"Dialect '{self.dialect_uid}' not found for machine "
                f"'{self.name}'. Falling back to 'grbl'."
            )
            self.dialect_uid = "grbl"
            self._hydrated_dialect = self.context.dialect_mgr.get("grbl")

    def set_dialect_uid(self, dialect_uid: str | None):
        if self.dialect_uid == dialect_uid:
            return
        self.dialect_uid = dialect_uid
        self._hydrated_dialect = None
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

    def set_supports_curves(self, supports: bool):
        if self.supports_curves == supports:
            return
        self.supports_curves = supports
        self.changed.send(self)

    def set_supports_arcs(self, supports: bool):
        if self.supports_arcs == supports:
            return
        self.supports_arcs = supports
        self.changed.send(self)

    def set_home_on_start(self, home_on_start: bool = True):
        if self.home_on_start == home_on_start:
            return
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
        if self.max_travel_speed == speed:
            return
        self.max_travel_speed = speed
        self.changed.send(self)

    def set_max_cut_speed(self, speed: int):
        if self.max_cut_speed == speed:
            return
        self.max_cut_speed = speed
        self.changed.send(self)

    def set_acceleration(self, acceleration: int):
        if self.acceleration == acceleration:
            return
        self.acceleration = acceleration
        self.changed.send(self)

    @property
    def axis_extents(self) -> tuple[float, float]:
        """The full range of machine axis movement (width, height)."""
        x_cfg = self.axes.get(Axis.X)
        y_cfg = self.axes.get(Axis.Y)
        return (
            x_cfg.extents[1] if x_cfg else 200.0,
            y_cfg.extents[1] if y_cfg else 200.0,
        )

    @property
    def reverse_x_axis(self) -> bool:
        cfg = self.axes.get(Axis.X)
        return cfg.direction == AxisDirection.REVERSED if cfg else False

    @property
    def reverse_y_axis(self) -> bool:
        cfg = self.axes.get(Axis.Y)
        return cfg.direction == AxisDirection.REVERSED if cfg else False

    @property
    def reverse_z_axis(self) -> bool:
        cfg = self.axes.get(Axis.Z)
        return cfg.direction == AxisDirection.REVERSED if cfg else False

    def _clamp_soft_limits(self):
        """Clamp soft limits to axis extents. Returns True if clamped."""
        if self._soft_limits is None:
            return False
        w, h = self.axis_extents
        x_min, y_min, x_max, y_max = self._soft_limits
        clamped = (
            max(0.0, min(x_min, w)),
            max(0.0, min(y_min, h)),
            max(0.0, min(x_max, w)),
            max(0.0, min(y_max, h)),
        )
        if clamped != self._soft_limits:
            self._soft_limits = clamped
            return True
        return False

    def set_axis_extents(self, width: float, height: float):
        if self.axis_extents == (width, height):
            return
        x_cfg = self.axes.get(Axis.X)
        y_cfg = self.axes.get(Axis.Y)
        if x_cfg:
            x_cfg.extents = (0, width)
        if y_cfg:
            y_cfg.extents = (0, height)
        ml, mt, mr, mb = self._work_margins
        ml = min(ml, width - 1)
        mr = min(mr, width - ml - 1)
        mt = min(mt, height - 1)
        mb = min(mb, height - mt - 1)
        self._work_margins = (ml, mt, mr, mb)
        self._clamp_soft_limits()
        self.changed.send(self)

    @property
    def work_margins(self) -> Rect:
        """
        The margins around the work area (left, top, right, bottom).
        These are positive distances from the axis extents edges.
        """
        return self._work_margins

    def set_work_margins(
        self, left: float, top: float, right: float, bottom: float
    ):
        new_margins = (left, top, right, bottom)
        if self._work_margins == new_margins:
            return
        self._work_margins = new_margins
        self._soft_limits = None
        self.changed.send(self)

    @property
    def work_area(self) -> Rect:
        """
        The usable work area within the axis extents (x, y, w, h).
        Computed from axis_extents and work_margins.
        """
        ml, mt, mr, mb = self._work_margins
        w, h = self.axis_extents
        return (ml, mt, max(1.0, w - ml - mr), max(1.0, h - mt - mb))

    @property
    def soft_limits(self) -> Rect | None:
        """
        Configurable safety bounds for jogging (x_min, y_min, x_max, y_max).
        None means use work_area bounds.
        """
        return self._soft_limits

    def set_soft_limits(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ):
        w, h = self.axis_extents
        clamped = (
            max(0.0, min(x_min, w)),
            max(0.0, min(y_min, h)),
            max(0.0, min(x_max, w)),
            max(0.0, min(y_max, h)),
        )
        if self._soft_limits == clamped:
            return
        self._soft_limits = clamped
        self.changed.send(self)

    def clear_soft_limits(self):
        if self._soft_limits is None:
            return
        self._soft_limits = None
        self.changed.send(self)

    def set_origin(self, origin: Origin):
        if self.origin == origin:
            return
        self.origin = origin
        self.changed.send(self)

    def set_reverse_x_axis(self, is_reversed: bool):
        """Sets if the X-axis coordinate display is inverted."""
        if self.reverse_x_axis == is_reversed:
            return
        cfg = self.axes.get(Axis.X)
        if cfg:
            cfg.direction = (
                AxisDirection.REVERSED if is_reversed else AxisDirection.NORMAL
            )
        self.changed.send(self)

    def set_reverse_y_axis(self, is_reversed: bool):
        """Sets if the Y-axis coordinate display is inverted."""
        if self.reverse_y_axis == is_reversed:
            return
        cfg = self.axes.get(Axis.Y)
        if cfg:
            cfg.direction = (
                AxisDirection.REVERSED if is_reversed else AxisDirection.NORMAL
            )
        self.changed.send(self)

    def set_reverse_z_axis(self, is_reversed: bool):
        """Sets if the Z-axis direction is reversed."""
        if self.reverse_z_axis == is_reversed:
            return
        cfg = self.axes.get(Axis.Z)
        if cfg:
            cfg.direction = (
                AxisDirection.REVERSED if is_reversed else AxisDirection.NORMAL
            )
        self.changed.send(self)

    def set_rotary_enabled_default(self, enabled: bool):
        if self.rotary_enabled_default == enabled:
            return
        self.rotary_enabled_default = enabled
        self.changed.send(self)

    def set_default_rotary_module_uid(self, uid: str | None):
        if self.default_rotary_module_uid == uid:
            return
        self.default_rotary_module_uid = uid
        self.changed.send(self)

    def set_wcs_origin_is_workarea_origin(self, value: bool):
        """Sets if the workarea origin should be treated as coordinate zero."""
        if self.wcs_origin_is_workarea_origin == value:
            return
        self.wcs_origin_is_workarea_origin = value
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

    def get_coordinate_space(self) -> "MachineSpace":
        """
        Get the machine's coordinate space configuration.

        Returns:
            A MachineSpace instance representing this machine's
            coordinate system configuration.
        """
        return MachineSpace.from_machine(self)

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
        if self.soft_limits_enabled == enabled:
            return
        self.soft_limits_enabled = enabled
        self.changed.send(self)

    def get_current_position(self) -> Pos:
        """Get the current work position of the machine."""
        return self.device_state.work_pos

    def get_soft_limits(self) -> Rect:
        """Get the soft limits as (x_min, y_min, x_max, y_max)."""
        if self._soft_limits is not None:
            x_min, y_min, x_max, y_max = self._soft_limits
            if self.reverse_x_axis:
                x_min, x_max = -x_max, -x_min
            if self.reverse_y_axis:
                y_min, y_max = -y_max, -y_min
            return (float(x_min), float(y_min), float(x_max), float(y_max))

        w, h = float(self.axis_extents[0]), float(self.axis_extents[1])

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
        x_pos, y_pos = current_pos[0], current_pos[1]
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
        x_pos, y_pos = current_pos[0], current_pos[1]
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

    @property
    def reports_granular_progress(self) -> bool:
        """Check if the machine's driver reports granular progress."""
        return self.controller.reports_granular_progress

    def can_home(self, axis: Axis | None = None) -> bool:
        """Check if the machine's driver supports homing for the given axis."""
        return self.controller.can_home(axis)

    async def home(self, axes=None):
        """Homes the specified axes or all axes if none specified."""
        await self.controller.home(axes)

    async def jog(self, deltas: dict[Axis, float], speed: int):
        """
        Jogs the machine along specified axes.

        Args:
            deltas: Dictionary mapping Axis enum members to distances in mm.
            speed: Speed in mm/min.
        """
        await self.controller.jog(deltas, speed)

    async def run_raw(self, gcode: str):
        """Executes a raw G-code string on the machine."""
        await self.controller.run_raw(gcode)

    def can_jog(self, axis: Axis | None = None) -> bool:
        """Check if machine's supports jogging for the given axis."""
        return self.controller.can_jog(axis)

    def add_head(self, head: Head):
        self.heads.append(head)
        head.changed.connect(self._on_head_changed)
        self.invalidate_assembly()
        self.changed.send(self)

    def get_head_by_uid(self, uid: str) -> Head | None:
        for head in self.heads:
            if head.uid == uid:
                return head
        return None

    def get_default_head(self) -> Head:
        """Returns the first head, or raises an error if none exist."""
        if not self.heads:
            raise ValueError("Machine has no heads configured.")
        return self.heads[0]

    def get_default_laser_head(self) -> LaserHead | None:
        """Returns the first laser head, or None if none exist."""
        for head in self.heads:
            if isinstance(head, LaserHead):
                return head
        return None

    def remove_head(self, head: Head):
        head.changed.disconnect(self._on_head_changed)
        self.heads.remove(head)
        self.invalidate_assembly()
        self.changed.send(self)

    def _on_head_changed(self, head, *args):
        self.invalidate_assembly()
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

    def add_rotary_module(self, module: RotaryModule):
        self.rotary_modules[module.uid] = module
        module.changed.connect(self._on_rotary_module_changed)
        self._sync_rotary_axis_config(module)
        self.invalidate_assembly()
        self.changed.send(self)

    def get_rotary_module_by_uid(self, uid: str) -> RotaryModule | None:
        return self.rotary_modules.get(uid)

    def get_default_rotary_module(self) -> RotaryModule | None:
        if self.default_rotary_module_uid:
            return self.get_rotary_module_by_uid(
                self.default_rotary_module_uid
            )
        return None

    def get_rotary_module_for_layer(
        self, layer: "Layer"
    ) -> RotaryModule | None:
        """Resolve the effective rotary module for *layer*.

        Returns the module referenced by
        :attr:`layer.rotary_module_uid` when it exists on this
        machine.  When the layer has rotary enabled but its module
        UID is missing or invalid, falls back to the machine's
        default module (or the first available module when no
        default is set) so that rotary mapping is still applied.
        """
        if not layer.rotary_enabled:
            return None
        if layer.rotary_module_uid:
            module = self.rotary_modules.get(layer.rotary_module_uid)
            if module is not None:
                return module
        default = self.get_default_rotary_module()
        if default is not None:
            return default
        if self.rotary_modules:
            return next(iter(self.rotary_modules.values()))
        return None

    def get_rotary_axis_for_layer(self, layer: "Layer") -> Axis | None:
        if not layer.rotary_enabled:
            return None
        module = self.get_rotary_module_for_layer(layer)
        return module.axis if module else None

    def remove_rotary_module(self, module: RotaryModule):
        module.changed.disconnect(self._on_rotary_module_changed)
        del self.rotary_modules[module.uid]
        if self._manages_axis_config(module):
            self.axes.remove_config(module.axis)
        if self.default_rotary_module_uid == module.uid:
            remaining = list(self.rotary_modules.keys())
            self.default_rotary_module_uid = (
                remaining[0] if remaining else None
            )
        self._mounted_rotaries = [
            r for r in self._mounted_rotaries if r.uid != module.uid
        ]
        self.invalidate_assembly()
        self.changed.send(self)

    def _on_rotary_module_changed(self, module, *args):
        self._sync_rotary_axis_config(module)
        self.invalidate_assembly()
        self.changed.send(self)

    @staticmethod
    def _manages_axis_config(module: RotaryModule) -> bool:
        return module.mode == RotaryMode.TRUE_4TH_AXIS and module.axis in {
            Axis.A,
            Axis.B,
            Axis.C,
            Axis.U,
        }

    def _sync_rotary_axis_config(self, module: RotaryModule) -> None:
        if not self._manages_axis_config(module):
            return
        existing = self.axes.get(module.axis)
        if existing is None:
            self.axes.add_config(
                AxisConfig(
                    letter=module.axis,
                    axis_type=AxisType.ROTARY,
                    extents=(0, 360),
                    rotary_diameter=module.default_diameter,
                )
            )
        elif existing.axis_type != AxisType.ROTARY:
            self.axes.remove_config(module.axis)
            self.axes.add_config(
                AxisConfig(
                    letter=module.axis,
                    axis_type=AxisType.ROTARY,
                    extents=(0, 360),
                    rotary_diameter=module.default_diameter,
                )
            )
        else:
            existing.rotary_diameter = module.default_diameter

    def add_nogo_zone(self, zone: Zone):
        self.nogo_zones[zone.uid] = zone
        zone.changed.connect(self._on_nogo_zone_changed)
        self.changed.send(self)

    def get_nogo_zone_by_uid(self, uid: str) -> Zone | None:
        return self.nogo_zones.get(uid)

    def remove_nogo_zone(self, zone: Zone):
        zone.changed.disconnect(self._on_nogo_zone_changed)
        del self.nogo_zones[zone.uid]
        self.changed.send(self)

    def _on_nogo_zone_changed(self, zone, *args):
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
        return any(
            h.frame_power_percent
            for h in self.heads
            if isinstance(h, LaserHead)
        )

    def can_focus(self):
        return any(
            h.focus_power_percent
            for h in self.heads
            if isinstance(h, LaserHead)
        )

    def validate_driver_setup(self) -> tuple[bool, str | None]:
        """
        Validates the machine's driver arguments against the driver's setup
        VarSet. Delegates to the controller.

        Returns:
            A tuple of (is_valid, error_message).
        """
        return self.controller.validate_driver_setup()

    async def set_power(
        self, head: Optional["Laser"] = None, percent: float = 0.0
    ) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control. If None, uses the default head.
            percent: Power percentage (0-1.0). 0 disables power.
        """
        await self.controller.set_power(head, percent)

    async def set_focus_power(
        self, head: Optional["Laser"] = None, percent: float = 0.0
    ) -> None:
        """
        Sets the laser power for focus mode.

        Args:
            head: The laser head to control. If None, uses the default head.
            percent: Power percentage (0-1.0). 0 disables power.
        """
        await self.controller.set_focus_power(head, percent)

    def get_active_wcs_offset(self) -> Point3D:
        """
        Returns the (x, y, z) offset for the currently active WCS.
        If the active_wcs is not in the known offsets dictionary, it assumes
        an absolute coordinate system with zero offset.
        """
        cs = self.coordinate_systems.get(self.active_wcs)
        return cs.offset if cs else (0.0, 0.0, 0.0)

    def get_workarea_origin_offset(self) -> tuple[float, float]:
        """
        Returns the position of the workarea origin in WORLD space.

        The workarea origin is at the corner specified by the machine's origin
        setting. This is used to convert from MACHINE coordinates to
        workarea-relative coordinates.

        Margins are: (left, top, right, bottom)

        Returns:
            Tuple of (x, y) in WORLD space.
        """
        ml, mt, mr, mb = self._work_margins
        width, height = self.axis_extents

        if self.origin == Origin.BOTTOM_LEFT:
            return (ml, mb)
        elif self.origin == Origin.TOP_LEFT:
            return (ml, height - mt)
        elif self.origin == Origin.BOTTOM_RIGHT:
            return (width - mr, mb)
        else:  # TOP_RIGHT
            return (width - mr, height - mt)

    def get_reference_offset(self) -> Point3D:
        """
        Returns the offset for converting from MACHINE to REFERENCE coords.

        REFERENCE coordinates are what the user sees in the UI. When
        wcs_origin_is_workarea_origin is False, this returns the active WCS
        offset. When True, this returns the workarea origin offset.

        Returns:
            Tuple of (x, y, z) offset in MACHINE space.
        """
        if self.wcs_origin_is_workarea_origin:
            x, y = self.get_workarea_origin_offset()
            return (x, y, 0.0)
        else:
            return self.get_active_wcs_offset()

    def get_reference_position_world(self) -> tuple[float, float]:
        """
        Returns the reference origin position in WORLD coordinates.

        This is used for positioning items in world space at the reference
        origin (either workarea origin or WCS origin, depending on settings).

        Returns:
            Tuple of (x, y) in WORLD space.
        """
        offset_x, offset_y, _ = self.get_reference_offset()
        machine_space = MachineSpace.from_machine(self)
        return machine_space.machine_point_to_world(offset_x, offset_y)

    def get_visual_extent_frame(self) -> Rect:
        """
        Returns the extent frame rectangle (x, y, width, height) in visual
        coordinates relative to the work area origin.

        The work area is at (0, 0) in its own coordinate system.
        The extent frame is positioned at (-margin_left, -margin_bottom)
        relative to the work area origin.

        Returns:
            Tuple of (x, y, width, height) where x,y is the frame position
            relative to the work area origin (0,0).
        """
        ml, mb = self._work_margins[0], self._work_margins[3]
        extent_w, extent_h = self.axis_extents
        return (float(-ml), float(-mb), float(extent_w), float(extent_h))

    def has_custom_work_area(self) -> bool:
        """
        Returns True if any margin is non-zero.
        """
        ml, mt, mr, mb = self._work_margins
        return ml != 0 or mt != 0 or mr != 0 or mb != 0

    def set_active_wcs(self, wcs: str):
        """
        Sets the active WCS on the model and notifies listeners.

        This updates the model state immediately. When called from the UI
        (e.g. WCS dropdown), use switch_active_wcs() on the controller
        instead, which also sends the G-code command to the device and
        confirms the switch.
        """
        if wcs != self.active_wcs:
            self.active_wcs = wcs
            self.changed.send(self)

    async def switch_active_wcs(self, wcs: str):
        """
        Switches the active WCS on both model and device.

        Sends the G-code WCS command, confirms via $G, and re-reads
        WCS offsets. Use this for UI-initiated WCS switches.
        """
        await self.controller.switch_active_wcs(wcs)

    async def set_work_origin(
        self, x: float, y: float, z: float, wcs_slot: str | None = None
    ):
        """
        Sets the work origin at the specified machine coordinates.

        Args:
            x: X-coordinate in machine space.
            y: Y-coordinate in machine space.
            z: Z-coordinate in machine space.
            wcs_slot: The WCS slot to update (e.g. "G54"). Defaults to active.
        """
        await self.controller.set_work_origin(x, y, z, wcs_slot)

    async def set_work_origin_here(
        self, axes: Axis, wcs_slot: str | None = None
    ):
        """
        Sets the work origin for the specified axes to the current machine
        position.

        Args:
            axes: Flag combination of axes to set (e.g. Axis.X | Axis.Y).
            wcs_slot: The WCS slot to update (e.g. "G54"). Defaults to active.
        """
        await self.controller.set_work_origin_here(axes, wcs_slot)

    async def sync_wcs_from_device(self):
        """Queries the device for current WCS offsets and updates state."""
        await self.controller.sync_wcs_from_device()

    async def sync_active_wcs_from_device(self):
        """Queries the device for its active WCS and updates state."""
        await self.controller.sync_active_wcs_from_device()

    def refresh_settings(self):
        """Public API for the UI to request a settings refresh."""
        task_mgr.add_coroutine(
            lambda ctx: self.controller.read_settings(),
            key=(self.id, "device-settings-read"),
        )

    def apply_setting(self, key: str, value: Any):
        """Public API for the UI to apply a single setting."""
        task_mgr.add_coroutine(
            lambda ctx: self.controller.write_setting(key, value),
            key=(
                self.id,
                "device-settings-write",
                key,
            ),  # Key includes setting key for uniqueness
        )

    def get_setting_vars(self) -> list["VarSet"]:
        """
        Gets the setting definitions from the machine's active driver
        as a VarSet.
        """
        return self.controller.get_setting_vars()

    def to_dict(self, include_frozen_dialect: bool = True) -> dict[str, Any]:
        data = {
            "machine": {
                "name": self.name,
                "driver": self.driver_name,
                "driver_args": self.driver_args,
                "driver_config": self.driver_config,
                "auto_connect": self.auto_connect,
                "clear_alarm_on_connect": self.clear_alarm_on_connect,
                "home_on_start": self.home_on_start,
                "single_axis_homing_enabled": self.single_axis_homing_enabled,
                "dialect_uid": self.dialect_uid,
                "active_wcs": self.active_wcs,
                "coordinate_systems": [
                    cs.to_dict() for cs in self.coordinate_systems.values()
                ],
                "supports_arcs": self.supports_arcs,
                "supports_curves": self.supports_curves,
                "arc_tolerance": self.arc_tolerance,
                "axes": self.axes.to_dict(),
                "axis_extents": list(self.axis_extents),
                "work_margins": list(self._work_margins),
                "soft_limits": list(self._soft_limits)
                if self._soft_limits
                else None,
                "origin": self.origin.value,
                "reverse_x_axis": self.reverse_x_axis,
                "reverse_y_axis": self.reverse_y_axis,
                "reverse_z_axis": self.reverse_z_axis,
                "rotary_enabled_default": self.rotary_enabled_default,
                "default_rotary_module_uid": (self.default_rotary_module_uid),
                "wcs_origin_is_workarea_origin": (
                    self.wcs_origin_is_workarea_origin
                ),
                "heads": [head.to_dict() for head in self.heads],
                "cameras": [camera.to_dict() for camera in self.cameras],
                "rotary_modules": [
                    rm.to_dict() for rm in self.rotary_modules.values()
                ],
                "nogo_zones": [z.to_dict() for z in self.nogo_zones.values()],
                "capabilities": (
                    [
                        c.value
                        for c in sorted(
                            self._explicit_capabilities, key=lambda c: c.value
                        )
                    ]
                    if self._explicit_capabilities
                    else None
                ),
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
                "units": {
                    "unit_system": self.unit_system.value,
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
        hook_data: dict[str, Any],
        current_dialect_uid: str | None,
        machine_name: str,
        context: RayforgeContext,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Checks for legacy JOB_START/JOB_END hooks and migrates them to a
        new custom dialect.

        Returns:
            A tuple containing the (potentially new) dialect UID and the
            cleaned hook_data dictionary.
        """
        if current_dialect_uid is None:
            return None, hook_data

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
            base_dialect = context.dialect_mgr.get(current_dialect_uid)
        except ValueError:
            logger.warning(
                f"Could not find base dialect '{current_dialect_uid}' for "
                f"migration. Using 'grbl' as a fallback."
            )
            base_dialect = context.dialect_mgr.get("grbl")

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

    @staticmethod
    def _parse_capabilities(
        raw: list[Any] | None,
    ) -> frozenset[MachineCapability] | None:
        """
        Parses a list of capability strings into a frozenset of
        MachineCapability. Returns None when the list is absent,
        and skips unknown values with a warning.
        """
        if raw is None:
            return None
        caps = set()
        for value in raw:
            try:
                caps.add(MachineCapability(value))
            except ValueError:
                logger.warning(f"Unknown machine capability '{value}'")
        return frozenset(caps)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        context: Optional["RayforgeContext"] = None,
    ) -> "Machine":
        if context is None:
            context = get_context()
        ma = cls(context)
        ma_data = data.get("machine", {})
        ma.id = ma_data.get("id", ma.id)
        ma.name = ma_data.get("name", ma.name)
        ma.driver_name = ma_data.get("driver")
        ma.driver_args = ma_data.get("driver_args", {})
        ma.driver_config = ma_data.get("driver_config", {})
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
        if dialect_uid is None:
            driver_cls = get_driver_cls(
                ma.driver_name if ma.driver_name else ""
            )
            if not driver_cls.uses_gcode:
                dialect_uid = None
            else:
                dialect_uid = ma_data.get("dialect", "grbl").lower()

        hook_data = ma_data.get("hookmacros", {})

        # Run the migration logic, which may update the dialect_uid and
        # hook_data
        dialect_uid, hook_data = cls._migrate_legacy_hooks_to_dialect(
            hook_data, dialect_uid, ma.name, context
        )

        dialect_uid, migrated = (
            context.dialect_mgr.migrate_builtin_dialect_to_copy(
                dialect_uid, ma.name
            )
        )
        ma.dialect_migrated = migrated
        ma.dialect_uid = dialect_uid
        ma.active_wcs = ma_data.get("active_wcs", ma.active_wcs)
        if "coordinate_systems" in ma_data:
            ma.coordinate_systems = {}
            for cs_data in ma_data["coordinate_systems"]:
                cs = CoordinateSystem.from_dict(cs_data)
                ma.coordinate_systems[cs.name] = cs
        elif "wcs_offsets" in ma_data:
            for name, offset in ma_data["wcs_offsets"].items():
                cs = ma.coordinate_systems.get(name)
                if cs:
                    cs.offset = tuple(offset)

        if "axes" in ma_data:
            ma.axes = AxisSet.from_dict(ma_data["axes"])
        else:
            legacy_extents = tuple(ma_data.get("dimensions", ma.axis_extents))
            if "axis_extents" in ma_data:
                legacy_extents = tuple(ma_data["axis_extents"])
            legacy_reverse_x = ma_data.get("reverse_x_axis", False)
            legacy_reverse_y = ma_data.get("reverse_y_axis", False)
            legacy_reverse_z = ma_data.get("reverse_z_axis", False)
            if "x_axis_negative" in ma_data:
                logger.info("Migrating legacy 'x_axis_negative' setting.")
                legacy_reverse_x = ma_data["x_axis_negative"]
            if "y_axis_negative" in ma_data:
                logger.info("Migrating legacy 'y_axis_negative' setting.")
                legacy_reverse_y = ma_data["y_axis_negative"]
            ma.axes = AxisSet.from_legacy(
                axis_extents=legacy_extents,
                reverse_x=legacy_reverse_x,
                reverse_y=legacy_reverse_y,
                reverse_z=legacy_reverse_z,
                rotary_modules=ma.rotary_modules,
            )

        if "work_margins" in ma_data:
            ma._work_margins = tuple(ma_data["work_margins"])
        elif "offsets" in ma_data:
            ox, oy = ma_data["offsets"]
            ma._work_margins = (ox, 0, 0, oy)

        if "soft_limits" in ma_data and ma_data["soft_limits"] is not None:
            ma._soft_limits = tuple(ma_data["soft_limits"])

        origin_value = ma_data.get("origin", None)
        if origin_value is not None:
            ma.origin = Origin(origin_value)
        else:  # Legacy support for y_axis_down
            ma.origin = (
                Origin.BOTTOM_LEFT
                if ma_data.get("y_axis_down", False) is False
                else Origin.TOP_LEFT
            )

        ma.rotary_enabled_default = ma_data.get(
            "rotary_enabled_default", False
        )
        ma.default_rotary_module_uid = ma_data.get("default_rotary_module_uid")

        ma.soft_limits_enabled = ma_data.get(
            "soft_limits_enabled", ma.soft_limits_enabled
        )

        ma.wcs_origin_is_workarea_origin = ma_data.get(
            "wcs_origin_is_workarea_origin", False
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

        macros_data = ma_data.get("macros", {})
        for uid, macro_data in macros_data.items():
            macro_data["uid"] = uid  # Ensure UID is consistent with key
            ma.macros[uid] = Macro.from_dict(macro_data)

        ma.heads = []
        for obj in ma_data.get("heads", {}):
            ma.add_head(head_from_dict(obj))
        ma._explicit_capabilities = cls._parse_capabilities(
            ma_data.get("capabilities")
        )
        ma.cameras = []
        for obj in ma_data.get("cameras", {}):
            ma.add_camera(Camera.from_dict(migrate_camera_data(obj)))
        for obj in ma_data.get("rotary_modules", []):
            ma.add_rotary_module(RotaryModule.from_dict(obj))
        for obj in ma_data.get("nogo_zones", []):
            ma.add_nogo_zone(Zone.from_dict(obj))
        speeds = ma_data.get("speeds", {})
        ma.max_cut_speed = speeds.get("max_cut_speed", ma.max_cut_speed)
        ma.max_travel_speed = speeds.get(
            "max_travel_speed", ma.max_travel_speed
        )
        ma.acceleration = speeds.get("acceleration", ma.acceleration)
        gcode = ma_data.get("gcode", {})
        ma.gcode_precision = gcode.get("gcode_precision", ma.gcode_precision)
        ma.supports_arcs = ma_data.get("supports_arcs", ma.supports_arcs)
        ma.supports_curves = ma_data.get("supports_curves", ma.supports_curves)
        ma.arc_tolerance = ma_data.get("arc_tolerance", ma.arc_tolerance)

        units = ma_data.get("units", {})
        unit_system_value = units.get("unit_system", "metric")
        try:
            ma.unit_system = UnitSystem(unit_system_value)
        except ValueError:
            logger.warning(
                f"Unknown unit_system '{unit_system_value}' in "
                f"machine config. Defaulting to metric."
            )
            ma.unit_system = UnitSystem.METRIC

        hours_data = ma_data.get("machine_hours", {})
        ma.machine_hours = MachineHours.from_dict(hours_data)
        ma.machine_hours.changed.connect(ma._on_machine_hours_changed)

        return ma
