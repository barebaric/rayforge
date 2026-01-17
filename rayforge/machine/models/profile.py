import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from .machine import Machine, Laser, Origin
from .macro import Macro, MacroTrigger
from ..driver import get_driver_cls
from .dialect import get_dialect, replace

if TYPE_CHECKING:
    from ...context import RayforgeContext


logger = logging.getLogger(__name__)


@dataclass
class DialectDefinition:
    """
    A type-safe container for dialect fields that can be overridden in a
    MachineProfile. All fields are optional.
    """

    label: Optional[str] = None
    description: Optional[str] = None
    laser_on: Optional[str] = None
    laser_off: Optional[str] = None
    tool_change: Optional[str] = None
    set_speed: Optional[str] = None
    travel_move: Optional[str] = None
    linear_move: Optional[str] = None
    arc_cw: Optional[str] = None
    arc_ccw: Optional[str] = None
    air_assist_on: Optional[str] = None
    air_assist_off: Optional[str] = None
    home_all: Optional[str] = None
    home_axis: Optional[str] = None
    move_to: Optional[str] = None
    jog: Optional[str] = None
    clear_alarm: Optional[str] = None
    set_wcs_offset: Optional[str] = None
    probe_cycle: Optional[str] = None
    preamble: Optional[List[str]] = None
    postscript: Optional[List[str]] = None


@dataclass
class MachineProfile:
    """
    A template for creating a new Machine instance with pre-configured
    settings. All fields are optional to allow for partial profiles.
    """

    name: str
    driver_class_name: Optional[str] = None
    dialect_uid: Optional[str] = None
    gcode_precision: Optional[int] = None
    supports_arcs: Optional[bool] = None
    dimensions: Optional[Tuple[int, int]] = None
    origin: Optional[Origin] = None
    max_travel_speed: Optional[int] = None
    max_cut_speed: Optional[int] = None
    driver_args: Optional[Dict[str, Any]] = None
    home_on_start: Optional[bool] = None
    heads: Optional[List[Dict[str, Any]]] = None
    hookmacros: Optional[List[Dict[str, Any]]] = None
    dialect_definition: Optional[DialectDefinition] = None

    def create_machine(self, context: "RayforgeContext") -> Machine:
        """
        Creates a Machine instance from this profile.

        Only attributes that are not None in the profile will be applied
        to the new Machine instance, allowing the Machine's own defaults to
        be used for any unspecified profile values.
        """
        m = Machine(context)
        m.name = self.name

        if self.driver_class_name:
            try:
                driver_cls = get_driver_cls(self.driver_class_name)
                m.set_driver(driver_cls, self.driver_args)
            except (ValueError, ImportError):
                # If driver class not found, we fall back to the default
                # NoDeviceDriver, which is a safe state.
                logger.error(
                    f"failed to create driver {self.driver_class_name}"
                    f" with args {self.driver_args}"
                )

        # Handle dialect creation. If the profile provides a dialect
        # definition, create and register a new custom dialect for this
        # machine.
        if self.dialect_definition:
            base_dialect_uid = self.dialect_uid or "grbl"
            try:
                base_dialect = get_dialect(base_dialect_uid)
            except ValueError:
                logger.warning(
                    f"Base dialect '{base_dialect_uid}' not found for "
                    f"profile '{self.name}'. Falling back to 'grbl'."
                )
                base_dialect = get_dialect("grbl")

            new_label = self.dialect_definition.label or _(
                "{label} (for {machine_name})"
            ).format(label=base_dialect.label, machine_name=self.name)

            # Create a new custom dialect based on the parent.
            new_dialect = base_dialect.copy_as_custom(new_label=new_label)

            # Get a dictionary of only the fields that were explicitly set
            # in the profile's dialect definition.
            overrides = {
                k: v
                for k, v in asdict(self.dialect_definition).items()
                if v is not None
            }

            # Apply the overrides to the new dialect instance.
            if overrides:
                new_dialect = replace(new_dialect, **overrides)

            # Add the new dialect to the manager (registers and saves it).
            context.dialect_mgr.add_dialect(new_dialect)
            m.dialect_uid = new_dialect.uid

        elif self.dialect_uid is not None:
            # No custom definition, just use the specified built-in or
            # existing dialect UID.
            m.dialect_uid = self.dialect_uid

        if self.gcode_precision is not None:
            m.gcode_precision = self.gcode_precision
        if self.supports_arcs is not None:
            m.supports_arcs = self.supports_arcs
        if self.dimensions is not None:
            m.dimensions = self.dimensions
        if self.origin is not None:
            m.origin = self.origin
        if self.max_travel_speed is not None:
            m.max_travel_speed = self.max_travel_speed
        if self.max_cut_speed is not None:
            m.max_cut_speed = self.max_cut_speed
        if self.home_on_start is not None:
            m.home_on_start = self.home_on_start
        if self.hookmacros is not None:
            for s_data in self.hookmacros:
                try:
                    # Profiles define hooks with an internal trigger field
                    trigger = MacroTrigger[s_data["trigger"]]
                    m.hookmacros[trigger] = Macro.from_dict(s_data)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid hook in profile: {e}")

        m.cameras = []

        if self.heads:
            # The machine is initialized with one head. We clear it before
            # applying profile-specific heads. This safely disconnects
            # signals.
            for head in m.heads[:]:
                m.remove_head(head)

            for head_profile in self.heads:
                # Create a laser head from the profile data. The dictionary
                # for each head should have a flat structure with keys that
                # Laser.from_dict can parse, such as "max_power",
                # "frame_power_percent", and "spot_size_mm".
                m.add_head(Laser.from_dict(head_profile))

        return m


PROFILES: List[MachineProfile] = [
    MachineProfile(
        name="Carvera Air",
        driver_class_name="SmoothieDriver",
        dialect_uid="smoothieware",
        gcode_precision=4,
        dimensions=(300, 200),
        origin=Origin.BOTTOM_LEFT,
        max_travel_speed=3000,
        max_cut_speed=3000,
        home_on_start=True,
        heads=[
            {
                "max_power": 1.0,
                "frame_power_percent": 1.0,
                "focus_power_percent": 1.0,
                "spot_size_mm": [0.1, 0.1],
            }
        ],
        dialect_definition=DialectDefinition(
            description=_("Custom dialect for the Carvera Air machine."),
            laser_on="",
            laser_off="G1 S0",
            travel_move="G0 X{x} Y{y} Z{z}",
            linear_move="G1 X{x} Y{y} Z{z}{s_command}{f_command}",
            arc_cw="G2 X{x} Y{y} Z{z} I{i} J{j}{s_command}{f_command}",
            arc_ccw="G3 X{x} Y{y} Z{z} I{i} J{j}{s_command}{f_command}",
            preamble=[
                "M321",
                "G0Z0",
                "G00 G54",
                "M3",
                "G21 ; Set units to mm",
                "G90 ; Absolute positioning",
            ],
            postscript=[
                "M5 ; Ensure laser is off",
                "G0 X0 Y0 ; Return to origin",
                ";USER END SCRIPT",
                "M322",
                ";USER END SCRIPT",
                "M2",
            ],
        ),
    ),
    MachineProfile(
        name="Sculpfun iCube",
        driver_class_name="GrblDriver",
        dialect_uid="grbl",
        gcode_precision=3,
        dimensions=(120, 120),
        origin=Origin.BOTTOM_LEFT,
        max_travel_speed=3000,
        max_cut_speed=1000,
        home_on_start=True,
        heads=[
            {
                "max_power": 1000,
                "frame_power_percent": 1.0,  # 1% power for framing
                "focus_power_percent": 1.0,  # 1% power for focusing
                "spot_size_mm": [0.1, 0.1],
            }
        ],
    ),
    MachineProfile(
        name="Sculpfun S30",
        driver_class_name="GrblSerialDriver",
        dialect_uid="grbl",
        gcode_precision=3,
        dimensions=(400, 400),
        origin=Origin.BOTTOM_LEFT,
        max_travel_speed=3000,
        max_cut_speed=1000,
        heads=[
            {
                "max_power": 1000,
                "frame_power_percent": 1.0,
                "focus_power_percent": 1.0,
                "spot_size_mm": [0.1, 0.1],
            }
        ],
    ),
    MachineProfile(
        name="xTool D1 Pro",
        driver_class_name="GrblNetworkDriver",
        dialect_uid="grbl",
        gcode_precision=3,
        dimensions=(430, 390),
        origin=Origin.BOTTOM_LEFT,
        max_travel_speed=3000,
        max_cut_speed=1000,
        home_on_start=True,
        driver_args={
            "host": "",
            "port": 8080,
            "ws_port": 8081,
        },
        heads=[
            {
                "max_power": 1000,
                "frame_power_percent": 1.0,
                "focus_power_percent": 1.0,
                "spot_size_mm": [0.05, 0.05],
            }
        ],
        dialect_definition=DialectDefinition(
            description=_("Custom preamble for the xTool D1 Pro machine."),
            preamble=[
                "G21 ;Set units to mm",
                "G90 ;Absolute positioning",
                "M5",
                "M17",
                "M106 S0",
            ],
        ),
    ),
    MachineProfile(
        name=_("Other Device"),
        driver_class_name="GrblDriver",
        dialect_uid="grbl",
        origin=Origin.BOTTOM_LEFT,
    ),
]
