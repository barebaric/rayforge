from dataclasses import dataclass, field
from typing import List, Dict, Optional


_DIALECT_REGISTRY: Dict[str, "GcodeDialect"] = {}


def register_dialect(dialect: "GcodeDialect"):
    """Adds a dialect to the central registry, keyed by its unique `name`."""
    if dialect.name in _DIALECT_REGISTRY:
        raise ValueError(
            f"Dialect with name '{dialect.name}' is already registered."
        )
    _DIALECT_REGISTRY[dialect.name] = dialect


def get_dialect(name: str) -> "GcodeDialect":
    """Retrieves a GcodeDialect instance from the registry by its name."""
    dialect = _DIALECT_REGISTRY.get(name.lower())
    if not dialect:
        raise ValueError(
            f"Unknown or unsupported G-code dialect name: '{name}'"
        )
    return dialect


def get_available_dialects() -> List["GcodeDialect"]:
    """Returns a list of all registered GcodeDialect instances."""
    # Sort by display name for consistent UI presentation
    return sorted(_DIALECT_REGISTRY.values(), key=lambda d: d.label)


@dataclass
class GcodeDialect:
    """
    A container for G-code command templates and formatting logic for a
    specific hardware dialect (e.g., GRBL, Marlin, Smoothieware).
    """

    name: str  # Stable, programmatic identifier (e.g., "grbl", "marlin")
    label: str  # User-facing name for UI (e.g., "GRBL")
    description: str

    # Command Templates
    laser_on: str
    laser_off: str
    set_speed: str
    travel_move: str
    linear_move: str
    arc_cw: str
    arc_ccw: str

    # Air Assist Control
    air_assist_on: str
    air_assist_off: str

    # Preamble & Postscript
    default_preamble: List[str] = field(default_factory=list)
    default_postscript: List[str] = field(default_factory=list)

    def format_laser_power(self, power: float) -> int:
        """
        Formats laser power value. Default is to convert to integer.
        Some dialects might require different scaling or formatting.
        """
        return int(power)

    def format_feedrate(self, speed: Optional[float]) -> str:
        """
        Formats the feed rate (F-word) for a command. Returns an empty
        string if the speed is None, preventing invalid G-code.
        """
        return f" F{speed}" if speed is not None else ""

    def __post_init__(self):
        """Automatically register the dialect instance after it's created."""
        register_dialect(self)


GRBL_DIALECT = GcodeDialect(
    name="grbl",
    label=_("GRBL (universal)"),
    description=_("Standard GRBL for most diode lasers and hobby CNCs."),
    laser_on="M4 S{power}",
    laser_off="M5",
    set_speed="",
    travel_move="G0 X{x:.3f} Y{y:.3f} Z{z:.3f}{f_command}",
    linear_move="G1 X{x:.3f} Y{y:.3f} Z{z:.3f}{f_command}",
    arc_cw="G2 X{x:.3f} Y{y:.3f} Z{z:.3f} I{i:.3f} J{j:.3f}{f_command}",
    arc_ccw="G3 X{x:.3f} Y{y:.3f} Z{z:.3f} I{i:.3f} J{j:.3f}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    default_preamble=[
        "G21 ;Set units to mm",
        "G90 ;Absolute positioning"
    ],
    default_postscript=[
        "M5 ;Ensure laser is off",
        "G0 X0 Y0 ;Return to origin",
    ],
)

GRBL_DIALECT_NOZ = GcodeDialect(
    name="grbl_noz",
    label=_("GRBL (no Z axis)"),
    description=_(
        "Standard GRBL, but removes Z axis commands "
        "for more efficient G-code."
    ),
    laser_on="M4 S{power}",
    laser_off="M5",
    set_speed="",
    travel_move="G0 X{x:.3f} Y{y:.3f}{f_command}",
    linear_move="G1 X{x:.3f} Y{y:.3f}{f_command}",
    arc_cw="G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}{f_command}",
    arc_ccw="G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    default_preamble=[
        "G21 ;Set units to mm",
        "G90 ;Absolute positioning"],
    default_postscript=[
        "M5 ;Ensure laser is off",
        "G0 X0 Y0 ;Return to origin",
    ],
)

SMOOTHIEWARE_DIALECT = GcodeDialect(
    name="smoothieware",
    label=_("Smoothieware"),
    description=_("G-code dialect for Smoothieware-based controllers."),
    laser_on="M3 S{power}",
    laser_off="M5",
    set_speed="",
    travel_move="G0 X{x:.3f} Y{y:.3f} Z{z:.3f}{f_command}",
    linear_move="G1 X{x:.3f} Y{y:.3f} Z{z:.3f}{f_command}",
    arc_cw="G2 X{x:.3f} Y{y:.3f} Z{z:.3f} I{i:.3f} J{j:.3f}{f_command}",
    arc_ccw="G3 X{x:.3f} Y{y:.3f} Z{z:.3f} I{i:.3f} J{j:.3f}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    default_preamble=["G21 ; Set units to mm", "G90 ; Absolute positioning"],
    default_postscript=[
        "M5 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
    ],
)

MARLIN_DIALECT = GcodeDialect(
    name="marlin",
    label=_("Marlin"),
    description=_(
        "G-code for Marlin-based controllers, common in 3D printers."
    ),
    laser_on="M4 S{power}",
    laser_off="M5",
    set_speed="",
    travel_move="G0 X{x:.3f} Y{y:.3f} Z{z:.3f}{f_command}",
    linear_move="G1 X{x:.3f} Y{y:.3f} Z{z:.3f}{f_command}",
    arc_cw="G2 X{x:.3f} Y{y:.3f} Z{z:.3f} I{i:.3f} J{j:.3f}{f_command}",
    arc_ccw="G3 X{x:.3f} Y{y:.3f} Z{z:.3f} I{i:.3f} J{j:.3f}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    default_preamble=["G21 ; Set units to mm", "G90 ; Absolute positioning"],
    default_postscript=[
        "M5 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
    ],
)
