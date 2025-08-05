from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class GcodeDialect:
    """
    A container for G-code command templates and formatting logic for a
    specific hardware dialect (e.g., GRBL, Marlin, Smoothieware).
    """

    name: str
    description: str

    # Command Templates
    # These templates will be used with .format() to generate G-code lines.
    # Common variables: {power}, {speed}, {x}, {y}, {i}, {j}
    laser_on: str  # e.g., "M4 S{power}" or "M3 S{power}"
    laser_off: str  # e.g., "M5"
    travel_move: str  # e.g., "G0 X{x:.3f} Y{y:.3f}"
    travel_move_with_speed: str  # e.g., "G0 X{x:.3f} Y{y:.3f} F{speed}"
    linear_move: str  # e.g., "G1 X{x:.3f} Y{y:.3f}"
    linear_move_with_speed: str  # e.g., "G1 X{x:.3f} Y{y:.3f} F{speed}"
    arc_cw: str  # e.g., "G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}"
    arc_cw_with_speed: (
        str  # e.g., "G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f} F{speed}"
    )
    arc_ccw: str  # e.g., "G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}"
    arc_ccw_with_speed: (
        str  # e.g., "G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f} F{speed}"
    )

    # Air Assist Control
    air_assist_on: str  # e.g., "M8"
    air_assist_off: str  # e.g., "M9"

    # Preamble & Postscript
    # Default sequences for this dialect. The user can override them in the
    # Machine settings.
    default_preamble: List[str] = field(default_factory=list)
    default_postscript: List[str] = field(default_factory=list)

    def format_laser_power(self, power: float) -> int:
        """
        Formats laser power value. Default is to convert to integer.
        Some dialects might require different scaling or formatting.
        """
        return int(power)


# Concrete Dialect Implementations
GRBL_DIALECT = GcodeDialect(
    name="GRBL",
    description=_("Standard GRBL for most diode lasers and hobby CNCs."),
    laser_on="M4 S{power}",  # Dynamic power mode for better grayscale
    laser_off="M5",
    travel_move="G0 X{x:.3f} Y{y:.3f}",
    travel_move_with_speed="G0 X{x:.3f} Y{y:.3f} F{speed}",
    linear_move="G1 X{x:.3f} Y{y:.3f}",
    linear_move_with_speed="G1 X{x:.3f} Y{y:.3f} F{speed}",
    arc_cw="G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}",
    arc_cw_with_speed="G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f} F{speed}",
    arc_ccw="G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}",
    arc_ccw_with_speed="G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f} F{speed}",
    air_assist_on="M8",
    air_assist_off="M9",
    default_preamble=[
        "G21 ; Set units to mm",
        "G90 ; Absolute positioning",
    ],
    default_postscript=[
        "M5 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
    ],
)

SMOOTHIEWARE_DIALECT = GcodeDialect(
    name="Smoothieware",
    description=_("G-code dialect for Smoothieware-based controllers."),
    # Smoothieware is mostly GRBL-compatible for laser operations
    laser_on="M3 S{power}",  # Often uses M3
    laser_off="M5",
    travel_move="G0 X{x:.3f} Y{y:.3f}",
    travel_move_with_speed="G0 X{x:.3f} Y{y:.3f} F{speed}",
    linear_move="G1 X{x:.3f} Y{y:.3f}",
    linear_move_with_speed="G1 X{x:.3f} Y{y:.3f} F{speed}",
    arc_cw="G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}",
    arc_cw_with_speed="G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f} F{speed}",
    arc_ccw="G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}",
    arc_ccw_with_speed="G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f} F{speed}",
    air_assist_on="M8",
    air_assist_off="M9",
    default_preamble=[
        "G21 ; Set units to mm",
        "G90 ; Absolute positioning",
    ],
    default_postscript=[
        "M5 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
    ],
)

MARLIN_DIALECT = GcodeDialect(
    name="Marlin",
    description=_(
        "G-code for Marlin-based controllers, common in 3D printers."
    ),
    # Marlin uses M3/M4 for inline laser power control, similar to GRBL.
    # M4 is preferred for dynamic power adjustment with speed changes.
    laser_on="M4 S{power}",
    laser_off="M5",
    travel_move="G0 X{x:.3f} Y{y:.3f}",
    travel_move_with_speed="G0 X{x:.3f} Y{y:.3f} F{speed}",
    linear_move="G1 X{x:.3f} Y{y:.3f}",
    linear_move_with_speed="G1 X{x:.3f} Y{y:.3f} F{speed}",
    arc_cw="G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}",
    arc_cw_with_speed="G2 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f} F{speed}",
    arc_ccw="G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f}",
    arc_ccw_with_speed="G3 X{x:.3f} Y{y:.3f} I{i:.3f} J{j:.3f} F{speed}",
    air_assist_on="M8",
    air_assist_off="M9",
    default_preamble=[
        "G21 ; Set units to mm",
        "G90 ; Absolute positioning",
    ],
    default_postscript=[
        "M5 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
    ],
)


DIALECTS: Dict[str, GcodeDialect] = {
    "GRBL": GRBL_DIALECT,
    "Smoothieware": SMOOTHIEWARE_DIALECT,
    "Marlin": MARLIN_DIALECT,
}


def get_dialect(name: str) -> GcodeDialect:
    """
    Retrieves a GcodeDialect instance from the registry by name.
    """
    dialect = DIALECTS.get(name)
    if not dialect:
        raise ValueError(f"Unknown or unsupported G-code dialect: '{name}'")
    return dialect
