from .dialect import GcodeDialect


GRBL_DIALECT = GcodeDialect(
    uid="grbl",
    label=_("GRBL (universal)"),
    description=_("Standard GRBL for most diode lasers and hobby CNCs."),
    laser_on="M4 S{power:.0f}",
    laser_off="M5",
    tool_change="T{tool_number}",
    set_speed="",
    travel_move="G0 X{x} Y{y} Z{z}",
    linear_move="G1 X{x} Y{y} Z{z}{f_command}",
    arc_cw="G2 X{x} Y{y} Z{z} I{i} J{j}{f_command}",
    arc_ccw="G3 X{x} Y{y} Z{z} I{i} J{j}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    preamble=["G21 ;Set units to mm", "G90 ;Absolute positioning"],
    postscript=[
        "M5 ;Ensure laser is off",
        "G0 X0 Y0 ;Return to origin",
    ],
)

GRBL_DIALECT_NOZ = GcodeDialect(
    uid="grbl_noz",
    label=_("GRBL (no Z axis)"),
    description=_(
        "Standard GRBL, but removes Z axis commands for more efficient G-code."
    ),
    laser_on="M4 S{power:.0f}",
    laser_off="M5",
    tool_change="T{tool_number}",
    set_speed="",
    travel_move="G0 X{x} Y{y}",
    linear_move="G1 X{x} Y{y}{f_command}",
    arc_cw="G2 X{x} Y{y} I{i} J{j}{f_command}",
    arc_ccw="G3 X{x} Y{y} I{i} J{j}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    preamble=["G21 ;Set units to mm", "G90 ;Absolute positioning"],
    postscript=[
        "M5 ;Ensure laser is off",
        "G0 X0 Y0 ;Return to origin",
    ],
)

SMOOTHIEWARE_DIALECT = GcodeDialect(
    uid="smoothieware",
    label=_("Smoothieware"),
    description=_("G-code dialect for Smoothieware-based controllers."),
    laser_on="M3 S{power:.0f}",
    laser_off="M5",
    tool_change="T{tool_number}",
    set_speed="",
    travel_move="G0 X{x} Y{y} Z{z}{f_command}",
    linear_move="G1 X{x} Y{y} Z{z}{f_command}",
    arc_cw="G2 X{x} Y{y} Z{z} I{i} J{j}{f_command}",
    arc_ccw="G3 X{x} Y{y} Z{z} I{i} J{j}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    preamble=["G21 ; Set units to mm", "G90 ; Absolute positioning"],
    postscript=[
        "M5 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
    ],
)

MARLIN_DIALECT = GcodeDialect(
    uid="marlin",
    label=_("Marlin"),
    description=_(
        "G-code for Marlin-based controllers, common in 3D printers."
    ),
    laser_on="M4 S{power:.0f}",
    laser_off="M5",
    tool_change="T{tool_number}",
    set_speed="",
    travel_move="G0 X{x} Y{y} Z{z}{f_command}",
    linear_move="G1 X{x} Y{y} Z{z}{f_command}",
    arc_cw="G2 X{x} Y{y} Z{z} I{i} J{j}{f_command}",
    arc_ccw="G3 X{x} Y{y} Z{z} I{i} J{j}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    preamble=["G21 ; Set units to mm", "G90 ; Absolute positioning"],
    postscript=[
        "M5 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
    ],
)

BUILTIN_DIALECTS = [
    GRBL_DIALECT,
    GRBL_DIALECT_NOZ,
    SMOOTHIEWARE_DIALECT,
    MARLIN_DIALECT,
]
