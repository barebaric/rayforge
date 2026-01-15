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
    home_all="$H",
    home_axis="$H{axis_letter}",
    move_to="$J=G90 G21 F{speed} X{x} Y{y}",
    jog="$J=G91 G21 F{speed}",
    clear_alarm="$X",
    set_wcs_offset="G10 L2 P{p_num} X{x} Y{y} Z{z}",
    probe_cycle="G38.2 {axis_letter}{max_travel} F{feed_rate}",
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
    home_all="$H",
    home_axis="$H{axis_letter}",
    move_to="$J=G90 G21 F{speed} X{x} Y{y}",
    jog="$J=G91 G21 F{speed}",
    clear_alarm="$X",
    set_wcs_offset="G10 L2 P{p_num} X{x} Y{y} Z{z}",
    probe_cycle="G38.2 {axis_letter}{max_travel} F{feed_rate}",
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
    home_all="$H",
    home_axis="G28 {axis_letter}0",
    move_to="G90 G0 X{x} Y{y}",
    jog="G91 G0 F{speed}",
    clear_alarm="M999",
    set_wcs_offset="G10 L20 P{p_num} X{x} Y{y} Z{z}",
    probe_cycle="G38.2 {axis_letter}{max_travel} F{feed_rate}",
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
    home_all="G28",
    home_axis="G28 {axis_letter}",
    move_to="G0 X{x} Y{y} F{speed}",
    jog="G91 G0 F{speed}",
    clear_alarm="M999",
    set_wcs_offset="G10 L2 P{p_num} X{x} Y{y} Z{z}",
    probe_cycle="G38.2 {axis_letter}{max_travel} F{feed_rate}",
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
