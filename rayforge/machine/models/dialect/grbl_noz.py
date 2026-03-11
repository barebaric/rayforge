from gettext import gettext as _
from .base import GcodeDialect

GRBL_DIALECT_NOZ = GcodeDialect(
    uid="grbl_noz",
    label=_("Grbl (Compat, no Z axis)"),
    description=_(
        "Grbl dialect with highest compatibility, "
        "but removes Z axis commands for more efficient G-code"
    ),
    can_g0_with_speed=False,
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
