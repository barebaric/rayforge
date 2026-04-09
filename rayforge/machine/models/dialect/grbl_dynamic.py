from gettext import gettext as _
from .base import GcodeDialect

GRBL_DYNAMIC_DIALECT = GcodeDialect(
    uid="grbl_dynamic",
    label=_("GRBL Dynamic"),
    description=_(
        "GRBL with M4 dynamic power (Depth-Aware) mode. "
        "S parameter is included in motion commands"
    ),
    can_g0_with_speed=False,
    omit_unchanged_coords=True,
    laser_on="M4 S0",
    laser_off="M5",
    focus_laser_on="M4 S{power:.0f}",
    tool_change="T{tool_number}",
    set_speed="",
    travel_move="G0{x_cmd}{y_cmd}{z_cmd}",
    linear_move="G1{x_cmd}{y_cmd}{z_cmd}{f_command}{s_command}",
    arc_cw="G2{x_cmd}{y_cmd}{z_cmd} I{i} J{j}{f_command}{s_command}",
    arc_ccw="G3{x_cmd}{y_cmd}{z_cmd} I{i} J{j}{f_command}{s_command}",
    bezier_cubic="",  # not supported by Grbl
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
