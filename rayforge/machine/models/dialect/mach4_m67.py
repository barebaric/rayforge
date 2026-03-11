from gettext import gettext as _
from .base import GcodeDialect

MACH4_M67_DIALECT = GcodeDialect(
    uid="mach4_m67",
    label=_("Mach4 (M67 Analog)"),
    description=_(
        "Mach4 with M67 analog output for high-speed raster engraving. "
        "Uses M67 E0 Q<0-255> for laser power instead of inline S commands, "
        "reducing buffer pressure on the controller."
    ),
    can_g0_with_speed=True,
    laser_on="M67 E0 Q{power:.0f}",
    laser_off="M67 E0 Q0",
    tool_change="M6 T{tool_number}",
    set_speed="",
    travel_move="G0 X{x} Y{y} Z{z}",
    linear_move="G1 X{x} Y{y} Z{z}{f_command}",
    arc_cw="G2 X{x} Y{y} Z{z} I{i} J{j}{f_command}",
    arc_ccw="G3 X{x} Y{y} Z{z} I{i} J{j}{f_command}",
    air_assist_on="M8",
    air_assist_off="M9",
    home_all="G28",
    home_axis="G28 {axis_letter}0",
    move_to="G0 X{x} Y{y}",
    jog="G91 G0 F{speed}",
    clear_alarm="M999",
    set_wcs_offset="G10 L2 P{p_num} X{x} Y{y} Z{z}",
    probe_cycle="G38.2 {axis_letter}{max_travel} F{feed_rate}",
    preamble=["G21 ; Set units to mm", "G90 ; Absolute positioning"],
    postscript=[
        "M67 E0 Q0 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
    ],
)
