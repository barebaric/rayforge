import logging
from copy import deepcopy
from gettext import gettext as _

from raydriver.grbl.parser import parse_msg, parse_ver

from ....core.varset import Var, VarSet

logger = logging.getLogger(__name__)


def parse_board(line: str) -> str | None:
    """Parse grblHAL's ``[BOARD:...]`` build-info line."""
    if not line.startswith("[BOARD:"):
        return None
    board = line[7:].rstrip("]").strip()
    return board or None


def extract_device_name(build_info: list[str]) -> str:
    """
    Extract a human-readable device name from build info lines.

    Checks ``[MSG:machine:...]`` / ``[MSG:mechine:...]`` lines first,
    then grblHAL's ``[BOARD:...]`` line, and finally the VER line's
    build-info field (e.g. ``[VER:1.1h.ORTUR:]`` → ``"ORTUR"``).

    Raydriver's parser has no ``[BOARD:...]`` support yet, so this
    small wrapper stays Rayforge-side until it lands upstream.
    """
    for line in build_info:
        msg = parse_msg(line)
        if msg is not None:
            key, value = msg
            if key.lower() in ("machine", "mechine"):
                return value
    for line in build_info:
        board = parse_board(line)
        if board is not None:
            return board
    for line in build_info:
        ver = parse_ver(line)
        if ver is not None:
            _, build_name = ver
            if build_name:
                return build_name
    return "Unknown Grbl Device"


# GRBL Typed Settings Definitions
_STEPPER_CONFIG_VARS = [
    Var(
        key="0",
        label="$0",
        var_type=int,
        description="Step pulse time, microseconds",
    ),
    Var(
        key="1",
        label="$1",
        var_type=int,
        description="Step idle delay, milliseconds",
    ),
    Var(
        key="2",
        label="$2",
        var_type=int,
        description="Step pulse invert, mask",
    ),
    Var(
        key="3",
        label="$3",
        var_type=int,
        description="Step direction invert, mask",
    ),
    Var(
        key="4",
        label="$4",
        var_type=int,
        description="Invert step enable pins, mask",
    ),
    Var(
        key="5",
        label="$5",
        var_type=int,
        description="Invert limit pins, mask",
    ),
    Var(
        key="6",
        label="$6",
        var_type=bool,
        description="Invert probe pin, boolean",
    ),
]

_CONTROL_REPORTING_VARS = [
    Var(
        key="10",
        label="$10",
        var_type=int,
        description="Status report options, mask",
    ),
    Var(
        key="11",
        label="$11",
        var_type=float,
        description="Junction deviation, mm",
    ),
    Var(
        key="12", label="$12", var_type=float, description="Arc tolerance, mm"
    ),
    Var(
        key="13",
        label="$13",
        var_type=bool,
        description="Report in inches, boolean",
    ),
]

_LIMITS_HOMING_VARS = [
    Var(
        key="20",
        label="$20",
        var_type=bool,
        description="Soft limits enable, boolean",
    ),
    Var(
        key="21",
        label="$21",
        var_type=bool,
        description="Hard limits enable, boolean",
    ),
    Var(
        key="22",
        label="$22",
        var_type=bool,
        description="Homing cycle enable, boolean",
    ),
    Var(
        key="23",
        label="$23",
        var_type=int,
        description="Homing direction invert, mask",
    ),
    Var(
        key="24",
        label="$24",
        var_type=float,
        description="Homing locate feed rate, mm/min",
    ),
    Var(
        key="25",
        label="$25",
        var_type=float,
        description="Homing search seek rate, mm/min",
    ),
    Var(
        key="26",
        label="$26",
        var_type=int,
        description="Homing switch debounce delay, milliseconds",
    ),
    Var(
        key="27",
        label="$27",
        var_type=float,
        description="Homing switch pull-off distance, mm",
    ),
]

_SPINDLE_LASER_VARS = [
    Var(
        key="30",
        label="$30",
        var_type=float,
        description="Maximum spindle speed, RPM",
    ),
    Var(
        key="31",
        label="$31",
        var_type=float,
        description="Minimum spindle speed, RPM",
    ),
    Var(
        key="32",
        label="$32",
        var_type=bool,
        description="Laser-mode enable, boolean",
    ),
]

_AXIS_CALIBRATION_VARS = [
    Var(
        key="100",
        label="$100",
        var_type=float,
        description="X-axis travel resolution, step/mm",
    ),
    Var(
        key="101",
        label="$101",
        var_type=float,
        description="Y-axis travel resolution, step/mm",
    ),
    Var(
        key="102",
        label="$102",
        var_type=float,
        description="Z-axis travel resolution, step/mm",
    ),
    Var(
        key="103",
        label="$103",
        var_type=float,
        description="A-axis travel resolution, step/degree",
    ),
]

_AXIS_KINEMATICS_VARS = [
    Var(
        key="110",
        label="$110",
        var_type=float,
        description="X-axis maximum rate, mm/min",
    ),
    Var(
        key="111",
        label="$111",
        var_type=float,
        description="Y-axis maximum rate, mm/min",
    ),
    Var(
        key="112",
        label="$112",
        var_type=float,
        description="Z-axis maximum rate, mm/min",
    ),
    Var(
        key="113",
        label="$113",
        var_type=float,
        description="A-axis maximum rate, degrees/min",
    ),
    Var(
        key="120",
        label="$120",
        var_type=float,
        description="X-axis acceleration, mm/sec^2",
    ),
    Var(
        key="121",
        label="$121",
        var_type=float,
        description="Y-axis acceleration, mm/sec^2",
    ),
    Var(
        key="122",
        label="$122",
        var_type=float,
        description="Z-axis acceleration, mm/sec^2",
    ),
    Var(
        key="123",
        label="$123",
        var_type=float,
        description="A-axis acceleration, degrees/sec^2",
    ),
]

_AXIS_TRAVEL_VARS = [
    Var(
        key="130",
        label="$130",
        var_type=float,
        description="X-axis maximum travel, mm",
    ),
    Var(
        key="131",
        label="$131",
        var_type=float,
        description="Y-axis maximum travel, mm",
    ),
    Var(
        key="132",
        label="$132",
        var_type=float,
        description="Z-axis maximum travel, mm",
    ),
    Var(
        key="133",
        label="$133",
        var_type=float,
        description="A-axis maximum travel, degrees",
    ),
]


def get_grbl_setting_varsets() -> list["VarSet"]:
    """
    Returns a list of VarSet instances populated with the standard GRBL setting
    definitions, grouped into sensible categories.
    """
    # Assuming `_` is a globally available translation function
    return [
        VarSet(
            vars=deepcopy(_STEPPER_CONFIG_VARS),
            title=_("Stepper Configuration"),
            description=_(
                "Settings related to stepper motor timing and signal polarity."
            ),
        ),
        VarSet(
            vars=deepcopy(_CONTROL_REPORTING_VARS),
            title=_("Control & Reporting"),
            description=_(
                "Settings for GRBL's motion control and status reporting."
            ),
        ),
        VarSet(
            vars=deepcopy(_LIMITS_HOMING_VARS),
            title=_("Limits & Homing"),
            description=_(
                "Settings for soft/hard limits and the homing cycle."
            ),
        ),
        VarSet(
            vars=deepcopy(_SPINDLE_LASER_VARS),
            title=_("Spindle & Laser"),
            description=_(
                "Settings for controlling the spindle or laser module."
            ),
        ),
        VarSet(
            vars=deepcopy(_AXIS_CALIBRATION_VARS),
            title=_("Axis Calibration"),
            description=_("Defines the steps-per-millimeter for each axis."),
        ),
        VarSet(
            vars=deepcopy(_AXIS_KINEMATICS_VARS),
            title=_("Axis Kinematics"),
            description=_(
                "Defines the maximum rate and acceleration for each axis."
            ),
        ),
        VarSet(
            vars=deepcopy(_AXIS_TRAVEL_VARS),
            title=_("Axis Travel"),
            description=_(
                "Defines the maximum travel distance for each axis."
            ),
        ),
    ]


def apply_setting_to_varset(varset: "VarSet", key: str, value_str: str):
    """
    Assigns a raw device value to the Var with *key* in *varset*.

    If the value cannot be coerced to the Var's declared type (e.g. a
    GrblHAL bitmask reported for a setting that classic Grbl defines
    as a boolean), the Var is widened to a string type so the value
    is still displayed instead of aborting the whole settings read.
    """
    var = varset[key]
    try:
        var.value = value_str
    except (TypeError, ValueError) as e:
        logger.warning(
            "Setting $%s cannot be coerced to %s (%s); "
            "displaying raw value instead.",
            key,
            var.var_type.__name__,
            e,
        )
        var.var_type = str
        var.value = value_str
