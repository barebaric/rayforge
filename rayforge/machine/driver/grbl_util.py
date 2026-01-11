import re
import asyncio
from copy import copy, deepcopy
from typing import Callable, Optional, List
from dataclasses import dataclass, field
from ...core.varset import Var, VarSet
from .driver import DeviceStatus, DeviceState, Pos, DeviceError


# GRBL Next-gen command requests
@dataclass
class CommandRequest:
    """A request to send a command and await its full response."""

    command: str
    op_index: Optional[int] = None
    response_lines: List[str] = field(default_factory=list)
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def payload(self) -> bytes:
        return (self.command + "\n").encode("utf-8")


# GRBL Network URLs
hw_info_url = "/command?plain=%5BESP420%5D&PAGEID="
fw_info_url = "/command?plain=%5BESP800%5D&PAGEID="
eeprom_info_url = "/command?plain=%5BESP400%5D&PAGEID="
command_url = "/command?commandText={command}&PAGEID="
upload_url = "/upload"
execute_url = "/command?commandText=%5BESP220%5D/{filename}"
status_url = command_url.format(command="?")


# GRBL Regex Parsers
pos_re = re.compile(r":(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)")
fs_re = re.compile(r"FS:(\d+),(\d+)")
grbl_setting_re = re.compile(r"\$(\d+)=([\d\.-]+)")
wcs_re = re.compile(r"\[(G5[4-9]):([\d\.-]+),([\d\.-]+),([\d\.-]+)\]")
prb_re = re.compile(r"\[PRB:([\d\.-]+),([\d\.-]+),([\d\.-]+):(\d)\]")


# GRBL Error Codes
GRBL_ERROR_CODES = {
    1: DeviceError(
        1,
        _("Expected Command Letter"),
        _(
            "G-code words consist of a letter and a value. Letter was not "
            "found."
        ),
    ),
    2: DeviceError(
        2,
        _("Bad Number Format"),
        _(
            "Missing the expected G-code word value or numeric value "
            "format is not valid."
        ),
    ),
    3: DeviceError(
        3,
        _("Invalid Statement"),
        _("Grbl '$' system command was not recognized or supported."),
    ),
    4: DeviceError(
        4,
        _("Value < 0"),
        _("Negative value received for an expected positive value."),
    ),
    5: DeviceError(
        5,
        _("Homing Disabled"),
        _("Homing cycle failure. Homing is not enabled via settings."),
    ),
    7: DeviceError(
        7,
        _("EEPROM Read Fail"),
        _(
            "An EEPROM read failed. Auto-restoring affected EEPROM to default "
            "values."
        ),
    ),
    8: DeviceError(
        8,
        _("Not Idle"),
        _(
            "Grbl '$' command cannot be used unless Grbl is IDLE. Ensures "
            "smooth operation during a job."
        ),
    ),
    9: DeviceError(
        9,
        _("G-Code Lock"),
        _("G-code commands are locked out during alarm or jog state."),
    ),
    10: DeviceError(
        10,
        _("Homing Not Enabled"),
        _("Soft limits cannot be enabled without homing also enabled."),
    ),
    11: DeviceError(
        11,
        _("Line Overflow"),
        _(
            "Max characters per line exceeded. File most likely formatted "
            "improperly."
        ),
    ),
    14: DeviceError(
        14,
        _("Line Length Exceeded"),
        _(
            "Build info or startup line exceeded EEPROM line length limit. "
            "Line not stored."
        ),
    ),
    15: DeviceError(
        15,
        _("Travel Exceeded"),
        _("Jog target exceeds machine travel. Jog command has been ignored."),
    ),
    17: DeviceError(
        17,
        _("Setting Disabled"),
        _("Laser mode requires PWM output."),
    ),
    20: DeviceError(
        20,
        _("Unsupported Command"),
        _(
            "Unsupported or invalid g-code command found in block. This "
            "usually means that you used the wrong Post-Processor to make "
            "your file, or that some incompatible code within needs to be "
            "manually deleted."
        ),
    ),
    21: DeviceError(
        21,
        _("Modal Group Violation"),
        _(
            "More than one g-code command from same modal group found in "
            "block."
        ),
    ),
    22: DeviceError(
        22,
        _("Undefined Feed Rate"),
        _("Feed rate has not yet been set or is undefined."),
    ),
    23: DeviceError(
        23,
        _("Motion Group Violation"),
        _(
            "More than one g-code command from motion group (G0, G1, G2, G3, "
            "G38.2, G80) found in block."
        ),
    ),
    24: DeviceError(
        24,
        _("Plane Selection Violation"),
        _(
            "More than one g-code command from plane selection group (G17, "
            "G18, G19) found in block."
        ),
    ),
    25: DeviceError(
        25,
        _("Distance Mode Violation"),
        _(
            "More than one g-code command from distance mode group (G90, G91) "
            "found in block."
        ),
    ),
    26: DeviceError(
        26,
        _("Feed Rate Mode Violation"),
        _(
            "More than one g-code command from feed rate mode group (G93, "
            "G94, G95) found in block."
        ),
    ),
    27: DeviceError(
        27,
        _("Units Violation"),
        _(
            "More than one g-code command from units group (G20, G21) found "
            "in block."
        ),
    ),
    28: DeviceError(
        28,
        _("Cutter Radius Compensation Violation"),
        _(
            "More than one g-code command from cutter radius compensation "
            "group (G40) found in block."
        ),
    ),
    29: DeviceError(
        29,
        _("Tool Length Offset Violation"),
        _(
            "More than one g-code command from tool length offset group (G43, "
            "G49) found in block."
        ),
    ),
    30: DeviceError(
        30,
        _("Cycle Retract Mode Violation"),
        _(
            "More than one g-code command from cycle retract mode group "
            "(G98, G99) found in block."
        ),
    ),
    31: DeviceError(
        31,
        _("Spindle Speed Mode Violation"),
        _(
            "More than one g-code command from spindle speed mode group "
            "(G96, G97) found in block."
        ),
    ),
    32: DeviceError(
        32,
        _("Coolant Mode Violation"),
        _(
            "More than one g-code command from coolant mode group "
            "(M7, M8, M9) found in block."
        ),
    ),
    33: DeviceError(
        33,
        _("Coordinate System Selection Violation"),
        _(
            "More than one g-code command from coordinate system selection "
            "group (G54-G59) found in block."
        ),
    ),
    34: DeviceError(
        34,
        _("Path Control Mode Violation"),
        _(
            "More than one g-code command from path control mode group "
            "(G61, G61.1, G64) found in block."
        ),
    ),
    35: DeviceError(
        35,
        _("Stop Mode Violation"),
        _(
            "More than one g-code command from stop mode group (M0, M1, M2, "
            "M30, M60) found in block."
        ),
    ),
    36: DeviceError(
        36,
        _("Spindle State Violation"),
        _(
            "More than one g-code command from spindle state group (M3, M4, "
            "M5) found in block."
        ),
    ),
    37: DeviceError(
        37,
        _("Tool Select Violation"),
        _(
            "More than one g-code command from tool select group (T) found in "
            "block."
        ),
    ),
    38: DeviceError(
        38,
        _("Non-Modal Actions Violation"),
        _(
            "More than one g-code command from non-modal actions group "
            "(G4, G10, G28, G30, G53, G92) found in block."
        ),
    ),
}


# GRBL WCS Helper
def gcode_to_p_number(wcs_slot: str) -> Optional[int]:
    """Converts a G-code WCS name (e.g., "G54") to its P-number."""
    try:
        # Check format, e.g. "G54"
        if not wcs_slot.startswith("G"):
            return None

        # G54 is P1, G55 is P2, etc.
        # Slice from index 1 to get the number "54", "55", etc.
        p_num = int(wcs_slot[1:]) - 53
        if 1 <= p_num <= 6:  # G54-G59
            return p_num
    except (ValueError, IndexError):
        pass
    return None


# GRBL State Parsers
def _parse_pos_triplet(pos: str) -> Optional[Pos]:
    match = pos_re.search(pos)
    if not match:
        return None
    pos_triplet = tuple(float(i) for i in match.groups())
    if len(pos_triplet) != 3:
        return None
    return pos_triplet


def error_code_to_device_error(error_code: str) -> DeviceError:
    try:
        code = int(error_code)
        return GRBL_ERROR_CODES[code]
    except (ValueError, TypeError):
        return DeviceError(
            -1,
            _("Unknown Error"),
            _("Invalid error code reported by machine."),
        )


def parse_state(
    state_str: str, default: DeviceState, logger: Callable
) -> DeviceState:
    state = copy(default)
    try:
        # Remove '<' and '>' and split by '|'
        status_parts = state_str[1:-1].split("|")
        status = None
        attribs = []
        for part in status_parts:
            if not part:
                continue
            if not status:  # First part is the status
                status = part.split(":")[0]
            else:
                attribs.append(part)

        if status:
            status_parts = status.split(":")
            status_name = status_parts[0]
            error_code = None
            if len(status_parts) > 1:
                error_code = status_parts[1]
            try:
                state.status = DeviceStatus[status_name.upper()]
                logger(message=f"Parsed status: {status_name}")
                if error_code is not None:
                    state.error = error_code_to_device_error(error_code)
                    logger(message=f"Parsed error code: {error_code}")
            except KeyError:
                logger(message=f"device sent an unsupported status: {status}")

        for attrib in attribs:
            if attrib.startswith("MPos:"):
                state.machine_pos = (
                    _parse_pos_triplet(attrib) or state.machine_pos
                )
            elif attrib.startswith("WPos:"):
                state.work_pos = _parse_pos_triplet(attrib) or state.work_pos
            elif attrib.startswith("FS:"):
                try:
                    match = fs_re.match(attrib)
                    if not match:
                        continue
                    fs = [int(i) for i in match.groups()]
                    state.feed_rate = int(fs[0])
                except (ValueError, IndexError):
                    logger(message=f"Invalid FS format: {attrib}")
    except ValueError as e:
        logger(message=f"Invalid status line format: {state_str}, error: {e}")
    return state


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
        var_type=bool,
        description="Invert step enable pin, boolean",
    ),
    Var(
        key="5",
        label="$5",
        var_type=bool,
        description="Invert limit pins, boolean",
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
]


def get_grbl_setting_varsets() -> List["VarSet"]:
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
