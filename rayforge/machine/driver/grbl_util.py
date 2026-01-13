import re
import asyncio
from copy import copy, deepcopy
from typing import Callable, Optional, List, cast
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
pos_re = re.compile(r":(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)")
fs_re = re.compile(r"FS:(\d+),(\d+)")
bf_re = re.compile(r"Bf:(\d+),(\d+)")
grbl_setting_re = re.compile(r"\$(\d+)=([\d\.-]+)")
wcs_re = re.compile(r"\[(G5[4-9]):([\d\.-]+),([\d\.-]+),([\d\.-]+)\]")
prb_re = re.compile(r"\[PRB:([\d\.-]+),([\d\.-]+),([\d\.-]+):(\d)\]")
# Regex to find the active WCS (G54-G59) from a $G parser state report
grbl_parser_state_re = re.compile(r".*(G5[4-9]).*")


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
    except (ValueError, TypeError, KeyError):
        return DeviceError(
            -1,
            _("Unknown Error"),
            _("Invalid error code reported by machine."),
        )


def parse_grbl_parser_state(response_lines: List[str]) -> Optional[str]:
    """
    Parses the response from a '$G' command to find the active WCS.
    Example response: '[G54 G17 G21 G90 G94 M5 M9 T0 F0 S0]'
    """
    for line in response_lines:
        match = grbl_parser_state_re.match(line)
        if match:
            return match.group(1)  # Return the found G-code (e.g., "G54")
    return None


def _split_status_line(state_str: str) -> tuple[str, list[str]]:
    """
    Split status line into status part and attribute parts.

    Args:
        state_str: Status string like '<Idle|MPos:10,20,30>'

    Returns:
        Tuple of (status_part, list_of_attributes)
    """
    status_parts = state_str[1:-1].split("|")
    status = None
    attribs = []
    for part in status_parts:
        if not part:
            continue
        if not status:
            status = part
        else:
            attribs.append(part)
    return status or "", attribs


def _parse_status_part(status_part: str) -> tuple[DeviceStatus, Optional[str]]:
    """
    Parse status part into DeviceStatus and optional error code.

    Args:
        status_part: Status part like 'Idle' or 'Alarm:1'

    Returns:
        Tuple of (DeviceStatus, error_code or None)
    """
    status_parts = status_part.split(":")
    status_name = status_parts[0]
    error_code = status_parts[1] if len(status_parts) > 1 else None
    try:
        return DeviceStatus[status_name.upper()], error_code
    except KeyError:
        return DeviceStatus.UNKNOWN, error_code


def _parse_position_attribute(attrib: str, pos_type: str) -> Optional[Pos]:
    """
    Parse a position attribute (MPos, WPos, or WCO).

    Args:
        attrib: Attribute string like 'MPos:10.0,20.0,30.0'
        pos_type: Type of position ('MPos', 'WPos', or 'WCO')

    Returns:
        Position tuple or None if parsing fails
    """
    if not attrib.startswith(f"{pos_type}:"):
        return None
    return _parse_pos_triplet(attrib)


def _parse_feed_rate(attrib: str) -> Optional[int]:
    """
    Parse feed rate from FS attribute.

    Args:
        attrib: Attribute string like 'FS:1000,0'

    Returns:
        Feed rate value or None if parsing fails
    """
    if not attrib.startswith("FS:"):
        return None
    match = fs_re.match(attrib)
    if not match:
        return None
    try:
        fs = [int(i) for i in match.groups()]
        return int(fs[0])
    except (ValueError, IndexError):
        return None


def _parse_buffer_state(attrib: str) -> Optional[tuple[int, int]]:
    """
    Parse buffer state from Bf attribute.

    Args:
        attrib: Attribute string like 'Bf:62,0'

    Returns:
        Tuple of (available_buffer_bytes, available_rx_buffer_bytes)
        or None if parsing fails
    """
    if not attrib.startswith("Bf:"):
        return None
    match = bf_re.match(attrib)
    if not match:
        return None
    try:
        bf = [int(i) for i in match.groups()]
        return (int(bf[0]), int(bf[1]))
    except (ValueError, IndexError):
        return None


def _recalculate_positions(
    machine_pos: Pos,
    work_pos: Pos,
    wco: Pos,
    mpos_found: bool,
    wpos_found: bool,
) -> tuple[Pos, Pos]:
    """
    Recalculate positions based on GRBL equations for consistency.

    Args:
        machine_pos: Current machine position
        work_pos: Current work position
        wco: Work coordinate offset
        mpos_found: Whether machine position was found in input
        wpos_found: Whether work position was found in input

    Returns:
        Tuple of (recalculated_machine_pos, recalculated_work_pos)
    """
    if mpos_found and all(v is not None for v in machine_pos):
        if all(v is not None for v in wco):
            m_float = cast(tuple[float, float, float], machine_pos)
            wco_float = cast(tuple[float, float, float], wco)
            return machine_pos, (
                m_float[0] - wco_float[0],
                m_float[1] - wco_float[1],
                m_float[2] - wco_float[2],
            )
    elif wpos_found and all(v is not None for v in work_pos):
        if all(v is not None for v in wco):
            w_float = cast(tuple[float, float, float], work_pos)
            wco_float = cast(tuple[float, float, float], wco)
            return (
                w_float[0] + wco_float[0],
                w_float[1] + wco_float[1],
                w_float[2] + wco_float[2],
            ), work_pos
    return machine_pos, work_pos


def parse_state(
    state_str: str,
    default: DeviceState,
    logger: Optional[Callable] = None,
) -> DeviceState:
    """
    Parse GRBL status string into DeviceState.

    Args:
        state_str: Status string like '<Idle|MPos:10.0,20.0,30.0|WPos:0,0,0>'
        default: Default DeviceState to use as base
        logger: Optional logger function for debugging

    Returns:
        Parsed DeviceState
    """
    state = copy(default)
    try:
        status_part, attribs = _split_status_line(state_str)

        if status_part:
            status, error_code = _parse_status_part(status_part)
            state.status = status
            if logger:
                logger(message=f"Parsed status: {status.name}")
            if error_code is not None:
                state.error = error_code_to_device_error(error_code)
                if logger:
                    logger(message=f"Parsed error code: {error_code}")

        mpos_found = False
        wpos_found = False
        for attrib in attribs:
            if attrib.startswith("MPos:"):
                parsed = _parse_position_attribute(attrib, "MPos")
                if parsed:
                    state.machine_pos = parsed
                    mpos_found = parsed[0] is not None
            elif attrib.startswith("WPos:"):
                parsed = _parse_position_attribute(attrib, "WPos")
                if parsed:
                    state.work_pos = parsed
                    wpos_found = parsed[0] is not None
            elif attrib.startswith("WCO:"):
                parsed = _parse_position_attribute(attrib, "WCO")
                if parsed:
                    state.wco = parsed
            elif attrib.startswith("FS:"):
                feed_rate = _parse_feed_rate(attrib)
                if feed_rate is not None:
                    state.feed_rate = feed_rate
            elif attrib.startswith("Bf:"):
                buffer_state = _parse_buffer_state(attrib)
                if buffer_state:
                    (
                        state.buffer_available,
                        state.buffer_rx_available,
                    ) = buffer_state

        state.machine_pos, state.work_pos = _recalculate_positions(
            state.machine_pos,
            state.work_pos,
            state.wco,
            mpos_found,
            wpos_found,
        )

    except (ValueError, TypeError) as e:
        if logger:
            logger(
                message=f"Invalid status line format: {state_str}, error: {e}"
            )
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
