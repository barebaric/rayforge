import re
import asyncio
from copy import copy, deepcopy
from typing import Callable, Optional, List
from dataclasses import dataclass, field
from ...core.varset import Var, VarSet
from .driver import DeviceStatus, DeviceState, Pos


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


# GRBL State Parsers
def _parse_pos_triplet(pos: str) -> Optional[Pos]:
    match = pos_re.search(pos)
    if not match:
        return None
    pos_triplet = tuple(float(i) for i in match.groups())
    if len(pos_triplet) != 3:
        return None
    return pos_triplet


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
            try:
                state.status = DeviceStatus[status.upper()]
                logger(message=f"Parsed status: {status}")
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
