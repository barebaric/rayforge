import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from gettext import gettext as _
from typing import cast

from raydriver.grbl.parser import parse_msg, parse_ver

from ....core.varset import Var, VarSet
from ..driver import DeviceState, Pos

logger = logging.getLogger(__name__)


# GRBL Next-gen command requests
@dataclass
class CommandRequest:
    """A request to send a command and await its full response."""

    command: str
    op_index: int | None = None
    response_lines: list[str] = field(default_factory=list)
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
fluidnc_command_url = "/?commandText={command}"
fluidnc_status_url = fluidnc_command_url.format(command="%3F")


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


def _recalculate_positions(
    machine_pos: Pos,
    work_pos: Pos,
    wco: Pos,
    mpos_found: bool,
    wpos_found: bool,
    wco_found: bool,
) -> tuple[Pos, Pos, Pos]:
    """
    Recalculate positions based on GRBL equations for consistency.
    Also infers WCO if missing but both MPos and WPos are present.

    Args:
        machine_pos: Current machine position
        work_pos: Current work position
        wco: Work coordinate offset
        mpos_found: Whether machine position was found in input
        wpos_found: Whether work position was found in input
        wco_found: Whether work coordinate offset was found in input

    Returns:
        Tuple of (recalculated_machine_pos, recalculated_work_pos,
        recalculated_wco)
    """
    n = max(len(machine_pos), len(work_pos), len(wco))

    def _pad(pos: Pos, default: float = 0.0) -> Pos:
        result = list(pos)
        while len(result) < n:
            result.append(default)
        return tuple(result)

    machine_pos = _pad(machine_pos)
    work_pos = _pad(work_pos)
    wco = _pad(wco)

    # 1. Infer WCO if explicitly missing but both MPos and WPos exist.
    # WCO = MPos - WPos
    if (
        mpos_found
        and wpos_found
        and not wco_found
        and all(v is not None for v in machine_pos)
        and all(v is not None for v in work_pos)
    ):
        m_float = cast(tuple[float, ...], machine_pos)
        w_float = cast(tuple[float, ...], work_pos)
        wco = tuple(m_float[i] - w_float[i] for i in range(n))

    # 2. Recalculate missing positions based on what we have.
    # If MPos is known, calculate WPos.
    if (
        mpos_found
        and all(v is not None for v in machine_pos)
        and all(v is not None for v in wco)
    ):
        m_float = cast(tuple[float, ...], machine_pos)
        wco_float = cast(tuple[float, ...], wco)
        return (
            machine_pos,
            tuple(m_float[i] - wco_float[i] for i in range(n)),
            wco,
        )

    # If WPos is known (and MPos isn't), calculate MPos.
    elif (
        wpos_found
        and all(v is not None for v in work_pos)
        and all(v is not None for v in wco)
    ):
        w_float = cast(tuple[float, ...], work_pos)
        wco_float = cast(tuple[float, ...], wco)
        return (
            tuple(w_float[i] + wco_float[i] for i in range(n)),
            work_pos,
            wco,
        )

    return machine_pos, work_pos, wco


def sync_state_wco(
    state: DeviceState,
    offsets: dict[str, Pos],
    active_wcs: str | None,
) -> None:
    """
    Refresh *state*'s work coordinate offset from authoritative
    ``$#`` data.

    GRBL includes WCO in status reports only when it changes, so
    right after a WCS write the cached WCO can be stale and
    ``machine_pos``, when derived from WPos + WCO, is wrong until
    the next report carries the new WCO. Zeroing again in that
    window would read a wrong machine position and write the old
    offset back. The ``$#`` read-back reports every slot, so the
    active slot's offset is authoritative and positions are
    recomputed from it immediately.
    """
    if not active_wcs or active_wcs not in offsets:
        return
    state.wco = offsets[active_wcs]
    state.machine_pos, state.work_pos, _ = _recalculate_positions(
        state.machine_pos,
        state.work_pos,
        state.wco,
        state.machine_pos[0] is not None,
        state.work_pos[0] is not None,
        True,
    )


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
