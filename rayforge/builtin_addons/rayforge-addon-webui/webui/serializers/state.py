"""
Machine state serialization for the web UI.

Reports driver ``DeviceState`` data (status, positions, WCS offset)
plus the transport connection status of the machine.
"""

from typing import TYPE_CHECKING, Any

from ..coordinates import beam_panel

if TYPE_CHECKING:
    from rayforge.machine.models.machine import Machine


def _pos_list(pos) -> list[Any]:
    return [value if value is not None else None for value in pos]


def _error_dict(error) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "code": error.code,
        "title": error.title,
        "description": error.description,
    }


def machine_state(machine: "Machine") -> dict[str, Any]:
    """The machine's live state as a JSON-safe dict."""
    state = machine.device_state
    return {
        "status": state.status.name,
        "error": _error_dict(state.error),
        "machine_pos": _pos_list(state.machine_pos),
        "work_pos": _pos_list(state.work_pos),
        "wco": _pos_list(state.wco),
        "feed_rate": state.feed_rate,
        "spindle_speed": state.spindle_speed,
        "connection": machine.connection_status.name,
        "connected": machine.is_connected(),
        "active_wcs": machine.active_wcs,
        "machine_id": machine.id,
        "machine_name": machine.name,
        "beam_position_panel": beam_panel(machine),
    }
