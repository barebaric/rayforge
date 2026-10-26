"""
Serialization helpers for the remote protocol.

Converts driver-level data structures (device state, transport
status, VarSets, WCS offsets) to and from JSON-safe dictionaries so
they can cross the client/server boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.varset import VarSet
from ..driver.driver import DeviceError, DeviceState, DeviceStatus
from ..transport import TransportStatus

if TYPE_CHECKING:
    from ..driver.driver import Pos


def device_status_to_name(status: DeviceStatus) -> str:
    return status.name


def device_status_from_name(name: str) -> DeviceStatus:
    return DeviceStatus[name]


def transport_status_to_name(status: TransportStatus) -> str:
    return status.name


def transport_status_from_name(name: str) -> TransportStatus:
    return TransportStatus[name]


def _pos_to_list(pos: Any) -> list[Any] | None:
    if pos is None:
        return None
    return list(pos)


def _pos_from_list(data: Any) -> Any:
    if data is None:
        return None
    return tuple(data)


def device_state_to_dict(state: DeviceState) -> dict[str, Any]:
    """Serializes a DeviceState snapshot."""
    error = None
    if state.error is not None:
        error = {
            "code": state.error.code,
            "title": state.error.title,
            "description": state.error.description,
        }
    return {
        "status": state.status.name,
        "error": error,
        "machine_pos": _pos_to_list(state.machine_pos),
        "work_pos": _pos_to_list(state.work_pos),
        "wco": _pos_to_list(state.wco),
        "feed_rate": state.feed_rate,
        "spindle_speed": state.spindle_speed,
        "buffer_available": state.buffer_available,
        "buffer_rx_available": state.buffer_rx_available,
    }


def device_state_from_dict(data: dict[str, Any]) -> DeviceState:
    """Reconstructs a DeviceState from its serialized form."""
    error = None
    error_data = data.get("error")
    if error_data is not None:
        error = DeviceError(
            code=error_data["code"],
            title=error_data["title"],
            description=error_data["description"],
        )
    return DeviceState(
        status=DeviceStatus[data.get("status", "UNKNOWN")],
        error=error,
        machine_pos=_pos_from_list(data.get("machine_pos")),
        work_pos=_pos_from_list(data.get("work_pos")),
        wco=_pos_from_list(data.get("wco", (0.0, 0.0, 0.0))),
        feed_rate=data.get("feed_rate"),
        spindle_speed=data.get("spindle_speed"),
        buffer_available=data.get("buffer_available"),
        buffer_rx_available=data.get("buffer_rx_available"),
    )


def offsets_to_dict(offsets: dict[str, Any]) -> dict[str, list[Any]]:
    """Serializes a WCS offsets mapping."""
    return {slot: list(offset) for slot, offset in offsets.items()}


def offsets_from_dict(
    data: dict[str, list[Any]],
) -> dict[str, tuple[Any, ...]]:
    """Reconstructs a WCS offsets mapping."""
    return {slot: tuple(offset) for slot, offset in data.items()}


def var_sets_to_dicts(var_sets: list[VarSet]) -> list[dict[str, Any]]:
    """Serializes VarSets, including their current values."""
    return [var_set.to_dict(include_value=True) for var_set in var_sets]


def var_sets_from_dicts(data: list[dict[str, Any]]) -> list[VarSet]:
    """Reconstructs VarSets from their serialized form."""
    return [VarSet.from_dict(entry) for entry in data]


def pos_to_json(pos: Pos | None) -> list[Any] | None:
    """Serializes an optional position tuple."""
    return _pos_to_list(pos)


def pos_from_json(data: list[Any] | None) -> Pos | None:
    """Reconstructs an optional position tuple."""
    if data is None:
        return None
    return tuple(data)  # type: ignore[return-value]
