"""Conversions between Rayforge driver state types and the session
state types reported by raydriver sessions.

Both model the same machine state: the session types are owned by
the raydriver library, while the driver types are Rayforge's domain.
These functions are the boundary between a session and its Driver.
"""

from raydriver.grbl.types import DeviceError as SessionDeviceError
from raydriver.grbl.types import DeviceState as SessionDeviceState
from raydriver.grbl.types import DeviceStatus as SessionDeviceStatus

from .driver import DeviceError, DeviceState, DeviceStatus


def error_from_session_state(
    session_error: SessionDeviceError,
) -> DeviceError:
    """Convert a session DeviceError into a Rayforge one."""
    return DeviceError(
        session_error.code,
        session_error.title,
        session_error.description,
    )


def from_session_state(
    session_state: SessionDeviceState,
) -> DeviceState:
    """Convert a session DeviceState into a Rayforge one."""
    error = None
    session_error = session_state.error
    if session_error is not None:
        error = error_from_session_state(session_error)
    return DeviceState(
        status=DeviceStatus[session_state.status.name],
        error=error,
        machine_pos=tuple(session_state.machine_pos),
        work_pos=tuple(session_state.work_pos),
        wco=tuple(session_state.wco),
        feed_rate=session_state.feed_rate,
        spindle_speed=session_state.spindle_speed,
        buffer_available=session_state.buffer_available,
        buffer_rx_available=session_state.buffer_rx_available,
    )


def to_session_state(state: DeviceState) -> SessionDeviceState:
    """Convert a Rayforge DeviceState into a session one."""
    session_state = SessionDeviceState()
    session_state.status = getattr(SessionDeviceStatus, state.status.name)
    if state.error is not None:
        session_state.error = SessionDeviceError(
            state.error.code,
            state.error.title,
            state.error.description,
        )
    session_state.machine_pos = list(state.machine_pos)
    session_state.work_pos = list(state.work_pos)
    session_state.wco = list(state.wco)
    session_state.feed_rate = state.feed_rate
    session_state.spindle_speed = state.spindle_speed
    session_state.buffer_available = state.buffer_available
    session_state.buffer_rx_available = state.buffer_rx_available
    return session_state
