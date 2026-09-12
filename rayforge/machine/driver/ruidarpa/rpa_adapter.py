"""
RPA Adapter — Main driver class for Ruida laser controllers via the
Ruida Protocol Analyzer (RPA) library.

Two modes:
- Direct mode: wraps ``RpaDirectDriver`` (in-process ``RdDriver``)
- TUI RPC mode: wraps ``RpcRdDriver`` (remote RPyC service, itself a
  GlueScript)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import replace
from functools import partial
from gettext import gettext as _
from operator import attrgetter
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
)

from rpalib.rpyc_client import RpcRdDriver
from ruidadriver.rd_status import RdStatusEvent

from rayforge.context import RayforgeContext
from rayforge.core.varset import BoolVar, FloatVar, HostnameVar, Var, VarSet
from rayforge.machine.driver.driver import (
    Axis,
    DeviceStatus,
    Driver,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
    PWMParams,
)
from rayforge.machine.driver.ruidarpa.rpa_direct_driver import RpaDirectDriver
from rayforge.machine.driver.ruidarpa.rpa_encoder import (
    DEFAULT_IMAGE_POWER_BIAS,
    DEFAULT_POWER_FLOOR,
)
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.transport import TransportStatus

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from rayforge.core.doc import Doc
    from rayforge.machine.models.head import Head
    from rayforge.machine.models.laser import Laser
    from rayforge.machine.models.machine import Machine
    from rayforge.pipeline.encoder.base import EncodedOutput, OpsEncoder

logger = logging.getLogger(__name__)

# Type alias for the two possible backends.
_RpaBackend: TypeAlias = RpaDirectDriver | RpcRdDriver

# Default speed for move_to() absolute jogs (mm/s), used until the first
# jog() records the GUI-configured speed. After that, move_to() reuses the
# last jog speed so absolute moves do not fight the jog speed.
# NOTE: 600 mm/s == DEFAULT_MAX_TRAVEL_SPEED_MMPM (36000 mm/min) below;
# keep them in sync to avoid drift.
DEFAULT_MOVE_TO_JOG_SPEED_MM_S = 600.0

# Driver-level default for the RPC sync request timeout (seconds); the
# RpcRdDriver class default is 5.0 for direct constructions.
DEFAULT_RPC_TIMEOUT_S = 30.0

# Ruida test-hardware speed limits (mm/min base units): 400 mm/s cut,
# 600 mm/s travel. Seeded into the machine only while it still holds the
# framework defaults (see _UNCONFIGURED_* below).
DEFAULT_MAX_CUT_SPEED_MMPM = 24000
DEFAULT_MAX_TRAVEL_SPEED_MMPM = 36000

# Framework defaults for the machine speed fields (machine.py:139-140).
# Instance attributes, not importable — mirrored here as the "user never
# configured" sentinel.
_UNCONFIGURED_MAX_CUT_SPEED_MMPM = 1000
_UNCONFIGURED_MAX_TRAVEL_SPEED_MMPM = 3000


def _unwrap_mm(value: object) -> float | None:
    """Extract the mm value from a POSITION_* field.

    StatusDict positions arrive as ``(float_mm, str_description)``
    tuples in both direct and TUI RPC modes. Plain numbers are
    accepted for forward compatibility.
    """
    inner = value[0] if isinstance(value, (list, tuple)) else value
    if isinstance(inner, (int, float)):
        return float(inner)
    return None


class RuidaRPAAdapter(Driver):
    """
    Main driver class for connecting to Ruida laser controllers via the
    Ruida Protocol Analyzer (RPA) library.

    Supports two connection modes:
    * **Direct mode** — wraps ``RdDriver`` from ``ruida-protocol-analyzer``
      in-process over USB or UDP.
    * **TUI RPC mode** — connects to a remote RPyC service running the
      RPA TUI adapter.
    """

    label = _("Ruida RPA")
    subtitle = _("Connect via Ruida Protocol Analyzer")
    supports_settings = False
    reports_granular_progress = False
    uses_gcode = False
    maturity = DriverMaturity.KNOWN_BUGGY
    supports_probing = False
    native_overscan = True

    # --- Reconnect constants ---
    CONNECTION_POLL_INTERVAL = 0.5
    RECONNECT_BASE_DELAY = 1.0
    RECONNECT_MAX_DELAY = 5.0
    RECONNECT_JITTER = 0.2  # ±20%

    def __init__(self, context: RayforgeContext, machine: Machine) -> None:
        super().__init__(context, machine)
        self._config: dict[str, Any] = {}
        self._tui_mode: bool = False
        self._rpc_timeout: float = DEFAULT_RPC_TIMEOUT_S
        self._magic: int | None = None
        self._backend: _RpaBackend | None = None
        self._listeners_registered: bool = False
        self._unreachable_warned: bool = False
        self._connection_task: asyncio.Task | None = None
        self._keep_running: bool = False
        self._is_connected: bool = False
        self._shutting_down: bool = False
        self._jog_speed_mm_s: float | None = None
        self._selected_wcs: str = "MACHINE"
        self._machine_paused: bool = False
        self._machine_job_running: bool = False

    # --- Properties ---

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Ruida Coordinates")

    @property
    def supported_wcs(self) -> list[str]:
        return ["MACHINE", "ANCHOR", "CURRENT", "SET_POINT"]

    @property
    def resource_uri(self) -> str | None:
        host = self._config.get("udp_host", "")
        usb = self._config.get("usb_device", "")
        if host:
            return f"ruidarpa://{host}"
        if usb:
            return f"ruidarpa://{usb}"
        return None

    # --- Protect ---

    async def get_protect(self) -> bool:
        """Return whether protect mode is enabled."""
        backend = self._backend
        if backend is None:
            return False
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: backend.protect_enabled
        )

    async def set_protect(self, enabled: bool) -> None:
        """Enable or disable protect mode."""
        if self._backend is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._backend.set_protect, enabled)

    # --- Classmethods ---

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        udp_host = kwargs.get("udp_host", "")
        usb_device = kwargs.get("usb_device", "")
        if not udp_host and not usb_device:
            raise DriverPrecheckError(
                _(
                    "At least one of 'Hostname' or 'USB device' "
                    "must be configured."
                )
            )

    @classmethod
    def get_setup_vars(cls) -> VarSet:
        return VarSet(
            vars=[
                HostnameVar(
                    key="udp_host",
                    label=_("Hostname"),
                    description=_(
                        "The IP address or hostname of the Ruida controller"
                    ),
                ),
                Var(
                    key="usb_device",
                    label=_("USB"),
                    var_type=str,
                    description=_(
                        "USB device path "
                        "(e.g., /dev/ttyUSB0, "
                        "0403:6001, "
                        "or COM3)"
                    ),
                ),
                Var(
                    key="magic_number",
                    label=_("Magic"),
                    var_type=str,
                    description=_(
                        "Controller magic number in hex "
                        "(e.g., 0x88). Leave empty for default."
                    ),
                    default=None,
                ),
                BoolVar(
                    key="tui",
                    label=_("TUI RPC"),
                    description=_(
                        "Enable TUI RPC connection to a remote RPA TUI service"
                    ),
                    default=False,
                ),
                FloatVar(
                    "timeout",
                    label=_("RPC timeout (s)"),
                    description=_(
                        "Maximum seconds to wait for each synchronous RPC. "
                        "Raise for large jobs with many layers."
                    ),
                    default=DEFAULT_RPC_TIMEOUT_S,
                    min_val=1.0,
                    digits=1,
                    visible_when=lambda v: v.get("tui", False),
                ),
                FloatVar(
                    key="power_floor",
                    label=_("VECTOR power floor"),
                    description=_(
                        "Minimum power percentage for VECTOR cut "
                        "compensation (e.g. 8 = 8%)."
                    ),
                    default=DEFAULT_POWER_FLOOR,
                    min_val=0.0,
                    max_val=100.0,
                    digits=3,
                ),
                FloatVar(
                    key="image_power_bias",
                    label=_("IMAGE power bias"),
                    description=_(
                        "Power bias added to IMAGE/raster "
                        "scan-line power (SET_POWER), % of max "
                        "(e.g. 8 = 8%)."
                    ),
                    default=DEFAULT_IMAGE_POWER_BIAS,
                    min_val=0.0,
                    max_val=100.0,
                    digits=1,
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: Machine) -> OpsEncoder:
        from rayforge.machine.driver.ruidarpa.rpa_encoder import (
            RuidaRPAEncoder,
        )

        return RuidaRPAEncoder()

    # --- Setup / Connect ---

    def _seed_machine_speed_defaults(self) -> None:
        """Seed machine speed limits with Ruida defaults when unconfigured.

        Machine Settings -> General is the source of truth for these
        values; the driver only fills in the Ruida hardware defaults
        while the machine still holds the framework defaults (i.e. the
        user has never configured them). set_max_* early-returns when
        unchanged and fires machine.changed for persistence.
        """
        if self._machine.max_cut_speed == _UNCONFIGURED_MAX_CUT_SPEED_MMPM:
            self._machine.set_max_cut_speed(DEFAULT_MAX_CUT_SPEED_MMPM)
            logger.debug(
                "Seeding machine max cut speed with Ruida default (%d mm/min)",
                DEFAULT_MAX_CUT_SPEED_MMPM,
                extra=self._log_extra("RPA"),
            )
        if (
            self._machine.max_travel_speed
            == _UNCONFIGURED_MAX_TRAVEL_SPEED_MMPM
        ):
            self._machine.set_max_travel_speed(DEFAULT_MAX_TRAVEL_SPEED_MMPM)
            logger.debug(
                "Seeding machine max travel speed with Ruida default "
                "(%d mm/min)",
                DEFAULT_MAX_TRAVEL_SPEED_MMPM,
                extra=self._log_extra("RPA"),
            )

    def _setup_implementation(self, **kwargs: Any) -> None:
        self._config = dict(kwargs)
        self._tui_mode = bool(kwargs.get("tui", False))

        try:
            timeout = float(kwargs.get("timeout", DEFAULT_RPC_TIMEOUT_S))
        except (TypeError, ValueError):
            raise DriverSetupError("RPC timeout must be a number") from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise DriverSetupError("RPC timeout must be positive")

        self._rpc_timeout = timeout

        raw_magic = kwargs.get("magic_number")
        if raw_magic is not None:
            magic_str = str(raw_magic).strip()
            if not magic_str:
                self._magic = None
            else:
                try:
                    magic = int(magic_str, 0)
                except (TypeError, ValueError):
                    raise DriverSetupError(
                        "Magic number must be a valid hex value "
                        "(e.g. 0x88 or 88)"
                    ) from None
                if not (0x00 <= magic <= 0xFF):
                    raise DriverSetupError(
                        "Magic number must be between 0x00 and 0xFF"
                    )
                self._magic = magic
        else:
            self._magic = None

        self._listeners_registered = False
        self._unreachable_warned = False
        self._seed_machine_speed_defaults()

        if self._tui_mode:
            # The RpcRdDriver self-connects in its constructor (opening
            # its own RPyC TCP connection), so it is constructed per
            # connection attempt inside _connection_loop rather than
            # here — constructing it during setup would fail if the
            # server is down.
            self._backend = None
            logger.debug(
                "RPA adapter configured for TUI RPC mode",
                extra=self._log_extra("TUI_RPC"),
            )
        else:
            self._backend = RpaDirectDriver()
            logger.debug(
                "RPA adapter configured for direct mode",
                extra=self._log_extra("RPA"),
            )

    async def _connect_implementation(self) -> None:
        if self._connection_task and not self._connection_task.done():
            logger.warning(
                "Connect called with active connection task",
                extra=self._log_extra("RPA"),
            )
            return

        self._keep_running = True
        self._connection_task = asyncio.create_task(
            self._connection_loop(),
            name="ruidarpa-connection-loop",
        )

    def _set_connected(self, connected: bool, log_message: str) -> None:
        """Record a connection-state transition and notify observers."""
        self._is_connected = connected
        if not connected:
            self._machine_paused = False
            self._machine_job_running = False
        self.state.status = (
            DeviceStatus.IDLE if connected else DeviceStatus.UNKNOWN
        )
        # connection_status_changed is sent BEFORE state_changed so the
        # connection notification cannot be lost if a state_changed receiver
        # raises.
        status = (
            TransportStatus.CONNECTED
            if connected
            else TransportStatus.DISCONNECTED
        )
        self.connection_status_changed.send(self, status=status, message="")
        self.state_changed.send(self, state=self.state)
        extra = self._log_extra("TUI_RPC" if self._tui_mode else "RPA")
        if connected:
            logger.info(log_message, extra=extra)
        else:
            logger.warning(log_message, extra=extra)

    async def _connection_loop(self) -> None:
        """Background reconnection loop with exponential backoff."""
        loop = asyncio.get_running_loop()
        delay = self.RECONNECT_BASE_DELAY
        label = "TUI_RPC" if self._tui_mode else "RPA"
        log_extra = self._log_extra(label)

        while self._keep_running:
            self.connection_status_changed.send(
                self, status=TransportStatus.CONNECTING, message=""
            )

            connected: bool = False
            try:
                if self._tui_mode:
                    # The RpcRdDriver is constructed lazily on the first
                    # attempt and reused across reconnect attempts, so
                    # listeners are registered at most once per instance
                    # rather than accumulating on the server.
                    backend = self._backend
                    if isinstance(backend, RpaDirectDriver):
                        raise DriverSetupError(
                            "TUI RPC mode requires the RPC backend"
                        )
                    if backend is None:
                        backend = RpcRdDriver(timeout=self._rpc_timeout)
                        self._backend = backend
                    udp_host = self._config.get("udp_host")
                    usb_device = self._config.get("usb_device")
                    started = await loop.run_in_executor(
                        None,
                        partial(
                            backend.start, udp_host, usb_device, self._magic
                        ),
                    )
                    connected = started
                    if connected:
                        if not self._listeners_registered:
                            await loop.run_in_executor(
                                None,
                                backend.register_status_listener,
                                self._on_rpa_status,
                            )
                            await loop.run_in_executor(
                                None,
                                backend.register_error_listener,
                                self._on_rpa_error,
                            )
                            await loop.run_in_executor(
                                None,
                                backend.register_reply_listener,
                                self._on_rpa_reply,
                            )
                            self._listeners_registered = True
                        # Neutralize the server driver's default head/tail
                        # composition: jobs are fully self-framed by the
                        # encoder, and run_job() prepends the driver's
                        # default head script otherwise.
                        await loop.run_in_executor(
                            None, backend.set_head_script, []
                        )
                        await loop.run_in_executor(
                            None, backend.set_tail_script, []
                        )
                else:
                    backend = self._backend
                    if not isinstance(backend, RpaDirectDriver):
                        raise DriverSetupError(
                            "Direct mode requires the direct backend"
                        )
                    driver = backend
                    udp_host = self._config.get("udp_host")
                    usb_device = self._config.get("usb_device")
                    connected = await loop.run_in_executor(
                        None,
                        partial(
                            driver.start, udp_host, usb_device, self._magic
                        ),
                    )
                    if connected:
                        # Direct mode retains callbacks across stop/start,
                        # so drop any stale registration before
                        # re-registering — repeated registers would
                        # double-fire every status event.
                        driver.unregister_status_listener(self._on_rpa_status)
                        driver.unregister_error_listener(self._on_rpa_error)
                        driver.unregister_reply_listener(self._on_rpa_reply)
                        driver.register_status_listener(self._on_rpa_status)
                        driver.register_error_listener(self._on_rpa_error)
                        driver.register_reply_listener(self._on_rpa_reply)
                        # Both modes now run via run_job(), which composes
                        # head + job + tail; clear the wrapped driver's
                        # default head/tail so self-framed encoder output
                        # is not corrupted.
                        await loop.run_in_executor(
                            None, driver.gluescript.set_head_script, []
                        )
                        await loop.run_in_executor(
                            None, driver.gluescript.set_tail_script, []
                        )

                if not connected:
                    raise ConnectionError(
                        "Failed to connect to Ruida controller"
                    )

                # start() succeeded (transport bound); the connection is
                # confirmed by the poll loop via is_connected.
                delay = self.RECONNECT_BASE_DELAY

                prev_alive = False
                while self._keep_running:
                    await asyncio.sleep(self.CONNECTION_POLL_INTERVAL)
                    assert backend is not None
                    # is_connected is a blocking RPC round trip (TUI) or a
                    # direct property read; evaluate it off the event loop
                    # thread so a hung-but-alive server cannot freeze the UI.
                    is_alive = await loop.run_in_executor(
                        None, attrgetter("is_connected"), backend
                    )
                    if is_alive and not prev_alive:
                        # False -> True edge: the machine came online (or was
                        # already on when polling started). If we were in an
                        # unreachable episode, the server is reachable again.
                        if self._unreachable_warned:
                            if self._tui_mode:
                                logger.info(
                                    "RPA server reachable",
                                    extra=log_extra,
                                )
                            self._unreachable_warned = False
                        self._set_connected(
                            True, "Connected to Ruida controller via RPA"
                        )
                    elif not is_alive and prev_alive:
                        # True -> False edge: the connection was lost.
                        self._set_connected(False, "RPA connection lost")
                    prev_alive = is_alive

            except asyncio.CancelledError:
                logger.debug("Connection loop cancelled", extra=log_extra)
                self._is_connected = False
                break
            except Exception as e:  # noqa: BLE001
                if not self._unreachable_warned:
                    self._unreachable_warned = True
                    if self._tui_mode:
                        if isinstance(e, ConnectionError):
                            logger.warning(
                                "RPA connection attempt failed: %s",
                                e,
                                extra=log_extra,
                            )
                        else:
                            logger.warning(
                                "RPA server unreachable: %s",
                                e,
                                extra=log_extra,
                            )
                    else:
                        logger.warning(
                            "RPA reconnect attempt failed: %s",
                            e,
                            extra=log_extra,
                        )
                self.connection_status_changed.send(
                    self, status=TransportStatus.ERROR, message=str(e)
                )
                self._is_connected = False
                self._machine_paused = False
                self._machine_job_running = False
                if self.state.status != DeviceStatus.UNKNOWN:
                    self.state.status = DeviceStatus.UNKNOWN
                    self.state_changed.send(self, state=self.state)
                await self._stop_backend()
                if self._tui_mode:
                    self._backend = None
                    self._listeners_registered = False

            # --- Reconnect delay with exponential backoff ---
            if self._keep_running:
                jitter = 1.0 + random.uniform(
                    -self.RECONNECT_JITTER, self.RECONNECT_JITTER
                )
                sleep_time = delay * jitter
                logger.debug(
                    "Reconnecting in %.1f seconds (base=%.1f, jitter=%.2f)",
                    sleep_time,
                    delay,
                    jitter,
                    extra=log_extra,
                )
                await asyncio.sleep(sleep_time)
                delay = min(delay * 2, self.RECONNECT_MAX_DELAY)

        logger.debug("Exiting RPA connection loop", extra=log_extra)

    # --- RPC / Direct driver callbacks ---

    def _on_rpa_status(self, event: Any) -> None:
        """Handle status events from the Ruida controller via RPC/direct mode.

        Called from the backend's background thread. Bridges to the adapter's
        state tracking.

        Args:
            event: A status string (e.g. 'CONNECTED', 'DISCONNECTED'), an
                RdStatusEvent enum member (direct mode), or a StatusDict dict
                for machine status updates.
        """
        if self._shutting_down:
            return
        # RdStatusEvent enum → string value (direct mode)
        if isinstance(event, RdStatusEvent):
            event = event.value
        if isinstance(event, str):
            if event == "CONNECTED" and not self._is_connected:
                self._set_connected(True, "RPA connected")
            elif (
                event in ("DISCONNECTED", "TERMINATED") and self._is_connected
            ):
                self._set_connected(False, "RPA disconnected")
        elif isinstance(event, dict):
            # StatusDict or RPyC netref — convert to local dict for reliable
            # type handling
            event = {k: event[k] for k in event}
            new_status = self._map_machine_status_to_device_status(event)
            if new_status != self.state.status:
                self.state = replace(self.state, status=new_status)
                self.state_changed.send(self, state=self.state)

            # Extract current position (values in mm)
            # POSITION_* values are (float_mm, str_description)
            pos_x = _unwrap_mm(event.get("POSITION_X"))
            pos_y = _unwrap_mm(event.get("POSITION_Y"))
            pos_z = _unwrap_mm(event.get("POSITION_Z"))

            if any(v is not None for v in (pos_x, pos_y, pos_z)):
                current = self.state.machine_pos
                new_x = (current[0] or 0.0) if pos_x is None else pos_x
                new_y = (current[1] or 0.0) if pos_y is None else pos_y
                new_z = (current[2] or 0.0) if pos_z is None else pos_z
                new_pos = (new_x, new_y, new_z)

                if new_pos != current:
                    self.state = replace(self.state, machine_pos=new_pos)
                    logger.debug(
                        "RPA position update: x=%.3f y=%.3f z=%.3f",
                        new_x,
                        new_y,
                        new_z,
                        extra=self._log_extra(
                            "TUI_RPC" if self._tui_mode else "RPA"
                        ),
                    )
                    self.state_changed.send(self, state=self.state)

    def _on_rpa_error(self, msg: str) -> None:
        """Handle error events from the Ruida controller."""
        if self._shutting_down:
            return
        logger.warning(
            "RPA error: %s",
            msg,
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )

    def _map_machine_status_to_device_status(
        self,
        event: dict[str, Any],
    ) -> DeviceStatus:
        """Maintain machine-status flags from a (possibly partial) status dict.

        StatusDict events only carry fields that changed; an absent flag means
        the controller state did not change for that flag.  Maintain the
        decoded ``MACHINE_STATUS_PAUSED`` / ``MACHINE_STATUS_JOB_RUNNING``
        flags and derive the current DeviceStatus from them.
        """
        paused = event.get("MACHINE_STATUS_PAUSED")
        if isinstance(paused, bool):
            self._machine_paused = paused
        job_running = event.get("MACHINE_STATUS_JOB_RUNNING")
        if isinstance(job_running, bool):
            self._machine_job_running = job_running
        if self._machine_paused:
            return DeviceStatus.HOLD
        if self._machine_job_running:
            return DeviceStatus.RUN
        return DeviceStatus.IDLE

    def _on_rpa_reply(self, replies: list[str]) -> None:
        """Handle reply data from the Ruida controller."""
        if self._shutting_down:
            return
        logger.debug(
            "RPA reply: %d lines",
            len(replies),
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )

    async def _stop_backend(self) -> None:
        """Stop and release the current backend driver."""
        backend = self._backend
        if backend is None:
            return

        loop = asyncio.get_running_loop()
        logger.debug(
            "Stopping RPA backend (%s)",
            "TUI RPC" if self._tui_mode else "direct",
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )
        try:
            if isinstance(backend, RpaDirectDriver):
                if backend.is_connected:
                    try:
                        backend.unregister_status_listener(self._on_rpa_status)
                    except Exception:
                        logger.exception("Error unregistering status listener")
                    try:
                        backend.unregister_error_listener(self._on_rpa_error)
                    except Exception:
                        logger.exception("Error unregistering error listener")
                    try:
                        backend.unregister_reply_listener(self._on_rpa_reply)
                    except Exception:
                        logger.exception("Error unregistering reply listener")
                await loop.run_in_executor(None, backend.stop)
            else:
                # stop() and close() are blocking RPyC round trips;
                # evaluate them off the event loop thread so a
                # hung-but-alive server cannot freeze the UI. close()
                # is idempotent and swallows teardown errors, so it must
                # run even when stop() raised on a dead transport.
                try:
                    await loop.run_in_executor(None, backend.stop)
                finally:
                    await loop.run_in_executor(None, backend.close)
        except Exception:
            logger.exception("Error stopping RPA backend")

    # --- Script execution ---

    async def _run_script(
        self, script_lines: list[str], auto_checksum: bool = False
    ) -> None:
        """Run raw rpascript via the backend's ``run``.

        All scripts — jobs and runtime commands — are sent raw through
        ``backend.run()``. The encoder output is already self-framed
        (REF_POINT/SET_ABSOLUTE/START_JOB…END_JOB), so head/tail
        composition is never used.

        Args:
            script_lines: Rpascript command lines to execute.
            auto_checksum: Whether to auto-calculate checksums. Passed
                for full job scripts (the current ruida-pa runner ignores
                this flag and patches END_JOB unconditionally).
        """
        if not script_lines:
            return
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._backend.run, script_lines, auto_checksum
        )

    def _backend_gluescript(self) -> Any:
        """Return the live backend GlueScript the transcript is replayed into.

        Direct mode wraps ``RdDriver`` (a GlueScript) behind
        ``RpaDirectDriver``; RPC mode's ``RpcRdDriver`` IS a GlueScript.
        The adapter replays the encoded GlueScript transcript into this
        instance via ``stage_gluescript``, which compiles it to rpascript
        for ``run_job``.
        """
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        if isinstance(self._backend, RpaDirectDriver):
            return self._backend.gluescript
        return self._backend

    # --- Job control ---

    async def run(
        self,
        encoded: EncodedOutput,
        doc: Doc,
        ops: Ops,
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        op_map = encoded.op_map

        if on_command_done is not None:
            num_ops = op_map.op_count if op_map else 0
            for op_index in range(num_ops):
                result = on_command_done(op_index)
                if inspect.isawaitable(result):
                    await result

        backend = self._backend_gluescript()
        loop = asyncio.get_running_loop()

        # The encoded text IS the GlueScript transcript (the source);
        # replay it into the live backend, which compiles it to rpascript
        # via stage_gluescript. The ops param is unused for replay — it is
        # kept for the Driver interface.
        transcript = encoded.text.splitlines()
        if not encoded.text.strip():
            logger.debug(
                "No rpascript commands to execute",
                extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
            )
        else:
            try:
                await loop.run_in_executor(
                    None, backend.stage_gluescript, transcript
                )
            except Exception:
                # Tear down the backend so a stale partial job cannot run.
                # If the teardown itself fails (e.g. dead transport), log
                # it and preserve the original stage error.
                try:
                    await loop.run_in_executor(None, backend.new_gluescript)
                except Exception:
                    logger.exception(
                        "Failed to reset backend after stage error"
                    )
                raise
            logger.info(
                "Executing rpascript job via run_job",
                extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
            )
            await loop.run_in_executor(None, backend.run_job)

        self.job_finished.send(self)

    async def run_raw(self, machine_code: str) -> None:
        lines = [
            line.strip() for line in machine_code.splitlines() if line.strip()
        ]
        if lines:
            logger.info(
                "Executing %d raw rpascript lines",
                len(lines),
                extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
            )
            await self._run_script(lines, auto_checksum=True)
        self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        if hold:
            await loop.run_in_executor(None, self._backend.pause)
            self._machine_paused = True
        else:
            await loop.run_in_executor(None, self._backend.resume)
            self._machine_paused = False
        # Derive from maintained flags (empty dict = no flags changed).
        new_status = self._map_machine_status_to_device_status({})
        if new_status != self.state.status:
            self.state = replace(self.state, status=new_status)
            self.state_changed.send(self, state=self.state)

    async def cancel(self, emergency: bool = False) -> None:
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._backend.stop_job)

    async def clear_alarm(self) -> None:
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._backend.stop_job)

    # --- Movement ---

    async def home(self, axes: Axis | None = None) -> None:
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        # Live homing commands auto-send server-side (and the direct
        # backend auto-sends via the wrapped RdDriver); the returned
        # lines must not be run() again.
        if axes is None or (axes & (Axis.X | Axis.Y)):
            await loop.run_in_executor(None, self._backend.home)
        if axes is not None and (axes & Axis.Z):
            await loop.run_in_executor(None, self._backend.home_z)

    async def move_to(self, pos_x: float, pos_y: float) -> None:
        """Move to an absolute position in machine-frame mm.

        Coordinates are machine-frame (same frame as POSITION_* status
        reporting: +X left of home, +Y down from home) and are passed
        through unchanged to the backend jog_xy_to.
        """
        logger.info(
            "move_to x=%.3f y=%.3f",
            pos_x,
            pos_y,
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        speed_mm_s = (
            self._jog_speed_mm_s
            if self._jog_speed_mm_s is not None
            else DEFAULT_MOVE_TO_JOG_SPEED_MM_S
        )
        await loop.run_in_executor(
            None, self._backend.jog_set_xy_speed, speed_mm_s
        )
        # Live jog commands auto-send server-side (and the direct backend
        # auto-sends via the wrapped RdDriver); the returned lines must
        # not be run() again.
        await loop.run_in_executor(None, self._backend.jog_xy_to, pos_x, pos_y)

    async def select_tool(self, tool_number: int) -> None:
        pass

    async def jog(self, speed: int, **deltas: float) -> None:
        # TODO: Jog speed is in mm/min, but the RPA TUI service expects mm/s.
        # Convert for now.
        speed_mm_per_s = speed / 60.0
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        self._jog_speed_mm_s = speed_mm_per_s
        # The speed is recorded before the early return so speed-only
        # jogs update the stored move_to() speed without touching the
        # backend.
        if not deltas:
            return
        loop = asyncio.get_running_loop()
        # Re-assert the XY jog speed on every jog: move_to() overrides
        # the backend speed, so a stale value would otherwise persist
        # until the next move.
        await loop.run_in_executor(
            None, self._backend.jog_set_xy_speed, speed_mm_per_s
        )
        if "z" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_set_z_speed, speed_mm_per_s
            )
        if "u" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_set_u_speed, speed_mm_per_s
            )
        # Live jog commands auto-send server-side (and the direct backend
        # auto-sends via the wrapped RdDriver); the returned lines must
        # not be run() again.
        if "x" in deltas and "y" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_xy_rel, deltas["x"], deltas["y"]
            )
        else:
            if "x" in deltas:
                await loop.run_in_executor(
                    None, self._backend.jog_x_rel, deltas["x"]
                )
            if "y" in deltas:
                await loop.run_in_executor(
                    None, self._backend.jog_y_rel, deltas["y"]
                )
        if "z" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_z_rel, deltas["z"]
            )
        if "u" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_u_rel, deltas["u"]
            )

    # --- Power / Laser ---

    async def set_power(self, head: Laser, percent: float) -> None:
        logger.warning(
            _(
                "set_power is not supported outside of a job with a "
                "Ruida controller — the correct handling is to be "
                "determined; no command was sent."
            )
        )

    async def set_focus_power(self, head: Laser, percent: float) -> None:
        logger.warning(
            _(
                "set_focus_power is not supported outside of a job with "
                "a Ruida controller — the correct handling is to be "
                "determined; no command was sent."
            )
        )

    # --- WCS ---

    async def select_wcs(self, wcs: str) -> None:
        if wcs not in self.supported_wcs:
            raise ValueError(
                f"Unsupported WCS {wcs!r} — valid names: "
                f"{', '.join(self.supported_wcs)}"
            )
        self._selected_wcs = wcs

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        raise NotImplementedError(
            _(
                "set_wcs_offset is not supported: protect mode blocks "
                "SET_SETTING, and the feature was dropped by user "
                "decision. Use select_wcs to pick a reference point."
            )
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        offsets: dict[str, Pos] = {
            "MACHINE": (0.0, 0.0, 0.0),
            "ANCHOR": (0.0, 0.0, 0.0),
            "CURRENT": (0.0, 0.0, 0.0),
            "SET_POINT": (0.0, 0.0, 0.0),
        }
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> str | None:
        return self._selected_wcs

    # --- Settings ---

    async def read_settings(self) -> None:
        await asyncio.sleep(0)
        self.settings_read.send(self, settings=[])

    async def write_setting(self, key: str, value: Any) -> None:
        pass

    def get_setting_vars(self) -> list[VarSet]:
        return [VarSet(title=_("No settings"))]

    # --- Probing ---

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        self.probe_status_changed.send(
            self, message=_("Probe not supported by RPA driver")
        )
        return None

    # --- Capabilities ---

    def can_jog(self, axis: Axis | None = None) -> bool:
        return True

    def supports_pwm(self, head: Head) -> bool:
        return isinstance(head, LaserHead) and head.laser_type.supports_pwm

    def get_pwm_params(self, head: Head) -> PWMParams | None:
        if not isinstance(head, LaserHead) or not self.supports_pwm(head):
            return None
        return PWMParams(
            frequency=head.pwm_frequency,
            max_frequency=head.max_pwm_frequency,
            pulse_width=head.pulse_width,
            min_pulse_width=head.min_pulse_width,
            max_pulse_width=head.max_pulse_width,
        )

    # --- Cleanup ---

    async def cleanup(self):
        self._shutting_down = True
        self._keep_running = False
        self._is_connected = False

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None

        await self._stop_backend()
        self._backend = None
        self._listeners_registered = False
        self._unreachable_warned = False

        self.connection_status_changed.send(
            self, status=TransportStatus.DISCONNECTED, message=""
        )
        await super().cleanup()
