"""
Driver that executes jobs on a remote rayforge-server.

The client machine is configured with this driver and the address of
a :mod:`rayforge.machine.remote.server` host plus the id of the
machine to drive there. Connecting fetches the server machine's
configuration so the local mirror can be kept identical (see
:mod:`rayforge.machine.remote.sync`), and jobs are submitted as
:class:`JobPayload`s: the heavy pipeline computation happens locally,
the timing-critical driver execution happens on the server, and the
laptop can disconnect at any time while the job keeps running.

Job execution (run, run_raw, cancel, feed hold) and absolute moves
are forwarded over the wire. Interactive device control that requires
a live protocol session (jogging, homing, probing, live power
control) is not available remotely yet.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

from ...core.varset import HostnameVar, PortVar, VarSet
from ..driver.driver import (
    DeviceConnectionError,
    DeviceStatus,
    Driver,
    DriverMaturity,
    DriverPrecheckError,
    Pos,
)
from ..job_status import JobKind
from ..transport import TransportStatus
from .payload import JobPayload
from .protocol import PROTOCOL_VERSION, read_message, write_message

if TYPE_CHECKING:
    from raygeo.ops import Ops
    from raygeo.ops.axis import Axis

    from ...context import RayforgeContext
    from ...core.doc import Doc
    from ...pipeline.encoder.base import EncodedOutput, OpsEncoder
    from ..models.laser import Laser
    from ..models.machine import Machine

logger = logging.getLogger(__name__)

#: Message types carrying asynchronous job events rather than
#: replies to an in-flight request.
_EVENT_TYPES = ("job_progress", "job_finished", "connection_lost")

#: Default server port, matching rayforge-server's default.
DEFAULT_PORT = 2517


class RemoteDriver(Driver):
    """Runs jobs on a machine driven by a remote rayforge-server."""

    label = _("Remote rayforge")
    subtitle = _("Execute jobs on a machine driven by rayforge-server")
    reports_granular_progress = True
    uses_gcode = True
    maturity = DriverMaturity.EXPERIMENTAL

    def __init__(self, context: RayforgeContext, machine: Machine):
        super().__init__(context, machine)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task | None = None
        self._request_lock = asyncio.Lock()
        self._pending: asyncio.Future | None = None
        self._events: asyncio.Queue = asyncio.Queue()
        self._remote_driver_name: str | None = None
        self._current_job_id: str | None = None

    @property
    def _connection_args(self) -> tuple[str, int, str]:
        args = self._machine.driver_args
        host = str(args.get("host") or "127.0.0.1")
        port = int(args.get("port") or DEFAULT_PORT)
        machine_id = str(args.get("machine_id") or "")
        return host, port, machine_id

    @property
    def resource_uri(self) -> str:
        host, port, machine_id = self._connection_args
        return f"remote://{host}:{port}/{machine_id}"

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates")

    @property
    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        if not kwargs.get("host"):
            raise DriverPrecheckError(
                _("Enter the host name of the rayforge-server.")
            )

    def _setup_implementation(self, **kwargs: Any) -> None:
        pass

    @classmethod
    def get_setup_vars(cls) -> VarSet:
        return VarSet(
            title=_("Remote rayforge-server"),
            vars=[
                HostnameVar(
                    key="host",
                    label=_("Hostname"),
                    description=_(
                        "The host name or IP address of the rayforge-server"
                    ),
                ),
                PortVar(
                    key="port",
                    label=_("Port"),
                    description=_("The rayforge-server port"),
                    default=DEFAULT_PORT,
                ),
                HostnameVar(
                    key="machine_id",
                    label=_("Machine ID"),
                    description=_(
                        "The ID of the machine to drive on the server. "
                        "Optional when the server hosts a single machine."
                    ),
                ),
            ],
        )

    @classmethod
    def create_encoder(cls, machine: Machine) -> OpsEncoder:
        """Fallback encoder used before the remote config is known.

        The real encoder is chosen per instance in :meth:`get_encoder`
        once the server machine's driver class has been fetched.
        """
        from ...pipeline.encoder.gcode import GcodeEncoder

        assert machine.dialect is not None
        return GcodeEncoder(machine.dialect)

    def get_encoder(self) -> OpsEncoder:
        """Returns the encoder of the remote machine's real driver.

        The server machine's configuration carries the driver class
        its hardware needs; the local mirror uses that class's
        encoder so the client encodes exactly what the server's
        driver expects. Falls back to the local dialect's G-code
        encoder until the remote configuration has been fetched.
        """
        # Imported here: the driver package's registry imports this
        # module, so a top-level import would be circular.
        from ..driver import get_driver_cls

        if self._remote_driver_name:
            driver_cls = get_driver_cls(self._remote_driver_name)
            return driver_cls.create_encoder(self._machine)
        return super().get_encoder()

    def get_setting_vars(self) -> list[VarSet]:
        return [VarSet(title=_("No settings"))]

    # -- Connection ------------------------------------------------------

    async def _connect_implementation(self) -> None:
        host, port, _machine_id = self._connection_args
        self.connection_status_changed.send(
            self, status=TransportStatus.CONNECTING
        )
        try:
            self._reader, self._writer = await asyncio.open_connection(
                host, port
            )
        except OSError as e:
            raise DeviceConnectionError(
                _("Cannot connect to {host}:{port}: {error}").format(
                    host=host, port=port, error=e
                )
            ) from e

        self._events = asyncio.Queue()
        self._read_task = asyncio.create_task(self._read_loop())

        welcome = await self._request(
            {"type": "hello", "version": PROTOCOL_VERSION}
        )
        if welcome.get("type") != "welcome":
            raise DeviceConnectionError(
                welcome.get("message", _("Handshake failed."))
            )

        config = await self._request(
            {
                "type": "get_machine_config",
                "machine_id": _machine_id,
            }
        )
        if config.get("type") != "machine_config":
            raise DeviceConnectionError(
                config.get("message", _("Server did not send a machine."))
            )
        machine_dict = config["config"].get("machine", {})
        self._remote_driver_name = machine_dict.get("driver")

        self.state.status = DeviceStatus.IDLE
        self.state_changed.send(self, state=self.state)
        self.connection_status_changed.send(
            self, status=TransportStatus.CONNECTED
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._reader = None
        self.connection_status_changed.send(
            self, status=TransportStatus.DISCONNECTED
        )

    async def _read_loop(self) -> None:
        """Routes incoming messages to requests or the event queue."""
        assert self._reader is not None
        try:
            while True:
                message = await read_message(self._reader)
                if message.get("type") in _EVENT_TYPES:
                    self._events.put_nowait(message)
                elif self._pending is not None and not self._pending.done():
                    self._pending.set_result(message)
                else:
                    logger.debug(f"Dropping unmatched message: {message}")
        except (asyncio.IncompleteReadError, ConnectionError):
            logger.info("Connection to rayforge-server lost.")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error reading from rayforge-server")
        finally:
            self._events.put_nowait({"type": "connection_lost"})
            if self._pending is not None and not self._pending.done():
                self._pending.set_exception(
                    DeviceConnectionError(_("Connection lost."))
                )

    async def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Sends a request and awaits its reply (one at a time)."""
        writer = self._writer
        if writer is None or writer.is_closing():
            raise DeviceConnectionError(_("Not connected."))
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            self._pending = loop.create_future()
            await write_message(writer, message)
            return await self._pending

    # -- Job execution ---------------------------------------------------

    async def run(
        self,
        encoded: EncodedOutput,
        doc: Doc,
        ops: Ops,
        on_command_done=None,
    ) -> None:
        """Submits the encoded job and tracks it until it finishes."""
        machine = self._machine
        machine_id = self._connection_args[2]
        payload = JobPayload(
            encoded=encoded,
            ops=ops,
            doc_name=doc.name if doc is not None else "",
            kind=JobKind.SEND,
            distance_mm=ops.distance(),
            estimated_seconds=ops.estimate_time(
                default_feed_rate=machine.max_cut_speed,
                default_rapid_rate=machine.max_travel_speed,
                acceleration=machine.acceleration,
            ),
            command_times=[
                float(t)
                for t in ops.estimate_command_times(
                    default_feed_rate=machine.max_cut_speed,
                    default_rapid_rate=machine.max_travel_speed,
                    acceleration=machine.acceleration,
                )
            ],
        )
        accepted = await self._request(
            {
                "type": "submit_job",
                "machine_id": machine_id,
                "payload": base64.b64encode(payload.to_bytes()).decode(
                    "ascii"
                ),
            }
        )
        if accepted.get("type") != "job_accepted":
            raise DeviceConnectionError(
                accepted.get("message", _("Job was rejected."))
            )
        job_id = accepted["job_id"]
        self._current_job_id = job_id

        while True:
            event = await self._events.get()
            if event.get("type") == "connection_lost":
                self._current_job_id = None
                raise DeviceConnectionError(_("Connection lost."))
            if event.get("job_id") != job_id:
                continue
            if event.get("type") == "job_progress":
                if on_command_done is not None and "op_index" in event:
                    result = on_command_done(event["op_index"])
                    if asyncio.iscoroutine(result):
                        await result
            elif event.get("type") == "job_finished":
                self._current_job_id = None
                outcome = event.get("outcome")
                if outcome == "FAILED":
                    raise DeviceConnectionError(
                        event.get("message") or _("Job failed.")
                    )
                self.job_finished.send(self)
                return

    async def run_raw(self, machine_code: str) -> None:
        machine_id = self._connection_args[2]
        reply = await self._request(
            {
                "type": "run_raw",
                "machine_id": machine_id,
                "code": machine_code,
            }
        )
        if reply.get("type") != "ok":
            raise DeviceConnectionError(
                reply.get("message", _("Command was rejected."))
            )

    async def cancel(self, emergency: bool = False) -> None:
        """Cancels the job currently running on the server, if any."""
        if not self.is_connected or self._current_job_id is None:
            return
        try:
            await self._request(
                {"type": "cancel_job", "job_id": self._current_job_id}
            )
        except DeviceConnectionError:
            logger.debug("cancel_job failed", exc_info=True)

    async def set_hold(self, hold: bool = True) -> None:
        """Requests or releases a feed hold on the server, if a job runs."""
        if not self.is_connected or self._current_job_id is None:
            return
        try:
            await self._request(
                {
                    "type": "set_hold",
                    "job_id": self._current_job_id,
                    "hold": hold,
                }
            )
        except DeviceConnectionError:
            logger.debug("set_hold failed", exc_info=True)

    # -- Not available remotely (yet) --------------------------------------

    def can_home(self, axis: Axis | None = None) -> bool:
        return False

    def can_jog(self, axis: Axis | None = None) -> bool:
        return False

    async def home(self, axes: Axis | None = None) -> None:
        raise DeviceConnectionError(
            _("Homing is not available via the remote driver yet.")
        )

    async def jog(self, speed: int, **deltas: float) -> None:
        raise DeviceConnectionError(
            _("Jogging is not available via the remote driver yet.")
        )

    async def select_tool(self, tool_number: int) -> None:
        raise DeviceConnectionError(
            _("Tool selection is not available via the remote driver yet.")
        )

    async def read_settings(self) -> None:
        raise DeviceConnectionError(
            _("Reading settings is not available via the remote driver yet.")
        )

    async def write_setting(self, key: str, value: Any) -> None:
        raise DeviceConnectionError(
            _("Writing settings is not available via the remote driver yet.")
        )

    async def clear_alarm(self) -> None:
        raise DeviceConnectionError(
            _("Clearing alarms is not available via the remote driver yet.")
        )

    async def set_power(self, head: Laser, percent: float) -> None:
        raise DeviceConnectionError(
            _("Live power control is not available via the remote driver yet.")
        )

    async def set_focus_power(self, head: Laser, percent: float) -> None:
        raise DeviceConnectionError(
            _("Live power control is not available via the remote driver yet.")
        )

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        raise DeviceConnectionError(
            _("Probing is not available via the remote driver yet.")
        )

    # -- Moves and WCS -----------------------------------------------------

    async def move_to(self, pos_x, pos_y, pos_z=None, speed=None) -> None:
        """Moves to an absolute position via a raw G-code command."""
        if self._machine.dialect is None:
            raise DeviceConnectionError(_("Machine has no dialect."))
        command = self._format_move_to(pos_x, pos_y, pos_z, speed)
        await self.run_raw(command)

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        """Stores the offset locally, like the dummy driver.

        WCS offsets are folded into the encoded job client-side, so
        local state is what matters for remote jobs.
        """
        cs = self._machine.coordinate_systems.get(wcs_slot)
        if cs is not None:
            cs.offset = (x, y, z if z is not None else 0.0)
            self.wcs_updated.send(
                self,
                offsets={
                    k: v.offset
                    for k, v in self._machine.coordinate_systems.items()
                },
            )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        offsets = cast(
            dict[str, Pos],
            {k: v.offset for k, v in self._machine.coordinate_systems.items()},
        )
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> str | None:
        """Treats the client's WCS selection as the source of truth."""
        return self._machine.active_wcs


# Registered at the end of the module: importing the driver registry
# above triggers the machine.driver package init, which must not
# import this module in return (that would be circular). Registering
# here puts the driver into the registry as soon as any code imports
# rayforge.machine.remote, without the registry importing it.
from ..driver import register_driver  # noqa: E402

register_driver(RemoteDriver)
