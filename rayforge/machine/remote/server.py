"""
Headless backend server for remote clients.

Runs on the machine that is physically attached to the laser cutter.
It owns the real drivers and controllers for the configured machines
and serves the remote protocol (:mod:`.protocol`): machine
configuration sync for client bootstrap, job submission, and a
generic operation router that forwards driver-level calls (jog, home,
WCS, settings, ...) from clients. Controller signals are pushed to
all connected clients as ``driver_event`` messages, so the clients
can mirror the machine state.

Jobs are owned by the server once accepted: the client may
disconnect at any time (e.g. close the laptop) and reconnect later
to observe a running job via ``get_job``. One job runs at a time
per server.

Threading model: the network loop is owned by :meth:`serve`, while
driver operations run on the task manager's event loop, as they do
in the GUI application. Events cross from the task loop to the
network loop via :meth:`asyncio.loop.call_soon_threadsafe`, and
operations cross back via :func:`asyncio.run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from rayforge import __version__

from ..job_monitor import JobMonitor
from ..job_status import JobOutcome
from .payload import JobPayload
from .protocol import (
    PROTOCOL_VERSION,
    encode_message,
    read_message,
    write_message,
)
from .serialize import (
    device_state_to_dict,
    offsets_to_dict,
    transport_status_to_name,
    var_sets_to_dicts,
)

if TYPE_CHECKING:
    from ...context import RayforgeContext
    from ...core.doc import Doc
    from ...core.varset import VarSet
    from ...shared.tasker.manager import TaskManager
    from ..models.machine import Machine

from ..driver.driver import DeviceState
from ..transport import TransportStatus

logger = logging.getLogger(__name__)


class RemoteJobServer:
    """Serves the remote protocol over TCP or a Unix domain socket."""

    def __init__(
        self,
        context: RayforgeContext,
        task_manager: TaskManager,
        host: str = "127.0.0.1",
        port: int = 0,
        socket_path: str | None = None,
    ):
        """
        Args:
            socket_path: Listen on a Unix domain socket at this path
                instead of TCP. Used for in-process local backends.
        """
        self._context = context
        self._task_manager = task_manager
        self._host = host
        self._port = port
        self._socket_path = socket_path
        self._server: asyncio.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connections: set[asyncio.StreamWriter] = set()
        self._job: _RunningJob | None = None

    @property
    def port(self) -> int:
        """The port the server listens on (TCP mode only)."""
        assert self._server is not None, "Server not started"
        return self._server.sockets[0].getsockname()[1]

    @property
    def address(self) -> str | tuple[str, int]:
        """The address clients connect to.

        The socket path in Unix socket mode, otherwise a
        ``(host, port)`` tuple.
        """
        assert self._server is not None, "Server not started"
        if self._socket_path is not None:
            return self._socket_path
        return self._host, self.port

    async def start(self) -> None:
        """Starts listening and wires the machine event forwarding."""
        self._loop = asyncio.get_running_loop()
        if self._socket_path is not None:
            # Remove a stale socket from a previous run.
            try:
                os.unlink(self._socket_path)
            except FileNotFoundError:
                pass
            self._server = await asyncio.start_unix_server(
                self._handle_connection, path=self._socket_path
            )
            address = self._socket_path
        else:
            self._server = await asyncio.start_server(
                self._handle_connection, self._host, self._port
            )
            address = f"{self._host}:{self.port}"
        for machine in self._context.machine_mgr.get_machines():
            self._wire_machine_events(machine)
        logger.info(f"Remote job server listening on {address}")

    async def stop(self) -> None:
        """Closes the listening socket and all client connections."""
        if self._server is None:
            return
        self._server.close()
        # Close the client connections first: wait_closed() waits for
        # the connection handlers, which block on reading from open
        # clients otherwise.
        for writer in list(self._connections):
            writer.close()
        self._connections.clear()
        await self._server.wait_closed()
        self._server = None
        if self._socket_path is not None:
            try:
                os.unlink(self._socket_path)
            except FileNotFoundError:
                pass

    async def serve_forever(self) -> None:
        """Runs until the server is stopped or the process exits."""
        assert self._server is not None, "Server not started"
        async with self._server:
            await self._server.serve_forever()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.info(f"Client connected: {peer}")
        self._connections.add(writer)
        try:
            while True:
                message = await read_message(reader)
                reply = await self._dispatch(message)
                reply["id"] = message.get("id")
                await write_message(writer, reply)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            logger.exception("Error while handling client")
            try:
                await write_message(
                    writer,
                    {
                        "type": "reply",
                        "id": None,
                        "ok": False,
                        "error": "internal error",
                    },
                )
            except Exception:
                logger.debug("Could not send error to client", exc_info=True)
        finally:
            self._connections.discard(writer)
            writer.close()
            logger.info(f"Client disconnected: {peer}")

    async def _dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handles one request, returning the reply payload."""
        mtype = message.get("type")
        try:
            if mtype == "hello":
                return self._hello(message)
            if mtype == "get_machine_configs":
                return self._machine_configs()
            if mtype == "get_job":
                return self._job_state()
            if mtype == "submit_job":
                return await self._submit_job(message)
            if mtype == "machine_op":
                return await self._machine_op(message)
            if mtype == "manager_op":
                return self._manager_op(message)
        except Exception as e:
            logger.exception(f"Request '{mtype}' failed")
            return {"type": "reply", "ok": False, "error": str(e)}
        return {
            "type": "reply",
            "ok": False,
            "error": f"Unknown message type: {mtype!r}",
        }

    def _hello(self, message: dict[str, Any]) -> dict[str, Any]:
        """Checks the protocol version and lists the machines."""
        version = message.get("version")
        if version != PROTOCOL_VERSION:
            return {
                "type": "reply",
                "ok": False,
                "error": (
                    f"Protocol version mismatch: client {version}, "
                    f"server {PROTOCOL_VERSION}."
                ),
            }
        machines = [
            {
                "id": machine.id,
                "name": machine.name,
                "driver_name": machine.driver_name,
                "connected": machine.is_connected(),
            }
            for machine in self._context.machine_mgr.get_machines()
        ]
        return {
            "type": "reply",
            "ok": True,
            "result": {
                "version": PROTOCOL_VERSION,
                "app_version": __version__,
                "machines": machines,
            },
        }

    def _machine_configs(self) -> dict[str, Any]:
        """Returns all machine configurations for client bootstrap."""
        configs = {}
        for machine in self._context.machine_mgr.get_machines():
            config = machine.to_dict(include_frozen_dialect=False)
            # to_dict() does not carry the machine's id; clients need
            # it to address the machine on subsequent requests.
            config.setdefault("machine", {})["id"] = machine.id
            configs[machine.id] = config
        return {"type": "reply", "ok": True, "result": {"configs": configs}}

    def _job_state(self) -> dict[str, Any]:
        if self._job is None:
            result = None
        else:
            result = self._job.state()
        return {"type": "reply", "ok": True, "result": result}

    # -- Machine configuration sync ----------------------------------------

    def _manager_op(self, message: dict[str, Any]) -> dict[str, Any]:
        """Applies machine CRUD changes from a client.

        The backend owns the machine configuration: frontends send
        their changes here so they persist and the backend machine
        controllers pick them up.
        """
        op = message.get("op")
        kwargs = message.get("kwargs", {})
        try:
            if op == "create_machine":
                return self._op_create_machine(kwargs)
            if op == "update_machine":
                return self._op_update_machine(kwargs)
            if op == "remove_machine":
                return self._op_remove_machine(kwargs)
        except Exception as e:
            logger.exception(f"Manager op '{op}' failed")
            return {"type": "reply", "ok": False, "error": str(e)}
        return {
            "type": "reply",
            "ok": False,
            "error": f"Unknown manager op: {op!r}",
        }

    def _op_create_machine(self, kwargs: dict) -> dict[str, Any]:
        """Creates a machine from a client's configuration."""
        from ..models.machine import Machine

        # Bind to the backend context: in unified local mode the
        # process-wide singleton is the frontend's context.
        machine = Machine.from_dict(kwargs["config"], context=self._context)
        self._context.machine_mgr.add_machine(machine)
        self._wire_machine_events(machine)
        return {"type": "reply", "ok": True, "result": {"id": machine.id}}

    def _op_update_machine(self, kwargs: dict) -> dict[str, Any]:
        """Replaces a machine from a client's configuration.

        Rejected while a job runs on the machine: controllers and
        the job monitor hold references to the machine object.
        """
        from ..models.machine import Machine

        machine_id = kwargs["machine_id"]
        if (
            self._job is not None
            and self._job.running
            and self._job.machine_id == machine_id
        ):
            return {
                "type": "reply",
                "ok": False,
                "error": "A job is running on this machine.",
            }
        machine_mgr = self._context.machine_mgr
        old_machine = machine_mgr.get_machine_by_id(machine_id)
        if old_machine is None:
            return {
                "type": "reply",
                "ok": False,
                "error": f"No machine with id {machine_id!r}.",
            }

        new_machine = Machine.from_dict(
            kwargs["config"], context=self._context
        )
        machine_mgr.remove_machine(machine_id)
        machine_mgr.add_machine(new_machine)
        self._wire_machine_events(new_machine)
        return {"type": "reply", "ok": True, "result": {"id": machine_id}}

    def _op_remove_machine(self, kwargs: dict) -> dict[str, Any]:
        """Removes a machine as requested by a client."""
        machine_id = kwargs["machine_id"]
        if (
            self._job is not None
            and self._job.running
            and self._job.machine_id == machine_id
        ):
            return {
                "type": "reply",
                "ok": False,
                "error": "A job is running on this machine.",
            }
        if self._context.machine_mgr.get_machine_by_id(machine_id) is None:
            return {
                "type": "reply",
                "ok": False,
                "error": f"No machine with id {machine_id!r}.",
            }
        self._context.machine_mgr.remove_machine(machine_id)
        return {"type": "reply", "ok": True, "result": None}

    # -- Job submission ---------------------------------------------------

    def _resolve_machine(
        self, message: dict[str, Any]
    ) -> Machine | dict[str, Any] | None:
        """Finds the machine a message addresses.

        When no id is given, the sole configured machine is used, so
        single-machine servers need no ids on the wire. Returns the
        machine, an error reply, or None when the id matches nothing.
        """
        machines = self._context.machine_mgr.get_machines()
        machine_id = message.get("machine_id")
        if not machine_id:
            if len(machines) == 1:
                return machines[0]
            return {
                "type": "reply",
                "ok": False,
                "error": "The server hosts multiple machines; "
                "a machine_id is required.",
            }
        return self._context.machine_mgr.get_machine_by_id(machine_id)

    def _machine_error(
        self, machine: Any, message: dict[str, Any]
    ) -> dict[str, Any]:
        if machine is None:
            return {
                "type": "reply",
                "ok": False,
                "error": f"No machine with id {message.get('machine_id')!r}.",
            }
        if isinstance(machine, dict):
            return machine
        return {
            "type": "reply",
            "ok": False,
            "error": "Unexpected machine resolution result.",
        }

    async def _submit_job(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._job is not None and self._job.running:
            return {
                "type": "reply",
                "ok": False,
                "error": "A job is already running.",
            }

        machine = self._resolve_machine(message)
        if machine is None or isinstance(machine, dict):
            return self._machine_error(machine, message)

        try:
            payload = JobPayload.from_bytes(
                base64.b64decode(message["payload"])
            )
        except (KeyError, ValueError, TypeError) as e:
            return {
                "type": "reply",
                "ok": False,
                "error": f"Bad payload: {e}",
            }
        if payload.ops is None:
            return {
                "type": "reply",
                "ok": False,
                "error": "Payload contains no ops.",
            }

        job = _RunningJob(self, machine.id, payload)
        self._job = job
        # The job runs on the task manager's loop, where the drivers
        # and their transports live; it survives client disconnects.
        job.future = asyncio.run_coroutine_threadsafe(
            self._execute_job(job, machine), self._task_manager.loop
        )
        return {
            "type": "reply",
            "ok": True,
            "result": {"job_id": job.id},
        }

    async def _execute_job(self, job: _RunningJob, machine: Machine) -> None:
        """Runs a submitted job on the task manager's event loop."""
        payload = job.payload
        ops = payload.ops
        assert ops is not None
        try:
            if not machine.is_connected():
                await machine.connect()

            estimated_seconds = payload.estimated_seconds
            if estimated_seconds is None:
                estimated_seconds = ops.estimate_time(
                    default_feed_rate=machine.max_cut_speed,
                    default_rapid_rate=machine.max_travel_speed,
                    acceleration=machine.acceleration,
                )
            monitor = JobMonitor(
                ops,
                machine=machine,
                kind=payload.kind,
                context=self._context,
                estimated_seconds=estimated_seconds,
                default_feed_rate=machine.max_cut_speed,
                default_rapid_rate=machine.max_travel_speed,
                acceleration=machine.acceleration,
            )
            job.monitor = monitor
            monitor.begin()
            try:
                # Drivers only consume the doc during encoding, which
                # happened on the client; the server has no document.
                doc = cast("Doc", None)
                on_command_done: Callable[[int], None] | None = None
                if machine.reports_granular_progress:
                    # Wrap the monitor so each acknowledged command is
                    # both tracked and broadcast to the clients.
                    def _on_command_done(op_index: int) -> None:
                        monitor.update_progress(op_index)
                        job.report_op_done(op_index)

                    on_command_done = _on_command_done

                await machine.driver.run(
                    payload.encoded,
                    doc,
                    ops,
                    on_command_done=on_command_done,
                )
            except asyncio.CancelledError:
                monitor.end(JobOutcome.CANCELLED)
                job.finish("CANCELLED", "")
                raise
            except Exception as e:
                logger.exception("Remote job failed")
                monitor.end(JobOutcome.FAILED, str(e))
                job.finish("FAILED", str(e))
            else:
                monitor.end(JobOutcome.COMPLETED)
                job.finish("COMPLETED", "")
                machine.add_machine_hours(estimated_seconds / 3600.0)
        except Exception as e:
            logger.exception("Remote job could not be started")
            job.finish("FAILED", str(e))

    # -- Machine operations -----------------------------------------------

    async def _machine_op(self, message: dict[str, Any]) -> dict[str, Any]:
        """Forwards a driver-level operation to the real driver."""
        machine = self._resolve_machine(message)
        if machine is None or isinstance(machine, dict):
            return self._machine_error(machine, message)

        op = message.get("op")
        kwargs = message.get("kwargs", {})
        handler = _OP_HANDLERS.get(cast(str, op))
        if handler is None:
            return {
                "type": "reply",
                "ok": False,
                "error": f"Unknown machine op: {op!r}",
            }
        if not machine.is_connected() and op not in (
            "connect",
            "get_driver_caps",
        ):
            return {
                "type": "reply",
                "ok": False,
                "error": "Machine is not connected.",
            }

        coroutine = handler(self, machine, kwargs)
        future = asyncio.run_coroutine_threadsafe(
            coroutine, self._task_manager.loop
        )
        try:
            result = await asyncio.wrap_future(future)
        except Exception as e:
            logger.exception(f"Machine op '{op}' failed")
            return {"type": "reply", "ok": False, "error": str(e)}
        return {"type": "reply", "ok": True, "result": result}

    async def _op_connect(self, machine: Machine, kwargs: dict) -> Any:
        await machine.connect()
        return self._status_snapshot(machine)

    def _status_snapshot(self, machine: Machine) -> dict[str, Any]:
        """The machine's current state, for clients that just connected."""
        return {
            "connection_status": transport_status_to_name(
                machine.connection_status
            ),
            "device_state": device_state_to_dict(machine.device_state),
        }

    async def _op_disconnect(self, machine: Machine, kwargs: dict) -> Any:
        await machine.disconnect()

    async def _op_home(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.home(kwargs.get("axes"))

    async def _op_jog(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.jog(
            speed=kwargs["speed"], **kwargs.get("deltas", {})
        )

    async def _op_run_raw(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.run_raw(kwargs["code"])

    async def _op_select_tool(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.select_tool(kwargs["tool_number"])

    async def _op_move_to(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.move_to(
            kwargs["x"],
            kwargs["y"],
            kwargs.get("z"),
            speed=kwargs.get("speed"),
        )

    async def _op_set_power(self, machine: Machine, kwargs: dict) -> Any:
        head = _find_head(machine, kwargs.get("head_uid"))
        await machine.driver.set_power(head, kwargs["percent"])

    async def _op_set_focus_power(self, machine: Machine, kwargs: dict) -> Any:
        head = _find_head(machine, kwargs.get("head_uid"))
        await machine.driver.set_focus_power(head, kwargs["percent"])

    async def _op_cancel(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.cancel(kwargs.get("emergency", False))

    async def _op_set_hold(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.set_hold(kwargs.get("hold", True))

    async def _op_clear_alarm(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.clear_alarm()

    async def _op_set_wcs_offset(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.set_wcs_offset(
            kwargs["slot"], kwargs["x"], kwargs["y"], kwargs.get("z")
        )

    async def _op_read_wcs_offsets(
        self, machine: Machine, kwargs: dict
    ) -> Any:
        offsets = await machine.driver.read_wcs_offsets()
        return offsets_to_dict(offsets) if offsets else None

    async def _op_read_parser_state(
        self, machine: Machine, kwargs: dict
    ) -> Any:
        return await machine.driver.read_parser_state()

    async def _op_select_wcs(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.select_wcs(kwargs["wcs"])

    async def _op_run_probe_cycle(self, machine: Machine, kwargs: dict) -> Any:
        from raygeo.ops.axis import Axis

        axis = getattr(Axis, str(kwargs["axis"]))
        pos = await machine.driver.run_probe_cycle(
            axis,
            kwargs["max_travel"],
            kwargs["feed_rate"],
        )
        return list(pos) if pos is not None else None

    async def _op_read_settings(self, machine: Machine, kwargs: dict) -> Any:
        var_sets = await _await_settings(machine.driver)
        return var_sets_to_dicts(var_sets)

    async def _op_write_setting(self, machine: Machine, kwargs: dict) -> Any:
        await machine.driver.write_setting(kwargs["key"], kwargs["value"])

    def _driver_caps(self, machine: Machine) -> dict[str, Any]:
        """Introspection data the client proxy serves synchronously."""
        driver = machine.driver
        pwm = {}
        for head in machine.heads:
            params = driver.get_pwm_params(head)
            pwm[head.uid] = {
                "supported": driver.supports_pwm(head),
                "params": params.__dict__ if params is not None else None,
            }
        return {
            "reports_granular_progress": driver.reports_granular_progress,
            "can_home": driver.can_home(),
            "can_jog": driver.can_jog(),
            "machine_space_wcs": driver.machine_space_wcs,
            "machine_space_wcs_display_name": (
                driver.machine_space_wcs_display_name
            ),
            "supported_wcs": driver.supported_wcs,
            "setting_vars": var_sets_to_dicts(driver.get_setting_vars()),
            "pwm": pwm,
        }

    async def _op_get_driver_caps(self, machine: Machine, kwargs: dict) -> Any:
        return self._driver_caps(machine)

    # -- Event forwarding -------------------------------------------------

    def _wire_machine_events(self, machine: Machine) -> None:
        """Broadcasts the machine's controller signals to all clients."""
        controller = machine.context.machine_mgr.get_controller(machine.id)
        controller.connection_status_changed.connect(
            self._on_connection_status(machine)
        )
        controller.state_changed.connect(self._on_state_changed(machine))
        controller.command_status_changed.connect(
            self._on_command_status_changed(machine)
        )
        controller.wcs_updated.connect(self._on_wcs_updated(machine))

    def _on_connection_status(self, machine: Machine) -> Callable[..., None]:
        def handler(sender, status=None, message=None):
            self._broadcast_threadsafe(
                {
                    "type": "driver_event",
                    "machine_id": machine.id,
                    "event": "connection_status_changed",
                    "payload": {
                        "status": transport_status_to_name(
                            cast(TransportStatus, status)
                        ),
                        "message": message,
                    },
                }
            )

        return handler

    def _on_state_changed(self, machine: Machine) -> Callable[..., None]:
        def handler(sender, state=None):
            self._broadcast_threadsafe(
                {
                    "type": "driver_event",
                    "machine_id": machine.id,
                    "event": "state_changed",
                    "payload": {
                        "state": device_state_to_dict(cast(DeviceState, state))
                    },
                }
            )

        return handler

    def _on_command_status_changed(
        self, machine: Machine
    ) -> Callable[..., None]:
        def handler(sender, status=None, message=None):
            self._broadcast_threadsafe(
                {
                    "type": "driver_event",
                    "machine_id": machine.id,
                    "event": "command_status_changed",
                    "payload": {
                        "status": transport_status_to_name(
                            cast(TransportStatus, status)
                        ),
                        "message": message,
                    },
                }
            )

        return handler

    def _on_wcs_updated(self, machine: Machine) -> Callable[..., None]:
        def handler(sender, **kwargs):
            offsets = {
                slot: cs.offset
                for slot, cs in machine.coordinate_systems.items()
            }
            self._broadcast_threadsafe(
                {
                    "type": "driver_event",
                    "machine_id": machine.id,
                    "event": "wcs_updated",
                    "payload": {"offsets": offsets_to_dict(offsets)},
                }
            )

        return handler

    def _broadcast(self, message: dict[str, Any]) -> None:
        """Sends a message to all clients; runs on the network loop."""
        if self._loop is None or self._loop.is_closed():
            return
        data = encode_message(message)
        for writer in list(self._connections):
            try:
                writer.write(data)
            except Exception:
                logger.debug("Dropping disconnected client", exc_info=True)
                self._connections.discard(writer)

    def _broadcast_threadsafe(self, message: dict[str, Any]) -> None:
        """Marshals a broadcast from the task loop to the network loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._broadcast, message)


async def _await_settings(driver: Any) -> list[VarSet]:
    """Runs read_settings and returns the VarSets it produced."""
    future: asyncio.Future = asyncio.get_running_loop().create_future()

    def on_settings_read(sender, var_sets: list):
        try:
            driver.settings_read.disconnect(on_settings_read)
        finally:
            if not future.done():
                future.set_result(var_sets)

    driver.settings_read.connect(on_settings_read)
    try:
        await driver.read_settings()
    except Exception:
        driver.settings_read.disconnect(on_settings_read)
        raise
    return await asyncio.wait_for(future, timeout=30)


def _find_head(machine: Machine, head_uid: str | None) -> Any:
    """Finds a head by uid, defaulting to the default laser head.

    The return value is intentionally untyped: the machine's heads
    are a mix of head types and the drivers expect LaserHead.
    """
    if head_uid is not None:
        for head in machine.heads:
            if head.uid == head_uid:
                return head
    head = machine.get_default_laser_head()
    if head is None:
        raise ValueError("Machine has no laser heads configured.")
    return head


_OP_HANDLERS: dict[str, Callable[..., Any]] = {
    "connect": RemoteJobServer._op_connect,
    "disconnect": RemoteJobServer._op_disconnect,
    "home": RemoteJobServer._op_home,
    "jog": RemoteJobServer._op_jog,
    "run_raw": RemoteJobServer._op_run_raw,
    "select_tool": RemoteJobServer._op_select_tool,
    "move_to": RemoteJobServer._op_move_to,
    "set_power": RemoteJobServer._op_set_power,
    "set_focus_power": RemoteJobServer._op_set_focus_power,
    "cancel": RemoteJobServer._op_cancel,
    "set_hold": RemoteJobServer._op_set_hold,
    "clear_alarm": RemoteJobServer._op_clear_alarm,
    "set_wcs_offset": RemoteJobServer._op_set_wcs_offset,
    "read_wcs_offsets": RemoteJobServer._op_read_wcs_offsets,
    "read_parser_state": RemoteJobServer._op_read_parser_state,
    "select_wcs": RemoteJobServer._op_select_wcs,
    "run_probe_cycle": RemoteJobServer._op_run_probe_cycle,
    "read_settings": RemoteJobServer._op_read_settings,
    "write_setting": RemoteJobServer._op_write_setting,
    "get_driver_caps": RemoteJobServer._op_get_driver_caps,
}


class _RunningJob:
    """Tracks the lifecycle of a job accepted by the server."""

    def __init__(
        self,
        server: RemoteJobServer,
        machine_id: str,
        payload: JobPayload,
    ):
        self.id = str(uuid.uuid4())
        self.machine_id = machine_id
        self.payload = payload
        self.running = True
        self.outcome: str | None = None
        self.message = ""
        self.metrics: dict[str, Any] | None = None
        self.monitor: JobMonitor | None = None
        self.future: Any = None
        self._server = server

    def report_op_done(self, op_index: int) -> None:
        """Broadcasts progress for an acknowledged command.

        Runs on the task manager loop, from the driver's
        on_command_done callback.
        """
        monitor = self.monitor
        metrics = monitor.metrics if monitor is not None else None
        self.metrics = metrics
        self._server._broadcast_threadsafe(
            {
                "type": "job_progress",
                "job_id": self.id,
                "op_index": op_index,
                "metrics": metrics,
            }
        )

    def finish(self, outcome: str, message: str) -> None:
        """Marks the job as ended and notifies the clients."""
        self.running = False
        self.outcome = outcome
        self.message = message
        self._server._broadcast_threadsafe(
            {
                "type": "job_finished",
                "job_id": self.id,
                "outcome": outcome,
                "message": message,
            }
        )

    def state(self) -> dict[str, Any]:
        """Returns a JSON-safe snapshot for the ``get_job`` reply."""
        return {
            "job_id": self.id,
            "machine_id": self.machine_id,
            "doc_name": self.payload.doc_name,
            "kind": self.payload.kind.name,
            "running": self.running,
            "outcome": self.outcome,
            "message": self.message,
            "metrics": self.metrics,
        }
