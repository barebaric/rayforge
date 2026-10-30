"""
Driver proxy for remote machines.

A :class:`RemoteDriverProxy` is the driver instance of a machine
mirror on a remote client (see :mod:`rayforge.machine.remote.client`).
It is never chosen by the user and never appears in the driver
registry: the client bootstrap substitutes it for the machine's real
driver class, so the entire application — controllers, UI, pipeline —
works against it exactly as against a local driver.

Every driver method is forwarded to the server as a ``machine_op``
request; driver signals arrive as ``driver_event`` messages and are
re-emitted locally. Introspection data that callers need
synchronously (settings variables, PWM support, progress reporting)
is cached from the server on connect.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ...core.varset import VarSet
from ..driver.driver import (
    DeviceConnectionError,
    Driver,
    Pos,
    PWMParams,
)
from ..job_status import JobKind
from .payload import JobPayload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from raygeo.ops import Ops
    from raygeo.ops.axis import Axis

    from ...context import RayforgeContext
    from ...core.doc import Doc
    from ...pipeline.encoder.base import EncodedOutput
    from ..models.head import Head
    from ..models.machine import Machine
    from .client import RemoteClient

logger = logging.getLogger(__name__)


class RemoteDriverProxy(Driver):
    """A driver whose hardware lives behind a rayforge-server."""

    label = "Remote"
    subtitle = "Driven by a rayforge-server"

    # Mirrored from the remote driver class on construction:
    reports_granular_progress: bool = False

    def __init__(
        self,
        context: RayforgeContext,
        machine: Machine,
        client: RemoteClient,
        machine_id: str,
        remote_driver_name: str | None,
    ):
        super().__init__(context, machine)
        self._client = client
        self._machine_id = machine_id
        self._remote_driver_name = remote_driver_name
        self._caps: dict[str, Any] | None = None
        client.register_proxy(machine_id, self)
        self._mirror_remote_class()

    def _mirror_remote_class(self) -> None:
        """Copies the remote driver class's introspection attributes."""
        from ..driver import get_driver_cls

        remote_cls = get_driver_cls(self._remote_driver_name or "")
        for attr in (
            "reports_granular_progress",
            "supports_settings",
            "uses_gcode",
            "native_overscan",
            "supports_unit_detection",
            "supports_multi_depth_raster",
            "supports_probing",
            "z_homing_modes",
            "label",
            "subtitle",
        ):
            setattr(self, attr, getattr(remote_cls, attr))

    # -- Introspection --

    @property
    def machine_space_wcs(self) -> str:
        if self._caps is not None:
            return self._caps.get("machine_space_wcs", "MACHINE")
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        if self._caps is not None:
            return self._caps.get(
                "machine_space_wcs_display_name", "Machine Coordinates"
            )
        return "Machine Coordinates"

    @property
    def supported_wcs(self) -> list[str]:
        if self._caps is not None:
            return list(self._caps.get("supported_wcs", ["G54"]))
        return ["G54"]

    def can_home(self, axis: Axis | None = None) -> bool:
        return bool(self._caps and self._caps.get("can_home"))

    def can_jog(self, axis: Axis | None = None) -> bool:
        return bool(self._caps and self._caps.get("can_jog"))

    def supports_pwm(self, head: Head) -> bool:
        pwm = (self._caps or {}).get("pwm", {})
        entry = pwm.get(head.uid, {})
        return bool(entry.get("supported"))

    def get_pwm_params(self, head: Head) -> PWMParams | None:
        pwm = (self._caps or {}).get("pwm", {})
        entry = pwm.get(head.uid, {})
        params = entry.get("params")
        if params is None:
            return None
        return PWMParams(**params)

    def get_setting_vars(self) -> list[VarSet]:
        from .serialize import var_sets_from_dicts

        if self._caps is None:
            return []
        return var_sets_from_dicts(self._caps.get("setting_vars", []))

    # -- Driver interface plumbing --

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        pass

    def _setup_implementation(self, **kwargs: Any) -> None:
        pass

    @classmethod
    def get_setup_vars(cls) -> VarSet:
        return VarSet(title="Remote")

    @classmethod
    def create_encoder(cls, machine: Machine) -> Any:
        raise NotImplementedError(
            "RemoteDriverProxy.create_encoder() is never used; "
            "get_encoder() delegates to the remote driver class."
        )

    def get_encoder(self) -> Any:
        """Returns the encoder of the remote machine's real driver."""
        from ..driver import get_driver_cls

        driver_cls = get_driver_cls(self._remote_driver_name or "")
        return driver_cls.create_encoder(self._machine)

    @property
    def resource_uri(self) -> str | None:
        # The server manages its own resource conflicts; sharing one
        # server between several client machines is fine.
        return None

    # -- Connection ----------------------------------------------------------

    async def _connect_implementation(self) -> None:
        from .serialize import (
            device_state_from_dict,
            transport_status_from_name,
        )

        self._caps = await self._client.machine_op(
            self._machine_id, "get_driver_caps"
        )
        snapshot = await self._client.machine_op(self._machine_id, "connect")
        # The server machine may already have been connected, in
        # which case no change signal fires: replay the reported
        # state so the local machine mirrors the server's.
        self.connection_status_changed.send(
            self,
            status=transport_status_from_name(
                snapshot.get("connection_status")
            ),
        )
        self.state_changed.send(
            self,
            state=device_state_from_dict(snapshot.get("device_state", {})),
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        self._client.unregister_proxy(self._machine_id, self)

    # -- Event handling -----------------------------------------------

    def _handle_event(self, event: str, payload: dict[str, Any]) -> None:
        """Re-emits a server-pushed driver signal locally."""
        from .serialize import (
            device_state_from_dict,
            offsets_from_dict,
            transport_status_from_name,
        )

        if event == "connection_status_changed":
            self.connection_status_changed.send(
                self,
                status=transport_status_from_name(payload["status"]),
                message=payload.get("message"),
            )
        elif event == "state_changed":
            self.state_changed.send(
                self, state=device_state_from_dict(payload["state"])
            )
        elif event == "command_status_changed":
            self.command_status_changed.send(
                self,
                status=transport_status_from_name(payload["status"]),
                message=payload.get("message"),
            )
        elif event == "wcs_updated":
            self.wcs_updated.send(
                self, offsets=offsets_from_dict(payload.get("offsets", {}))
            )
        else:
            logger.debug(f"Dropping unhandled remote event: {event}")

    # -- Job execution ------------------------------------------------

    async def run(
        self,
        encoded: EncodedOutput,
        doc: Doc,
        ops: Ops,
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        """Submits the encoded job and tracks it until it finishes."""
        machine = self._machine
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
        )
        job_id = await self._client.submit_job(self._machine_id, payload)
        queue = self._client.job_events(job_id)
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "job_progress":
                    if on_command_done is not None:
                        result = on_command_done(event["op_index"])
                        if asyncio.iscoroutine(result):
                            await result
                elif event.get("type") == "job_finished":
                    outcome = event.get("outcome")
                    if outcome == "FAILED":
                        raise DeviceConnectionError(
                            event.get("message") or "Job failed."
                        )
                    self.job_finished.send(self)
                    return
        finally:
            self._client.remove_job_events(job_id)

    async def cancel(self, emergency: bool = False) -> None:
        await self._client.machine_op(
            self._machine_id, "cancel", emergency=emergency
        )

    async def set_hold(self, hold: bool = True) -> None:
        await self._client.machine_op(self._machine_id, "set_hold", hold=hold)

    # -- Interactive commands --

    async def run_raw(self, machine_code: str) -> None:
        await self._client.machine_op(
            self._machine_id, "run_raw", code=machine_code
        )

    async def home(self, axes: Axis | None = None) -> None:
        await self._client.machine_op(
            self._machine_id, "home", axes=axes.name if axes else None
        )

    async def jog(self, speed: int, **deltas: float) -> None:
        await self._client.machine_op(
            self._machine_id, "jog", speed=speed, deltas=deltas
        )

    async def move_to(self, pos_x, pos_y, pos_z=None, speed=None) -> None:
        await self._client.machine_op(
            self._machine_id,
            "move_to",
            x=pos_x,
            y=pos_y,
            z=pos_z,
            speed=speed,
        )

    async def select_tool(self, tool_number: int) -> None:
        await self._client.machine_op(
            self._machine_id, "select_tool", tool_number=tool_number
        )

    async def set_power(self, head: Head, percent: float) -> None:
        await self._client.machine_op(
            self._machine_id, "set_power", head_uid=head.uid, percent=percent
        )

    async def set_focus_power(self, head: Head, percent: float) -> None:
        await self._client.machine_op(
            self._machine_id,
            "set_focus_power",
            head_uid=head.uid,
            percent=percent,
        )

    async def clear_alarm(self) -> None:
        await self._client.machine_op(self._machine_id, "clear_alarm")

    async def read_settings(self) -> None:
        from .serialize import var_sets_from_dicts

        result = await self._client.machine_op(
            self._machine_id, "read_settings"
        )
        self.settings_read.send(
            self, var_sets=var_sets_from_dicts(result or [])
        )

    async def write_setting(self, key: str, value: Any) -> None:
        await self._client.machine_op(
            self._machine_id, "write_setting", key=key, value=value
        )

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        await self._client.machine_op(
            self._machine_id,
            "set_wcs_offset",
            slot=wcs_slot,
            x=x,
            y=y,
            z=z,
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        from .serialize import offsets_from_dict

        result = await self._client.machine_op(
            self._machine_id, "read_wcs_offsets"
        )
        offsets = offsets_from_dict(result or {})
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> str | None:
        return await self._client.machine_op(
            self._machine_id, "read_parser_state"
        )

    async def select_wcs(self, wcs: str) -> None:
        await self._client.machine_op(self._machine_id, "select_wcs", wcs=wcs)

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        from .serialize import pos_from_json

        result = await self._client.machine_op(
            self._machine_id,
            "run_probe_cycle",
            axis=axis.name,
            max_travel=max_travel,
            feed_rate=feed_rate,
        )
        return pos_from_json(result)
