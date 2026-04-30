import asyncio
import inspect
import json
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    cast,
    TYPE_CHECKING,
)
from gettext import gettext as _

import aiohttp

from ....context import RayforgeContext
from ....core.ops.axis import Axis
from ....core.varset import (
    AppKeyVar,
    HostnameVar,
    PortVar,
    VarSet,
)
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ....pipeline.encoder.base import EncodedOutput, OpsEncoder
from ....pipeline.encoder.gcode import GcodeEncoder
from ...transport import TransportStatus
from ..driver import (
    DeviceConnectionError,
    DeviceError,
    DeviceStatus,
    Driver,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)

if TYPE_CHECKING:
    from ....core.doc import Doc
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)

_STATE_MAP: Dict[str, DeviceStatus] = {
    "Operational": DeviceStatus.IDLE,
    "Printing": DeviceStatus.RUN,
    "Pausing": DeviceStatus.HOLD,
    "Paused": DeviceStatus.HOLD,
    "Cancelling": DeviceStatus.RUN,
    "Starting": DeviceStatus.RUN,
    "Error": DeviceStatus.ALARM,
    "Offline": DeviceStatus.UNKNOWN,
    "Offline after error": DeviceStatus.ALARM,
    "Opening serial connection": DeviceStatus.UNKNOWN,
    "Connecting": DeviceStatus.UNKNOWN,
    "Closed": DeviceStatus.UNKNOWN,
}

_POLL_INTERVAL = 2.0
_RECONNECT_INTERVAL = 5.0
_WS_PING_INTERVAL = 30.0


class OctoPrintDriver(Driver):
    """
    Submits G-code jobs to an OctoPrint server via its REST API and
    monitors live printer state through SockJS WebSocket push updates
    with a polling REST fallback.
    """

    label = _("OctoPrint")
    subtitle = _("Submit G-code to an OctoPrint server")
    supports_settings = False
    reports_granular_progress = False
    uses_gcode = True
    maturity = DriverMaturity.UNTESTED

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self.host: Optional[str] = None
        self.port: int = 80
        self._api_key: Optional[str] = None
        self._base_url: Optional[str] = None
        self.keep_running: bool = False
        self._connection_task: Optional[asyncio.Task] = None
        self._job_active: bool = False
        self._job_done_event = asyncio.Event()
        self._session_key: Optional[str] = None
        self._user_name: Optional[str] = None
        self._auth_retried: bool = False

    @property
    def machine_space_wcs(self) -> str:
        return "G53"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates (G53)")

    @property
    def resource_uri(self) -> Optional[str]:
        if self.host:
            return f"tcp://{self.host}:{self.port}"
        return None

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        if host and not is_valid_hostname_or_ip(host):
            raise DriverPrecheckError(
                _("Invalid hostname or IP address: '{host}'").format(host=host)
            )

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                HostnameVar(
                    key="host",
                    label=_("Hostname"),
                    description=_(
                        "IP address or hostname of the OctoPrint server"
                    ),
                ),
                PortVar(
                    key="port",
                    label=_("Port"),
                    description=_("HTTP port of the OctoPrint server"),
                    default=80,
                ),
                AppKeyVar(
                    key="api_key",
                    label=_("API Key"),
                    app_name="RayForge",
                    probe_url="http://{host}:{port}/plugin/appkeys/probe",
                    request_url="http://{host}:{port}/plugin/appkeys/request",
                    poll_url="http://{host}:{port}"
                    "/plugin/appkeys/request"
                    "/{{app_token}}",
                    description=_(
                        "Enter an API key manually or click "
                        "'Request Access' to obtain one via "
                        "OctoPrint's Application Keys plugin."
                    ),
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        assert machine.dialect is not None
        return GcodeEncoder(machine.dialect)

    @staticmethod
    def _extract_api_key(data: Optional[str]) -> Optional[str]:
        if not data:
            return None
        try:
            tokens = json.loads(data)
            if isinstance(tokens, dict):
                return tokens.get("api_key") or tokens.get("access_token")
            return str(tokens)
        except (json.JSONDecodeError, TypeError):
            return data.strip() if data else None

    def _setup_implementation(self, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        port = cast(int, kwargs.get("port", 80))
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))

        raw_key = kwargs.get("api_key", "")
        api_key = self._extract_api_key(str(raw_key) if raw_key else "")
        if not api_key:
            raise DriverSetupError(
                _(
                    "API key must be configured. Use the "
                    "'Request Access' button or enter an API "
                    "key manually."
                )
            )

        self.host = host
        self.port = port
        self._api_key = api_key
        self._base_url = f"http://{host}:{port}"

    async def cleanup(self):
        self.keep_running = False
        self._job_active = False
        self._job_done_event.set()
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None
        await super().cleanup()

    async def _api_request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers["X-Api-Key"] = self._api_key

        log_data = f"{method} {url}"
        logger.debug(
            log_data,
            extra={
                "log_category": "RAW_IO",
                "direction": "TX",
                "data": log_data,
            },
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=headers, **kwargs
                ) as response:
                    if response.status == 403:
                        await self._handle_auth_error()
                        raise DeviceConnectionError(
                            _(
                                "Authentication failed. "
                                "API key may be invalid or expired."
                            )
                        )
                    response.raise_for_status()

                    if response.status == 204:
                        return None

                    ct = response.content_type or ""
                    if "json" in ct:
                        data = await response.json()
                    else:
                        data = await response.text()

            logger.debug(
                f"Response ({response.status}): {str(data)[:200]}"
                if isinstance(data, str)
                else f"Response ({response.status}): <json>",
                extra={
                    "log_category": "RAW_IO",
                    "direction": "RX",
                    "data": str(data)[:500].encode("utf-8", errors="replace"),
                },
            )
            return data
        except aiohttp.ClientError as e:
            msg = _(
                "Could not connect to OctoPrint at "
                "'{host}:{port}'. Check the address and "
                "network connection."
            ).format(host=self.host, port=self.port)
            raise DeviceConnectionError(msg) from e

    async def _handle_auth_error(self) -> None:
        if not self._auth_retried:
            self._auth_retried = True
            try:
                login = await self._api_request(
                    "POST",
                    "/api/login",
                    json={"passive": True},
                )
                if login and login.get("name"):
                    self._session_key = login.get("session")
                    self._user_name = login.get("name")
                    return
            except Exception:
                pass

        self.state.error = DeviceError(
            code=403,
            title=_("Authentication Failed"),
            description=_(
                "The API key is invalid or has expired. "
                "Please re-authenticate in device settings."
            ),
        )
        self.state.status = DeviceStatus.UNKNOWN
        self.state_changed.send(self, state=self.state)

    async def _verify_connection(self) -> None:
        login = await self._api_request(
            "POST", "/api/login", json={"passive": True}
        )
        if login is None:
            raise DeviceConnectionError(_("OctoPrint returned no login data."))
        self._session_key = login.get("session")
        self._user_name = login.get("name")
        self._auth_retried = False

    async def _connect_implementation(self) -> None:
        if not self.host:
            self._update_connection_status(
                TransportStatus.DISCONNECTED,
                "No host configured",
            )
            return
        self.keep_running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        while self.keep_running:
            self._update_connection_status(TransportStatus.CONNECTING)
            try:
                await self._verify_connection()
                try:
                    await self._run_websocket()
                except Exception as ws_err:
                    logger.info(
                        f"WebSocket unavailable, using polling: {ws_err}"
                    )
                    await self._run_polling()
            except asyncio.CancelledError:
                return
            except DeviceConnectionError as e:
                self._update_connection_status(TransportStatus.ERROR, str(e))
            except Exception as e:
                self._update_connection_status(TransportStatus.ERROR, str(e))

            if self.keep_running:
                self._update_connection_status(TransportStatus.SLEEPING)
                await asyncio.sleep(_RECONNECT_INTERVAL)

    async def _run_websocket(self) -> None:
        ws_url = f"ws://{self.host}:{self.port}/sockjs/websocket"
        self._update_connection_status(TransportStatus.CONNECTING)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(
                    ws_url,
                    heartbeat=_WS_PING_INTERVAL,
                ) as ws:
                    msg = await ws.receive()
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        raise DeviceConnectionError(
                            _("Unexpected WebSocket frame.")
                        )

                    if msg.data == "o":
                        if self._session_key and self._user_name:
                            auth_inner = json.dumps(
                                {
                                    "auth": (
                                        f"{self._user_name}:"
                                        f"{self._session_key}"
                                    )
                                }
                            )
                            await ws.send_str(json.dumps([auth_inner]))
                    elif msg.data.startswith("c["):
                        raise DeviceConnectionError(
                            _("Server closed WebSocket connection.")
                        )

                    self._update_connection_status(TransportStatus.CONNECTED)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._process_sockjs_frame(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                        if not self.keep_running:
                            break
            except aiohttp.ClientError as e:
                raise DeviceConnectionError(str(e)) from e

    def _process_sockjs_frame(self, data: str) -> None:
        if data == "o" or data == "h":
            return
        if data.startswith("c["):
            logger.info("SockJS close frame received")
            return
        if not data.startswith("a["):
            return

        try:
            messages = json.loads(data[1:])
        except json.JSONDecodeError:
            return

        for raw_msg in messages:
            if not isinstance(raw_msg, str):
                continue
            try:
                payload = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue
            self._process_push_message(payload)

    def _process_push_message(self, payload: Dict[str, Any]) -> None:
        if "current" in payload:
            self._handle_current_update(payload["current"])
        elif "history" in payload:
            self._handle_current_update(payload["history"])
        elif "event" in payload:
            self._handle_event(payload["event"])

    def _handle_current_update(self, data: Dict[str, Any]) -> None:
        state = data.get("state")
        if state:
            self._update_state_from_octoprint(state)

        progress = data.get("progress")
        if progress and self._job_active:
            completion = progress.get("completion")
            if completion is not None:
                self._update_job_progress(completion)

    def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type == "PrintDone":
            self._on_job_completed(success=True)
        elif event_type == "PrintFailed":
            self._on_job_completed(success=False)
        elif event_type == "PrintCancelled":
            self._on_job_completed(success=True)
        elif event_type == "Connected":
            logger.info(
                "OctoPrint printer connected",
                extra=self._log_extra("MACHINE_EVENT"),
            )
        elif event_type == "Disconnected":
            self.state.status = DeviceStatus.UNKNOWN
            self.state_changed.send(self, state=self.state)

    def _on_job_completed(self, success: bool) -> None:
        if not self._job_active:
            return
        if success:
            self.state.status = DeviceStatus.IDLE
            self.state.error = None
        else:
            self.state.status = DeviceStatus.ALARM
            self.state.error = DeviceError(
                code=1,
                title=_("Print Failed"),
                description=_(
                    "OctoPrint reported that the print job "
                    "failed. Check OctoPrint for details."
                ),
            )
        self.state_changed.send(self, state=self.state)
        self._job_active = False
        self._job_done_event.set()

    async def _run_polling(self) -> None:
        self._update_connection_status(TransportStatus.CONNECTED)
        while self.keep_running:
            try:
                data = await self._api_request(
                    "GET",
                    "/api/printer?exclude=temperature,sd",
                )
                if data:
                    state = data.get("state")
                    if state:
                        self._update_state_from_octoprint(state)

                if self._job_active:
                    job = await self._api_request("GET", "/api/job")
                    if job:
                        progress = job.get("progress")
                        if progress:
                            completion = progress.get("completion")
                            if completion is not None:
                                self._update_job_progress(completion)

                        job_state = job.get("state", "")
                        if job_state not in (
                            "Printing",
                            "Pausing",
                            "Paused",
                            "Cancelling",
                            "Starting",
                        ):
                            self._on_job_completed(
                                success="Error" not in job_state
                            )

            except DeviceConnectionError:
                raise
            except Exception as e:
                logger.warning(f"Polling error: {e}")

            await asyncio.sleep(_POLL_INTERVAL)

    def _update_state_from_octoprint(self, state_data: Dict[str, Any]) -> None:
        flags = state_data.get("flags", {})
        text = state_data.get("text", "Unknown")

        if flags.get("error"):
            new_status = DeviceStatus.ALARM
        elif flags.get("closedOrError") and not flags.get("operational"):
            new_status = DeviceStatus.UNKNOWN
        elif flags.get("printing"):
            new_status = DeviceStatus.RUN
        elif flags.get("paused"):
            new_status = DeviceStatus.HOLD
        elif flags.get("pausing"):
            new_status = DeviceStatus.HOLD
        elif flags.get("cancelling"):
            new_status = DeviceStatus.RUN
        elif flags.get("operational"):
            new_status = DeviceStatus.IDLE
        else:
            new_status = _STATE_MAP.get(text, DeviceStatus.UNKNOWN)

        if new_status != self.state.status:
            old = self.state.status.name
            self.state.status = new_status
            logger.info(
                f"State: {old} -> {new_status.name}",
                extra=self._log_extra("STATE_CHANGE"),
            )
            self.state_changed.send(self, state=self.state)

    def _update_job_progress(self, completion: float) -> None:
        logger.debug(
            f"Job progress: {completion:.1f}%",
            extra=self._log_extra("DRIVER_EVENT"),
        )

    async def run(
        self,
        encoded: "EncodedOutput",
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        if not self.host:
            raise DeviceConnectionError(
                _("Driver not configured with a host.")
            )

        gcode = encoded.text

        try:
            if on_command_done is not None:
                op_map = encoded.op_map
                num_ops = 0
                if op_map and op_map.op_to_machine_code:
                    num_ops = max(op_map.op_to_machine_code.keys()) + 1
                for i in range(num_ops):
                    result = on_command_done(i)
                    if inspect.isawaitable(result):
                        await result

            await self._upload_and_print(gcode, "rayforge.gcode")

            self._job_active = True
            self._job_done_event.clear()

            self._update_command_status(TransportStatus.IDLE)

            while self._job_active and self.keep_running:
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._update_connection_status(TransportStatus.ERROR, str(e))
            raise
        finally:
            self._job_active = False
            self.job_finished.send(self)

    async def _upload_and_print(self, gcode: str, filename: str) -> None:
        form = aiohttp.FormData()
        form.add_field(
            "file",
            gcode.encode("utf-8"),
            filename=filename,
            content_type="application/octet-stream",
        )
        form.add_field("select", "true")
        form.add_field("print", "true")

        url = f"{self._base_url}/api/files/local"
        assert self._api_key is not None
        headers: Dict[str, str] = {"X-Api-Key": self._api_key}

        log_data = f"POST {url} with file '{filename}' size {len(gcode)}"
        logger.debug(
            log_data,
            extra={
                "log_category": "RAW_IO",
                "direction": "TX",
                "data": log_data,
            },
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, data=form, headers=headers
                ) as response:
                    if response.status == 403:
                        await self._handle_auth_error()
                        raise DeviceConnectionError(
                            _("Authentication failed during upload.")
                        )
                    if response.status == 409:
                        raise DeviceConnectionError(
                            _(
                                "Printer is busy or not operational. "
                                "Cannot start a new job."
                            )
                        )
                    response.raise_for_status()
                    data = await response.json()

            logger.debug(
                f"Upload response: {data}",
                extra={
                    "log_category": "RAW_IO",
                    "direction": "RX",
                    "data": str(data)[:500].encode("utf-8"),
                },
            )

            if not data.get("effectivePrint", False):
                raise DeviceConnectionError(
                    _(
                        "OctoPrint accepted the file but could not "
                        "start printing. The printer may not be "
                        "operational or is already busy."
                    )
                )

        except aiohttp.ClientError as e:
            msg = _(
                "Could not upload file to OctoPrint at '{host}:{port}'."
            ).format(host=self.host, port=self.port)
            raise DeviceConnectionError(msg) from e

    async def run_raw(self, machine_code: str) -> None:
        lines = [
            line.strip() for line in machine_code.splitlines() if line.strip()
        ]
        if not lines:
            return

        for line in lines:
            logger.info(line, extra=self._log_extra("USER_COMMAND"))

        if len(lines) == 1:
            await self._api_request(
                "POST",
                "/api/printer/command",
                json={"command": lines[0]},
            )
        else:
            await self._api_request(
                "POST",
                "/api/printer/command",
                json={"commands": lines},
            )
        self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        if hold:
            await self._api_request(
                "POST",
                "/api/job",
                json={
                    "command": "pause",
                    "action": "pause",
                },
            )
        else:
            await self._api_request(
                "POST",
                "/api/job",
                json={
                    "command": "pause",
                    "action": "resume",
                },
            )

    async def cancel(self) -> None:
        await self._api_request(
            "POST",
            "/api/job",
            json={"command": "cancel"},
        )
        self._job_active = False
        self._job_done_event.set()
        self.job_finished.send(self)

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def home(self, axes: Optional[Axis] = None) -> None:
        if axes is None:
            axis_list = ["x", "y", "z"]
        else:
            axis_list = []
            for a in Axis:
                if axes & a:
                    assert a.name
                    axis_list.append(a.name.lower())

        await self._api_request(
            "POST",
            "/api/printer/printhead",
            json={"command": "home", "axes": axis_list},
        )

    async def move_to(self, pos_x: float, pos_y: float) -> None:
        await self._api_request(
            "POST",
            "/api/printer/printhead",
            json={
                "command": "jog",
                "x": pos_x,
                "y": pos_y,
                "absolute": True,
            },
        )

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        params: Dict[str, Any] = {
            "command": "jog",
            "speed": speed,
        }
        params.update(deltas)
        await self._api_request(
            "POST",
            "/api/printer/printhead",
            json=params,
        )

    async def select_tool(self, tool_number: int) -> None:
        await self._api_request(
            "POST",
            "/api/printer/command",
            json={"command": f"T{tool_number}"},
        )

    async def read_settings(self) -> None:
        self.settings_read.send(self, settings=[])

    async def write_setting(self, key: str, value: Any) -> None:
        raise NotImplementedError(
            _(
                "OctoPrint does not support writing "
                "device firmware settings through its API."
            )
        )

    def get_setting_vars(self) -> List["VarSet"]:
        return []

    async def clear_alarm(self) -> None:
        await self._api_request(
            "POST",
            "/api/printer/command",
            json={"command": "M999"},
        )
        self.state.error = None
        self.state.status = DeviceStatus.IDLE
        self.state_changed.send(self, state=self.state)

    async def set_power(self, head: "Laser", percent: float) -> None:
        if percent <= 0:
            cmd = "M5"
        else:
            power_abs = percent * head.max_power
            cmd = f"M3 S{power_abs:.0f}"
        await self._api_request(
            "POST",
            "/api/printer/command",
            json={"command": cmd},
        )

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        await self.set_power(head, percent)

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        p_map = {
            "G54": 1,
            "G55": 2,
            "G56": 3,
            "G57": 4,
            "G58": 5,
            "G59": 6,
        }
        p_num = p_map.get(wcs_slot)
        if p_num is None:
            raise ValueError(f"Invalid WCS slot: {wcs_slot}")
        cmd = f"G10 L2 P{p_num} X{x:.3f} Y{y:.3f} Z{z:.3f}"
        await self._api_request(
            "POST",
            "/api/printer/command",
            json={"command": cmd},
        )

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        return {}

    async def run_probe_cycle(
        self,
        axis: Axis,
        max_travel: float,
        feed_rate: int,
    ) -> Optional[Pos]:
        assert axis.name, "Probing requires a named axis."
        axis_letter = axis.name.upper()
        cmd = f"G38.2 {axis_letter}{max_travel:.3f} F{feed_rate}"
        self.probe_status_changed.send(
            self, message=f"Probing {axis_letter}..."
        )
        await self._api_request(
            "POST",
            "/api/printer/command",
            json={"command": cmd},
        )
        self.probe_status_changed.send(
            self,
            message=_(
                "Probe command sent. OctoPrint does not "
                "report probe results via its API."
            ),
        )
        return None

    def _update_command_status(
        self,
        status: TransportStatus,
        message: Optional[str] = None,
    ) -> None:
        log_data = f"Command status: {status.name}"
        if message:
            log_data += f" - {message}"
        logger.info(log_data, extra=self._log_extra("MACHINE_EVENT"))
        self.command_status_changed.send(self, status=status, message=message)

    def _update_connection_status(
        self,
        status: TransportStatus,
        message: Optional[str] = None,
    ) -> None:
        log_data = f"Connection status: {status.name}"
        if message:
            log_data += f" - {message}"
        logger.info(log_data, extra=self._log_extra("MACHINE_EVENT"))
        self.connection_status_changed.send(
            self, status=status, message=message
        )
