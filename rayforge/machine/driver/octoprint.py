import asyncio
import inspect
import json
import logging
import math
from typing import (
    Optional,
    cast,
    Any,
    TYPE_CHECKING,
    List,
    Callable,
    Union,
    Awaitable,
    Dict,
)

from gettext import gettext as _

import aiohttp

from rayforge.core.varset.octoprintauthflowvar import OctoprintAuthFlowVar
from rayforge.machine.transport.validators import is_valid_hostname_or_ip
from rayforge.machine.transport.websocket import WebSocketTransport
from ...context import RayforgeContext
from ...core.varset import VarSet
from ...pipeline.encoder.base import OpsEncoder, MachineCodeOpMap
from ...pipeline.encoder.gcode import GcodeEncoder
from ..transport import TransportStatus
from .driver import (
    DeviceStatus,
    Driver,
    Axis,
    DriverPrecheckError,
    Pos,
)
from .octoprint_api import (
    JobInfos,
    Progress,
    SocketMessagesTypes,
    ConnectionInfos,
    resolve_status,
)

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ..models.machine import Machine
    from ..models.laser import Laser


logger = logging.getLogger(__name__)


class OctoprintDriver(Driver):
    label = _("Octoprint")
    subtitle = _("Connect to an Octoprint device over the network")
    supports_settings = True
    reports_granular_progress = True

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self.host = None
        self.port = None
        self.base_url = None
        self.token = None
        self.http_session = None
        self.websocket = None
        self.connecting = False
        self.connection_flags = []
        self.progress_notifier = None
        self.last_op_index = -1
        self.op_n = 0
        self._connection_task: Optional[asyncio.Task] = None
        self._cmd_lock = asyncio.Lock()

        self._offsets: Dict[str, Pos] = cast(
            Dict[str, Pos], machine.wcs_offsets.copy()
        )

        # Ensure standard keys exist
        defaults = ["G54", "G55", "G56", "G57", "G58", "G59"]
        for key in defaults:
            if key not in self._offsets:
                self._offsets[key] = (0.0, 0.0, 0.0)

    @property
    def resource_uri(self) -> Optional[str]:
        if self.host:
            return f"tcp://{self.host}:{self.port}"
        return None

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                OctoprintAuthFlowVar(
                    key="authorize_button",
                    label=_("Authorize"),
                    description=_("Click to start the authorization process"),
                ),
            ]
        )

    def setup_button_callbacks(self) -> None:
        """Wires up instance-specific button callbacks after instantiation."""
        # This is called after the driver is instantiated, so we have self
        pass

    def get_setting_vars(self) -> List["VarSet"]:
        return [VarSet(title=_("No settings"))]

    @property
    def machine_space_wcs(self) -> str:
        """
        Returns the machine space coordinate system identifier.
        This is an immutable coordinate system with zero offset.
        """
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        """
        Returns a human-readable display name for the machine space
        coordinate system.
        """
        return _("Machine Coordinates")

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        token, host, port = OctoprintAuthFlowVar.parse_value(
            cast(str, kwargs.get("authorize_button", ""))
        )
        if not is_valid_hostname_or_ip(host):
            raise DriverPrecheckError(
                _("Invalid hostname or IP address: '{host}'").format(host=host)
            )

    def _setup_implementation(self, **kwargs: Any) -> None:
        token, host, port = OctoprintAuthFlowVar.parse_value(
            cast(str, kwargs.get("authorize_button", ""))
        )
        self.token = token
        self.host = host
        self.port = port

        self.base_url = f"http://{self.host}:{self.port}/"
        self.socket_url = f"ws://{self.host}:{self.port}/sockjs/websocket"
        
        self.websocket = WebSocketTransport(self.socket_url)
        self.websocket.status_changed.connect(self.on_websocket_status_changed)
        self.websocket.received.connect(self.on_websocket_data_received)

    def on_websocket_data_received(self, sender, raw_data: bytes):
        try:
            data_str = raw_data.decode("utf-8").strip()
        except UnicodeDecodeError:
            logger.warning(
                f"Received non-UTF8 data on WebSocket: {raw_data!r}"
            )
            return
        logger.debug(f"WebSocket data received: {data_str}")
        data: Dict[str, Any] = json.loads(raw_data)
        msg_type = list(data.keys())[0]
        data = data[msg_type]
        match msg_type:
            case SocketMessagesTypes.CONNECTED:
                pass
            case SocketMessagesTypes.HISTORY:
                # TODO: Fetch profiles to get printer name
                active_flags = [
                    flag
                    for flag, active in data["state"]["flags"].items()
                    if active
                ]
                self.state.status = resolve_status(active_flags)
                self.state_changed.send(self, state=self.state)
                self.connection_flags = data["state"]["flags"]
            case SocketMessagesTypes.CURRENT:
                state = data["state"]
                if set(state["flags"]) != set(self.connection_flags):
                    active_flags = [
                        flag
                        for flag, active in state["flags"].items()
                        if active
                    ]
                    self.state.status = resolve_status(active_flags)
                    self.state_changed.send(self, state=self.state)
                    self.connection_flags = state["flags"]
                if data.get("job"):
                    current_job = JobInfos.from_dict(data["job"])
                    pass
                if data.get("progress"):
                    progress = Progress.from_dict(data["progress"])
                    if self.progress_notifier:
                        op_i = round(self.op_n * progress.completion)
                        if self.last_op_index != op_i:
                            self.last_op_index = op_i
                            self.progress_notifier(self.last_op_index)

    def on_websocket_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        self.connection_status_changed.send(
            self, status=status, message=message
        )
        if status == TransportStatus.CONNECTED:
            if not self.http_session:
                return
            self._connection_task = asyncio.create_task(self._on_web_socket_open())
            

    @classmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        """Returns a GcodeEncoder configured for the machine's dialect."""
        return GcodeEncoder(machine.dialect)

    async def _connect_implementation(self) -> None:
        # Simulate connection sequence
        if not self.base_url or not self.token:
            self.connection_status_changed.send(
                self,
                status=TransportStatus.ERROR,
                message=_("Missing connection parameters"),
            )
            return
        assert self.websocket != None
        self.connecting = True
        websocket = self.websocket
        self.connection_status_changed.send(
            self, status=TransportStatus.CONNECTING
        )
        self.http_session = aiohttp.ClientSession(
            base_url=self.base_url, headers={"X-Api-Key": self.token}
        )
        async with self.http_session.get("/api/version") as response:
            if response.status != 200:
                self.connection_status_changed.send(
                    self,
                    status=TransportStatus.ERROR,
                    message=_("Failed to connect to Octoprint API. Is CORS Enabled ?"),
                )
                return
        await websocket.connect()
        self.state.status = DeviceStatus.UNKNOWN
        self.state_changed.send(self, state=self.state)

        # Upon connect, broadcast WCS state (matches loaded machine state)
        self.wcs_updated.send(self, offsets=self._offsets)

    async def _on_web_socket_open(self) -> None:
        assert self.http_session != None and self.websocket != None
        websocket = self.websocket
        async with self.http_session.post(
            "/api/login",
            json={"passive": True},
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                self.connection_status_changed.send(
                    self,
                    status=TransportStatus.ERROR,
                    message=_("Failed to authenticate with Octoprint API"),
                )
                return
            data = await response.json()
            usr_name = data.get("name")
            session_id = data.get("session")
            subscribe_payload = {
                "subscribe": {
                    "state": {"logs": True, "messages": False},
                    "event": True,
                    "plugins": False,
                }
            }
            while not websocket.is_connected:
                await asyncio.sleep(1)
            await websocket.send(
                json.dumps(subscribe_payload).encode("utf-8")
            )
            auth_payload = {"auth": f"{usr_name}:{session_id}"}
            await websocket.send(json.dumps(auth_payload).encode("utf-8"))
        if self.connecting:
            self.connection_status_changed.send(
                self, status=TransportStatus.CONNECTED
            )
            self.connecting = False
    async def run(
        self,
        machine_code: Any,
        op_map: "MachineCodeOpMap",
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        """
        Dummy implementation that simulates command execution.

        This implementation iterates through the ops defined in op_map and
        simulates execution by calling the on_command_done callback for each
        command with a small delay.
        """
        # We assume ops are indexed 0..N-1.
        num_ops = 0
        if op_map and op_map.op_to_machine_code:
            num_ops = max(op_map.op_to_machine_code.keys()) + 1

        # Simulate command execution with delays
        for op_index in range(num_ops):
            # Small delay to simulate execution time
            await asyncio.sleep(0.01)

            # Call the callback if provided, awaiting it if it's a coroutine
            if on_command_done is not None:
                try:
                    result = on_command_done(op_index)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    # Don't let callback exceptions stop execution
                    pass
        self.job_finished.send(self)

    async def run_raw(self, gcode: str) -> None:
        """
        Dummy implementation that simulates raw G-code execution.
        """
        lines = [line.strip() for line in gcode.splitlines() if line.strip()]
        for line in lines:
            logger.info(line, extra=self._log_extra("USER_COMMAND"))
            await asyncio.sleep(0.01)
        self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        pass

    async def cancel(self) -> None:
        pass

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        """Dummy driver supports homing for all axes."""
        return True

    async def home(self, axes: Optional[Axis] = None) -> None:
        pass

    async def move_to(self, pos_x, pos_y) -> None:
        pass

    async def select_tool(self, tool_number: int) -> None:
        pass

    async def read_settings(self) -> None:
        pass

    async def write_setting(self, key: str, value: Any) -> None:
        pass

    async def clear_alarm(self) -> None:
        pass

    async def set_power(self, head: "Laser", percent: float) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control.
            percent: Power percentage (0.0-1.0). 0 disables power.
        """
        # Dummy driver doesn't control any hardware, so just log the call
        logger.info(
            f"set_power called with head {head.uid} at {percent * 100:.1f}%",
            extra={"log_category": "DRIVER_CMD"},
        )

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """Dummy driver supports jogging for all axes."""
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        pass

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        """Dummy implementation, updates internal state."""
        self._offsets[wcs_slot] = (x, y, z)
        # Notify machine that the driver updated offsets
        self.wcs_updated.send(self, offsets=self._offsets)

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        """Dummy implementation, returns internal state."""
        self.wcs_updated.send(self, offsets=self._offsets)
        return self._offsets

    async def read_parser_state(self) -> Optional[str]:
        """
        Simulate reading the active WCS state.
        Returns the machine's active WCS to treat the client's selection
        as the source of truth for the dummy driver.
        """
        return self._machine.active_wcs

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        """
        Dummy implementation, simulates a successful probe after a short delay.
        """
        self.probe_status_changed.send(
            self, message=f"Simulating probe cycle for axis {axis.name}..."
        )
        await asyncio.sleep(0.5)
        # Simulate a successful probe at a fixed position
        simulated_pos = (10.0, 15.0, -1.0)
        self.probe_status_changed.send(
            self, message=f"Probe triggered at {simulated_pos}"
        )
        return simulated_pos

    async def cleanup(self):
        logger.info("Cleaning up OctoprintDriver resources...")
        self.keep_running = False
        if self._connection_task:
            self._connection_task.cancel()
        if self.websocket:
            await self.websocket.disconnect()
            self.websocket.received.disconnect(self.on_websocket_data_received)
            self.websocket.status_changed.disconnect(
                self.on_websocket_status_changed
            )
            self.websocket = None
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
        await super().cleanup()
