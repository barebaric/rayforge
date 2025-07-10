import re
import asyncio
import aiohttp
from copy import copy
from typing import Callable, Optional
from ..transport import HttpTransport, WebSocketTransport, TransportStatus
from ..opsencoder.gcode import GcodeEncoder
from ..models.ops import Ops
from ..models.machine import Machine
from .driver import Driver, DeviceStatus, DeviceState, Pos


hw_info_url = '/command?plain=%5BESP420%5D&PAGEID='
fw_info_url = '/command?plain=%5BESP800%5D&PAGEID='
eeprom_info_url = '/command?plain=%5BESP400%5D&PAGEID='
command_url = '/command?commandText={command}&PAGEID='
upload_url = '/upload'
upload_list_url = '/upload?path=/&PAGEID=0'
execute_url = '/command?commandText=%5BESP220%5D/{filename}'
status_url = command_url.format(command='?')

pos_re = re.compile(r':(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)')
fs_re = re.compile(r'FS:(\d+),(\d+)')


def _parse_pos_triplet(pos) -> Optional[Pos]:
    match = pos_re.search(pos)
    if not match:
        return None
    pos = tuple(float(i) for i in match.groups())
    if len(pos) != 3:
        return None
    return pos


def _parse_state(
    state_str: str, default: DeviceState, logger: Callable
) -> DeviceState:
    """
    Example state_str:
    Run|MPos:10.0,20.0,0.0|WPos:10.0,20.0,0.0|W0:10.0,20.0,0.0|FS:1500,0

    - First field is always the status.
    - MPos is position in machine coords.
    - WPos is position in work coords.
    - No idea what W0 is.
    - FS: tuple of feed rate and spindle speed

    Also note that not always all fields are included, and sometimes
    others not listed here appear.
    """
    # Split out the status.
    state = copy(default)
    try:
        status, *attribs = state_str.split('|')
        status = status.split(':')[0]
    except ValueError:
        return state

    try:
        state.status = DeviceStatus[status.upper()]
    except KeyError:
        logger(
            message=f"device sent an unupported status: {status}"
        )

    # Parse the substrings.
    for attrib in attribs:
        if attrib.startswith('MPos:'):
            state.machine_pos = _parse_pos_triplet(
                attrib
            ) or state.machine_pos

        elif attrib.startswith('WPos:'):
            state.work_pos = _parse_pos_triplet(attrib) or state.work_pos

        elif attrib.startswith('FS:'):
            try:
                match = fs_re.match(attrib)
                if not match:
                    continue
                fs = [int(i) for i in match.groups()]
                state.feed_rate = int(fs[0])
                # We ignore fs[1] (="spindle speed")
            except (ValueError, IndexError):
                pass

        else:
            pass  # Ignore everything else

    return state


class GrblDriver(Driver):
    """
    Handles GRBL based devices via HTTP+WebSocket
    """
    label = "GRBL"
    subtitle = 'Send GRBL-compatible Gcode via network connection'

    def __init__(self):
        super().__init__()
        self.encoder = GcodeEncoder()
        self.http = None
        self.websocket = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None

    def setup(self, host: str = ""):
        assert not self.did_setup
        if not host:
            return  # Leave unconfigured
        super().setup()

        # Initialize transports
        self.http_base = f'http://{host}'
        self.http = HttpTransport(
            f'{self.http_base}{status_url}',
            receive_interval=.5
        )
        self.http.received.connect(self.on_http_data_received)
        self.http.status_changed.connect(self.on_http_status_changed)

        self.websocket = WebSocketTransport(
            f'ws://{host}:81/',
            self.http_base
        )
        self.websocket.received.connect(self.on_websocket_data_received)
        self.websocket.status_changed.connect(self.on_websocket_status_changed)

    async def cleanup(self):
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
        if self.http:
            await self.http.disconnect()
            self.http.received.disconnect(self.on_http_data_received)
            self.http.status_changed.disconnect(self.on_http_status_changed)
            self.http = None
        await super().cleanup()

    async def _get_hardware_info(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.http_base}{hw_info_url}"
            ) as response:
                data = await response.text()
        return data

    async def _get_firmware_info(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.http_base}{fw_info_url}"
            ) as response:
                data = await response.text()
        return data

    async def _get_eeprom_info(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.http_base}{eeprom_info_url}"
            ) as response:
                data = await response.text()
        return data

    async def _send_command(self, command):
        async with aiohttp.ClientSession() as session:
            url = command_url.format(command=command)
            async with session.get(
                f"{self.http_base}{url}"
            ) as response:
                data = await response.text()
        return data

    async def _upload(self, gcode, filename):
        form = aiohttp.FormData([])
        form.add_field('path', '/')
        form.add_field(f'/{filename}S', str(len(gcode)))
        form.add_field('myfile[]', gcode, filename=filename)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.http_base}{upload_url}",
                data=form
            ) as response:
                data = await response.text()
        return data

    async def _execute(self, filename):
        async with aiohttp.ClientSession() as session:
            url = execute_url.format(filename=filename)
            async with session.get(f"{self.http_base}{url}") as response:
                data = await response.text()
        await session.close()
        return data

    async def connect(self):
        self.keep_running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        while self.keep_running:
            self._on_connection_status_changed(TransportStatus.CONNECTING)
            try:
                hw_info = await self._get_hardware_info()
                self._log(hw_info)
                fw_info = await self._get_firmware_info()
                self._log(fw_info)
                eeprom_info = await self._get_eeprom_info()
                self._log(eeprom_info)

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.http.connect())
                    tg.create_task(self.websocket.connect())
            except Exception as e:
                self._on_connection_status_changed(
                    TransportStatus.ERROR,
                    str(e)
                )
            finally:
                if self.websocket:
                    await self.websocket.disconnect()
                if self.http:
                    await self.http.disconnect()

            self._on_connection_status_changed(TransportStatus.SLEEPING)
            await asyncio.sleep(5)

    async def run(self, ops: Ops, machine: Machine) -> None:
        gcode = self.encoder.encode(ops, machine)

        try:
            await self._upload(gcode, 'rayforge.gcode')
            await self._execute('rayforge.gcode')
        except Exception as e:
            self._on_connection_status_changed(
                TransportStatus.ERROR,
                str(e)
            )
            raise

    async def set_hold(self, hold: bool = True) -> None:
        if hold:
            await self._send_command('!')
        else:
            await self._send_command('~')

    async def cancel(self) -> None:
        await self._send_command('%18')

    async def home(self) -> None:
        await self._send_command('$H')

    async def move_to(self, pos_x, pos_y) -> None:
        cmd = f"$J=G90 G21 F1500 X{float(pos_x)} Y{float(pos_y)}"
        await self._send_command(cmd)

    def on_http_data_received(self, sender, data: bytes):
        pass

    def on_http_status_changed(self,
                               sender,
                               status: TransportStatus,
                               message: Optional[str] = None):
        self._on_command_status_changed(status, message)

    def on_websocket_data_received(self, sender, data: bytes):
        data_str = data.decode('utf-8')
        for line in data_str.splitlines():
            self._log(line)
            if not line.startswith('<') or not line.endswith('>'):
                continue
            state = _parse_state(line[1:-1], self.state, self._log)
            if state != self.state:
                self.state = state
                self._on_state_changed()

    def on_websocket_status_changed(self,
                                    sender,
                                    status: TransportStatus,
                                    message: Optional[str] = None):
        self._on_connection_status_changed(status, message)
