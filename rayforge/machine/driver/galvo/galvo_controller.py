"""
EzCad2/LMC Galvo Controller.
"""

import asyncio
import logging
import struct
from typing import Optional

from .galvo_connection import GalvoConnection
from .galvo_consts import (
    LIST_BUFFER_SIZE,
    COMMAND_SIZE,
    NOP_COMMAND,
    SOURCE_CO2,
    SOURCE_FIBER,
    STATUS_BUSY,
    STATUS_READY,
    listDelayTime,
    listEndOfList,
    listJumpDelay,
    listJumpSpeed,
    listJumpTo,
    listLaserOffDelay,
    listLaserOnDelay,
    listLaserOnPoint,
    listMarkCurrent,
    listMarkFreq,
    listMarkPowerRatio,
    listMarkSpeed,
    listMarkTo,
    listPolygonDelay,
    listQSwitchPeriod,
    listReadyMark,
    listSetCo2FPK,
    listWritePort,
    listFiberYLPMPulseWidth,
    EnableLaser,
    ExecuteList,
    SetPwmPulseWidth,
    GetVersion,
    GotoXY,
    ResetList,
    SetControlMode,
    SetDelayMode,
    SetFirstPulseKiller,
    SetLaserMode,
    SetTiming,
    SetStandby,
    SetPwmHalfPeriod,
    WritePort,
    SetEndOfList,
    EnableZ,
    SetFpkParam2,
    SetFlyRes,
    Fiber_SetMo,
    WriteAnalogPort1,
)
from .galvo_protocol import GalvoResponse

logger = logging.getLogger(__name__)

DRIVER_STATE_IDLE = 0
DRIVER_STATE_MARKING = 1
DRIVER_STATE_LIGHTING = 2


class GalvoController:
    """
    Manages an EzCad2/LMC galvo controller board.
    """

    def __init__(
        self,
        connection: Optional[GalvoConnection] = None,
        source: str = SOURCE_FIBER,
        galvos_per_mm: int = 500,
    ):
        self._connection: Optional[GalvoConnection] = connection
        self._source = source
        self._galvos_per_mm = galvos_per_mm
        self._list_lock = asyncio.Lock()

        self._state = DRIVER_STATE_IDLE
        self._active_list: Optional[bytearray] = None
        self._active_index: int = 0
        self._num_list_packets: int = 0
        self._list_executing: bool = False

        self._last_x: int = 0x8000
        self._last_y: int = 0x8000

        self._speed: Optional[float] = None
        self._travel_speed: Optional[float] = None
        self._frequency: Optional[float] = None
        self._power: Optional[float] = None
        self._fpk: Optional[float] = None
        self._pulse_width: Optional[int] = None

        self._delay_on: Optional[float] = None
        self._delay_off: Optional[float] = None
        self._delay_poly: Optional[float] = None

        self._port_bits: int = 0
        self._laser_pin: int = 0

        self.first_pulse_killer: int = 200
        self.pwm_pulse_width: int = 125
        self.pwm_half_period: int = 125
        self.standby_p1: int = 2000
        self.standby_p2: int = 20
        self.timing_mode: int = 1
        self.delay_mode: int = 1
        self.laser_mode: int = 1
        self.control_mode: int = 0
        self.fpk_max_voltage: int = 0xFFB
        self.fpk_min_voltage: int = 1
        self.fpk_t1: int = 409
        self.fpk_t2: int = 100
        self.fly_resolution_1: int = 0
        self.fly_resolution_2: int = 99
        self.fly_resolution_3: int = 1000
        self.fly_resolution_4: int = 25

        self._initialized: bool = False

    @property
    def is_connected(self) -> bool:
        return self._connection is not None and self._connection.is_connected

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        self._source = value

    @property
    def state(self) -> int:
        return self._state

    async def connect(self) -> None:
        if self._connection is None:
            from .galvo_mock_connection import MockGalvoConnection

            self._connection = MockGalvoConnection()
        if not self._connection.is_connected:
            self._connection.open()
            self._initialize()

    async def disconnect(self) -> None:
        if self._connection and self._connection.is_connected:
            self._connection.close()
        self._initialized = False

    def _initialize(self) -> None:
        self._send_single(EnableLaser)
        self._send_single(SetControlMode, self.control_mode)
        self._send_single(SetLaserMode, self.laser_mode)
        self._send_single(SetDelayMode, self.delay_mode)
        self._send_single(SetTiming, self.timing_mode)
        self._send_single(SetStandby, self.standby_p1, self.standby_p2)
        self._send_single(SetFirstPulseKiller, self.first_pulse_killer)
        self._send_single(SetPwmHalfPeriod, self.pwm_half_period)
        self._send_single(SetPwmPulseWidth, self.pwm_pulse_width)
        if self._source == SOURCE_FIBER:
            self._send_single(Fiber_SetMo, 0)
        self._send_single(
            SetFpkParam2,
            self.fpk_max_voltage,
            self.fpk_min_voltage,
            self.fpk_t1,
            self.fpk_t2,
        )
        self._send_single(
            SetFlyRes,
            self.fly_resolution_1,
            self.fly_resolution_2,
            self.fly_resolution_3,
            self.fly_resolution_4,
        )
        self._send_single(EnableZ)
        self._send_single(WriteAnalogPort1, 0x7FF)
        self._initialized = True

    async def enter_marking_mode(self) -> None:
        if self._state == DRIVER_STATE_MARKING:
            return
        self._state = DRIVER_STATE_MARKING
        self._reset_list()
        self._port_on(self._laser_pin)
        self._list_write_port()
        if self._source == SOURCE_FIBER:
            self._send_single(Fiber_SetMo, 1)
        self._reset_cached_params()
        self._list_write(listReadyMark)

    async def enter_idle_mode(self) -> None:
        if self._state == DRIVER_STATE_IDLE:
            return
        self._end_list()
        if not self._list_executing and self._num_list_packets:
            self._execute_list()
        self._list_executing = False
        self._num_list_packets = 0
        await self._wait_idle()
        if self._source == SOURCE_FIBER:
            self._send_single(Fiber_SetMo, 0)
        self._port_off(self._laser_pin)
        self._send_single(WritePort, self._port_bits)
        self._state = DRIVER_STATE_IDLE

    def _reset_cached_params(self) -> None:
        self._speed = None
        self._travel_speed = None
        self._frequency = None
        self._power = None
        self._fpk = None
        self._pulse_width = None
        self._delay_on = None
        self._delay_off = None
        self._delay_poly = None

    def _reset_list(self) -> None:
        self._active_list = None
        self._active_index = 0
        self._send_single(ResetList)

    def _end_list(self) -> None:
        if self._active_list is None or self._active_index == 0:
            return
        if self._connection is None:
            return
        self._list_write(listEndOfList)
        packet = bytes(self._active_list[: self._active_index])
        padded = bytearray(packet)
        remaining = LIST_BUFFER_SIZE - len(padded)
        padded.extend(NOP_COMMAND * (remaining // COMMAND_SIZE))
        self._connection.write(packet=bytes(padded))
        self._send_single(SetEndOfList, 0)
        self._num_list_packets += 1
        self._active_list = None
        self._active_index = 0
        if self._num_list_packets > 2 and not self._list_executing:
            self._execute_list()
            self._list_executing = True

    def _list_write(
        self,
        command: int,
        v1: int = 0,
        v2: int = 0,
        v3: int = 0,
        v4: int = 0,
        v5: int = 0,
    ) -> None:
        if self._active_index >= LIST_BUFFER_SIZE:
            return
        if self._active_list is None:
            self._active_list = bytearray(LIST_BUFFER_SIZE)
            self._active_index = 0
        struct.pack_into(
            "<6H",
            self._active_list,
            self._active_index,
            command,
            v1,
            v2,
            v3,
            v4,
            v5,
        )
        self._active_index += COMMAND_SIZE

    def _send_single(
        self,
        command: int,
        v1: int = 0,
        v2: int = 0,
        v3: int = 0,
        v4: int = 0,
        v5: int = 0,
    ) -> "GalvoResponse":
        if self._connection is None:
            return GalvoResponse(0, 0, 0, 0)
        cmd = struct.pack("<6H", command, v1, v2, v3, v4, v5)
        self._connection.write(packet=cmd)
        resp = self._connection.read()
        return GalvoResponse.from_bytes(resp)

    def _execute_list(self) -> None:
        self._send_single(ExecuteList)

    def _list_write_port(self) -> None:
        self._list_write(listWritePort, self._port_bits)

    def _port_on(self, bit: int) -> None:
        self._port_bits |= 1 << bit

    def _port_off(self, bit: int) -> None:
        self._port_bits &= ~(1 << bit)

    def _get_status(self) -> int:
        resp = self._send_single(GetVersion)
        return resp.v3

    def _is_busy(self) -> bool:
        return bool(self._get_status() & STATUS_BUSY)

    def _is_ready(self) -> bool:
        return bool(self._get_status() & STATUS_READY)

    async def _wait_idle(self) -> None:
        while self._is_busy():
            await asyncio.sleep(0.01)

    async def _wait_ready(self) -> None:
        while not self._is_ready():
            await asyncio.sleep(0.01)

    def goto(self, x: int, y: int) -> None:
        if x == self._last_x and y == self._last_y:
            return
        x = max(0, min(0xFFFF, x))
        y = max(0, min(0xFFFF, y))
        distance = int(
            abs(complex(x, y) - complex(self._last_x, self._last_y))
        )
        if distance > 0xFFFF:
            distance = 0xFFFF
        self._list_write(listJumpTo, x, y, 0, distance)
        self._last_x = x
        self._last_y = y

    def mark(self, x: int, y: int) -> None:
        if x == self._last_x and y == self._last_y:
            return
        x = max(0, min(0xFFFF, x))
        y = max(0, min(0xFFFF, y))
        distance = int(
            abs(complex(x, y) - complex(self._last_x, self._last_y))
        )
        if distance > 0xFFFF:
            distance = 0xFFFF
        self._list_write(listMarkTo, x, y, 0, distance)
        self._last_x = x
        self._last_y = y

    def goto_xy(
        self, x: int, y: int, angle: int = 0, distance: int = 0
    ) -> None:
        self._last_x = x
        self._last_y = y
        self._send_single(GotoXY, x, y, angle, distance)

    def dwell(self, time_ms: int) -> None:
        dwell_time = time_ms * 100
        while dwell_time > 0:
            d = min(dwell_time, 60000)
            self._list_write(listLaserOnPoint, d)
            dwell_time -= d

    def wait(self, time_ms: int) -> None:
        wait_time = time_ms * 100
        while wait_time > 0:
            d = min(wait_time, 60000)
            self._list_write(listDelayTime, d)
            wait_time -= d

    def set_mark_speed(self, speed: float) -> None:
        speed_val = int(self._convert_speed(speed))
        if speed_val > 0xFFFF:
            speed_val = 0xFFFF
        self._list_write(listMarkSpeed, speed_val)

    def set_travel_speed(self, speed: float) -> None:
        if self._travel_speed == speed:
            return
        self._travel_speed = speed
        speed_val = int(self._convert_speed(speed))
        if speed_val > 0xFFFF:
            speed_val = 0xFFFF
        self._list_write(listJumpSpeed, speed_val)

    def set_laser_on_delay(self, delay: float) -> None:
        if self._delay_on == delay:
            return
        self._delay_on = delay
        sign = 0x0000 if delay >= 0 else 0x8000
        self._list_write(listLaserOnDelay, abs(int(delay)), sign)

    def set_laser_off_delay(self, delay: float) -> None:
        if self._delay_off == delay:
            return
        self._delay_off = delay
        sign = 0x0000 if delay >= 0 else 0x8000
        self._list_write(listLaserOffDelay, abs(int(delay)), sign)

    def set_jump_delay(self, delay: float) -> None:
        if self._delay_on == delay:
            return
        self._delay_on = delay
        sign = 0x0000 if delay >= 0 else 0x8000
        self._list_write(listJumpDelay, abs(int(delay)), sign)

    def set_polygon_delay(self, delay: float) -> None:
        if self._delay_poly == delay:
            return
        self._delay_poly = delay
        val = int(min(65535, max(0, delay / 10.0)))
        self._list_write(listPolygonDelay, val)

    def set_power(self, power_percent: float) -> None:
        if self._power == power_percent:
            return
        self._power = power_percent
        if self._source == SOURCE_FIBER:
            current = self._convert_power(power_percent)
            self._list_write(listMarkCurrent, current)
        elif self._source == SOURCE_CO2:
            freq = self._frequency or 1.0
            ratio = int(round(200 * power_percent / freq))
            self._list_write(listMarkPowerRatio, ratio)

    def set_frequency(self, freq_khz: float) -> None:
        if self._frequency == freq_khz:
            return
        self._frequency = freq_khz
        if self._source == SOURCE_FIBER:
            period = self._convert_frequency(freq_khz, base=20000.0)
            self._list_write(listQSwitchPeriod, period)
        elif self._source == SOURCE_CO2:
            period = self._convert_frequency(freq_khz, base=10000.0)
            self._list_write(listMarkFreq, period)

    def set_fpk(self, fpk: float) -> None:
        if self._source not in (SOURCE_CO2,):
            return
        if self._fpk == fpk:
            return
        self._fpk = fpk
        val = int(round(2000.0 / (self._frequency or 1.0)))
        self._list_write(listSetCo2FPK, val, val)

    def set_pulse_width(self, pulse_width: int) -> None:
        if self._pulse_width == pulse_width:
            return
        self._pulse_width = pulse_width
        self._list_write(listFiberYLPMPulseWidth, pulse_width)

    def _convert_speed(self, speed_mm_s: float) -> int:
        return int(speed_mm_s * self._galvos_per_mm / 1000.0)

    def _convert_frequency(
        self, freq_khz: float, base: float = 20000.0
    ) -> int:
        return int(round(base / freq_khz)) & 0xFFFF

    def _convert_power(self, power_percent: float) -> int:
        return int(round(power_percent * 0xFFF / 100.0))
