from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class State:
    power: float = 0.0
    air_assist: bool = False
    cut_speed: Optional[int] = None
    travel_speed: Optional[int] = None
    active_laser_uid: Optional[str] = None
    frequency: Optional[int] = None
    pulse_width: Optional[float] = None

    def __copy__(self):
        return State(
            power=self.power,
            air_assist=self.air_assist,
            cut_speed=self.cut_speed,
            travel_speed=self.travel_speed,
            active_laser_uid=self.active_laser_uid,
            frequency=self.frequency,
            pulse_width=self.pulse_width,
        )

    def allow_rapid_change(self, target_state: State) -> bool:
        return self.air_assist == target_state.air_assist
