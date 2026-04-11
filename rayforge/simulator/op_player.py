from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..core.layer import Layer
from ..core.ops import Ops
from ..core.ops.axis import Axis
from ..core.ops.commands import LayerStartCommand, MovingCommand
from ..machine.models.rotary_module import RotaryMode
from .machine_state import MachineState

if TYPE_CHECKING:
    from ..core.doc import Doc
    from ..machine.models.machine import Machine


class OpPlayer:
    def __init__(self, ops: Ops, machine: Machine, doc: Doc):
        if not ops or ops.is_empty():
            raise ValueError("OpPlayer requires a non-empty Ops")
        self.ops = ops
        self._machine = machine
        self._doc = doc
        self._current_index: int = -1
        self._current_rotary_axis: Optional[Axis] = None
        self._source_axis: Axis = Axis.Y
        self._mode: RotaryMode = RotaryMode.TRUE_4TH_AXIS
        self._mm_per_rotation: float = 0.0
        self._diameter: float = 0.0
        self.state = self._create_home_state()

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def source_axis(self) -> Axis:
        return self._source_axis

    def _create_home_state(self) -> MachineState:
        state = MachineState.from_axis_set(self._machine.axes)
        space = self._machine.get_coordinate_space()
        home_x, home_y = space.machine_point_to_world(0.0, 0.0)
        state.axes[Axis.X] = home_x
        state.axes[Axis.Y] = home_y
        return state

    def _update_rotary_config(self, layer_uid: str) -> None:
        item = self._doc.find_descendant_by_uid(layer_uid)
        if not isinstance(item, Layer):
            self._current_rotary_axis = None
            self._source_axis = Axis.Y
            self._mode = RotaryMode.TRUE_4TH_AXIS
            self._mm_per_rotation = 0.0
            self._diameter = 0.0
            return
        self._diameter = item.rotary_diameter
        module = (
            self._machine.rotary_modules.get(item.rotary_module_uid)
            if item.rotary_module_uid
            else None
        )
        if module:
            self._source_axis = module.source_axis
            self._mode = module.mode
            self._mm_per_rotation = module.mm_per_rotation
            if module.mode == RotaryMode.TRUE_4TH_AXIS:
                self._current_rotary_axis = module.axis
            else:
                self._current_rotary_axis = None
        else:
            self._source_axis = Axis.Y
            self._current_rotary_axis = (
                self._machine.get_rotary_axis_for_layer(item)
            )
            self._mode = RotaryMode.TRUE_4TH_AXIS
            self._mm_per_rotation = 0.0

    def _apply_rotary_mapping(self) -> None:
        if self._mode == RotaryMode.TRUE_4TH_AXIS:
            ra = self._current_rotary_axis
            if ra is None:
                return
            value = self.state.axes.get(self._source_axis, 0.0)
            self.state.axes[ra] = value
        elif self._mode == RotaryMode.PASSTHROUGH:
            if self._mm_per_rotation > 0 and self._diameter > 0:
                fw_val = self.state.axes.get(self._source_axis, 0.0)
                mm_val = (
                    fw_val * math.pi * self._diameter / self._mm_per_rotation
                )
                self.state.axes[self._source_axis] = mm_val

    def seek(self, index: int):
        self.state = self._create_home_state()
        self._current_index = -1
        self._current_rotary_axis = None
        self._source_axis = Axis.Y
        self._mode = RotaryMode.TRUE_4TH_AXIS
        self._mm_per_rotation = 0.0
        self._diameter = 0.0
        self.advance_to(index)

    def advance_to(self, index: int):
        if index < self._current_index:
            raise ValueError(
                f"Cannot advance backwards: current="
                f"{self._current_index}, requested={index}. "
                f"Use seek() instead."
            )
        if index >= len(self.ops):
            raise IndexError(
                f"Index {index} out of range "
                f"(ops has {len(self.ops)} commands)"
            )
        for i in range(self._current_index + 1, index + 1):
            cmd = self.ops.commands[i]
            if isinstance(cmd, LayerStartCommand):
                self._update_rotary_config(cmd.layer_uid)
            self.state.apply_command(cmd, i)
            if isinstance(cmd, MovingCommand):
                self._apply_rotary_mapping()
        self._current_index = index

    def seek_last_movement(self) -> Optional[int]:
        last = None
        for i, cmd in enumerate(self.ops):
            if isinstance(cmd, MovingCommand):
                last = i
        if last is not None:
            self.seek(last)
        return last
