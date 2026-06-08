from bisect import bisect_right
from typing import List, Optional, Tuple

from blinker import Signal
from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.types import CommandCategory, CommandType

from ..core.doc import Doc
from ..core.layer import Layer
from ..machine.models.machine import Machine
from ..machine.models.rotary_module import RotaryMode
from .machine_state import MachineState

_SNAPSHOT_INTERVAL = 1000


class OpPlayer:
    def __init__(self, ops: Ops, machine: Machine, doc: Doc):
        if not ops or ops.is_empty():
            raise ValueError("OpPlayer requires a non-empty Ops")
        self.ops = ops
        self._machine = machine
        self._doc = doc
        self._current_index: int = -1
        self._source_axis: Axis = Axis.Y
        self._rotary_axis: Optional[Axis] = None
        self._prev_layer_uid: Optional[str] = None
        self.state = self._create_home_state()
        self.layer_changed = Signal()
        self._snapshots: List[Tuple[
            int, MachineState, Axis, Optional[Axis]
        ]] = []
        self._build_snapshots()

    def _build_snapshots(self):
        n = self.ops.len()
        if n <= _SNAPSHOT_INTERVAL:
            return
        temp_player = _SnapshotBuilder(
            self.ops, self._machine, self._doc, self._create_home_state()
        )
        interval = _SNAPSHOT_INTERVAL
        for target in range(interval, n, interval):
            temp_player.advance_to(target - 1)
            self._snapshots.append((
                target,
                temp_player.state.copy(),
                temp_player._source_axis,
                temp_player._rotary_axis,
            ))

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def source_axis(self) -> Axis:
        return self._source_axis

    @property
    def rotary_axis(self) -> Optional[Axis]:
        return self._rotary_axis

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
            self._source_axis = Axis.Y
            self._rotary_axis = None
            return
        module = (
            self._machine.rotary_modules.get(item.rotary_module_uid)
            if item.rotary_module_uid
            else None
        )
        if module:
            self._source_axis = Axis.Y
            if module.mode == RotaryMode.TRUE_4TH_AXIS:
                self._rotary_axis = module.axis
            else:
                self._rotary_axis = Axis.Y
        else:
            self._source_axis = Axis.Y
            self._rotary_axis = self._machine.get_rotary_axis_for_layer(item)

    def seek(self, index: int):
        if index >= self.ops.len():
            raise IndexError(
                f"Index {index} out of range "
                f"(ops has {self.ops.len()} commands)"
            )
        if index < 0:
            index = 0

        snapshot_idx = self._find_snapshot(index)
        if snapshot_idx is not None:
            snap_index, snap_state, snap_source, snap_rotary = (
                self._snapshots[snapshot_idx]
            )
            self.state = snap_state.copy()
            self._current_index = snap_index - 1
            self._source_axis = snap_source
            self._rotary_axis = snap_rotary
        else:
            self.state = self._create_home_state()
            self._current_index = -1
            self._source_axis = Axis.Y
            self._rotary_axis = None

        self._prev_layer_uid = None
        self.advance_to(index)
        self._emit_layer_change()

    def _find_snapshot(self, index: int) -> Optional[int]:
        if not self._snapshots:
            return None
        positions = [s[0] for s in self._snapshots]
        pos = bisect_right(positions, index)
        if pos == 0:
            return None
        return pos - 1

    def advance_to(self, index: int):
        if index < self._current_index:
            raise ValueError(
                f"Cannot advance backwards: current="
                f"{self._current_index}, requested={index}. "
                f"Use seek() instead."
            )
        if index >= self.ops.len():
            raise IndexError(
                f"Index {index} out of range "
                f"(ops has {self.ops.len()} commands)"
            )
        for i in range(self._current_index + 1, index + 1):
            ct = self.ops.command_type(i)
            if ct == CommandType.LAYER_START:
                self._update_rotary_config(self.ops.layer_uid(i))
            self.state.apply_command(self.ops, i)
        self._current_index = index

    def seek_last_movement(self) -> Optional[int]:
        last = None
        for i in range(self.ops.len()):
            if self.ops.category(i) == CommandCategory.MOVING:
                last = i
        if last is not None:
            self.seek(last)
        return last

    def seek_to_fraction(self, fraction: float):
        target = int(self.ops.len() * fraction)
        target = max(0, min(target, self.ops.len() - 1))
        self.seek(target)

    def seek_to_first_layer(self):
        for i in range(self.ops.len()):
            if self.ops.command_type(i) == CommandType.LAYER_START:
                self.seek(i)
                return i
        return 0

    def get_current_layer(self, doc: Doc) -> Optional[Layer]:
        uid = self.state.current_layer_uid
        if uid:
            item = doc.find_descendant_by_uid(uid)
            if isinstance(item, Layer):
                return item
        return None

    def _emit_layer_change(self):
        uid = self.state.current_layer_uid
        if uid != self._prev_layer_uid:
            self._prev_layer_uid = uid
            self.layer_changed.send(self, layer_uid=uid)


class _SnapshotBuilder:
    def __init__(
        self,
        ops: Ops,
        machine: Machine,
        doc: Doc,
        initial_state: MachineState,
    ):
        self.ops = ops
        self._machine = machine
        self._doc = doc
        self._current_index: int = -1
        self._source_axis: Axis = Axis.Y
        self._rotary_axis: Optional[Axis] = None
        self.state = initial_state

    def advance_to(self, index: int):
        for i in range(self._current_index + 1, index + 1):
            ct = self.ops.command_type(i)
            if ct == CommandType.LAYER_START:
                item = self._doc.find_descendant_by_uid(
                    self.ops.layer_uid(i)
                )
                if isinstance(item, Layer):
                    module = (
                        self._machine.rotary_modules.get(
                            item.rotary_module_uid
                        )
                        if item.rotary_module_uid
                        else None
                    )
                    if module:
                        self._source_axis = Axis.Y
                        if module.mode == RotaryMode.TRUE_4TH_AXIS:
                            self._rotary_axis = module.axis
                        else:
                            self._rotary_axis = Axis.Y
                    else:
                        self._source_axis = Axis.Y
                        self._rotary_axis = (
                            self._machine.get_rotary_axis_for_layer(item)
                        )
                else:
                    self._source_axis = Axis.Y
                    self._rotary_axis = None
            self.state.apply_command(self.ops, i)
        self._current_index = index
