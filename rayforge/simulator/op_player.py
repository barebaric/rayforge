from bisect import bisect_right
from typing import List, Optional, Tuple

from blinker import Signal
from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.types import CommandCategory, CommandType

from ..core.doc import Doc
from ..core.layer import Layer
from ..machine.kinematic_mapping import resolve_layer_rotary
from ..machine.models.machine import Machine
from .machine_state import MachineState

_SNAPSHOT_INTERVAL = 1000


def create_home_state(machine: Machine) -> MachineState:
    """Create the machine-origin (home) state for a playback session."""
    state = MachineState.from_axis_set(machine.axes)
    space = machine.get_coordinate_space()
    home_x, home_y = space.machine_point_to_world(0.0, 0.0)
    state.axes[Axis.X] = home_x
    state.axes[Axis.Y] = home_y
    return state


def build_snapshots(
    ops: Ops,
    machine: Machine,
    doc: Doc,
) -> List[Tuple[int, MachineState, Axis, Optional[Axis]]]:
    """Build the seek-acceleration snapshots for *ops*.

    Returns a fresh list of ``(target, state, source_axis, rotary_axis)``
    tuples spaced every ``_SNAPSHOT_INTERVAL`` commands, or an empty list
    for short op lists. The returned list may be built off the main thread
    and attached to an :class:`OpPlayer` via ``set_snapshots``.
    """
    n = ops.len()
    if n <= _SNAPSHOT_INTERVAL:
        return []
    builder = SnapshotBuilder(ops, machine, doc, create_home_state(machine))
    snapshots: List[Tuple[int, MachineState, Axis, Optional[Axis]]] = []
    for target in range(_SNAPSHOT_INTERVAL, n, _SNAPSHOT_INTERVAL):
        builder.advance_to(target - 1)
        # reached_textures is only needed for real-time playback,
        # not for seeking. Clear before snapshot to avoid copying
        # a set that grows to millions of entries.
        builder.state.reached_textures.clear()
        snapshots.append(
            (
                target,
                builder.state.copy(),
                builder._source_axis,
                builder._rotary_axis,
            )
        )
    return snapshots


class OpPlayer:
    def __init__(
        self,
        ops: Ops,
        machine: Machine,
        doc: Doc,
        build_snapshots: bool = True,
    ):
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
        self._snapshots: List[
            Tuple[int, MachineState, Axis, Optional[Axis]]
        ] = []
        if build_snapshots:
            self._build_snapshots()

    @property
    def snapshots(self):
        """The seek-acceleration snapshots (may be replaced asynchronously)."""
        return self._snapshots

    def set_snapshots(self, snapshots):
        """Replaces the seek-acceleration snapshots from an async build."""
        self._snapshots = snapshots

    def _build_snapshots(self):
        self._snapshots = build_snapshots(self.ops, self._machine, self._doc)

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
        return create_home_state(self._machine)

    def _update_rotary_config(self, layer_uid: str) -> None:
        item = self._doc.find_descendant_by_uid(layer_uid)
        layer = item if isinstance(item, Layer) else None
        cfg = resolve_layer_rotary(layer, self._machine)
        self._source_axis = cfg.source_axis
        self._rotary_axis = cfg.rotary_axis

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
            snap_index, snap_state, snap_source, snap_rotary = self._snapshots[
                snapshot_idx
            ]
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
        self._sync_machine_config()

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
            self._sync_machine_config()

    def _sync_machine_config(self):
        """Configure the machine for the layer at the current position.

        Falls back to the first layer of the document while the player is
        in the preamble (before the first LAYER_START command).
        """
        layer = self.get_current_layer(self._doc)
        if layer is None and self._doc.layers:
            layer = self._doc.layers[0]
        self._machine.configure_for_layer(layer)


class SnapshotBuilder:
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
                item = self._doc.find_descendant_by_uid(self.ops.layer_uid(i))
                layer = item if isinstance(item, Layer) else None
                cfg = resolve_layer_rotary(layer, self._machine)
                self._source_axis = cfg.source_axis
                self._rotary_axis = cfg.rotary_axis
            self.state.apply_command(self.ops, i)
        self._current_index = index
