from bisect import bisect_right

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
) -> list[tuple[int, MachineState, Axis, Axis | None]]:
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
    snapshots: list[tuple[int, MachineState, Axis, Axis | None]] = []
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
        time_ops: Ops | None = None,
    ):
        if not ops or ops.is_empty():
            raise ValueError("OpPlayer requires a non-empty Ops")
        self.ops = ops
        # Time model backing: defaults to *ops* itself. Rotary-mapped
        # ops keep their endpoints at a constant Y (the real rotation
        # lives in extra axes), which distorts line distances and makes
        # arcs degenerate, so playback passes the unmapped ops here.
        # Command indices and order are identical, so the cumulative
        # time index stays in sync with *ops*.
        self._time_ops = time_ops if time_ops is not None else ops
        self._machine = machine
        self._doc = doc
        self._current_index: int = -1
        self._source_axis: Axis = Axis.Y
        self._rotary_axis: Axis | None = None
        self._prev_layer_uid: str | None = None
        self.state = self._create_home_state()
        self._home_axes: dict[Axis, float] = dict(self.state.axes)
        self.layer_changed = Signal()
        self._snapshots: list[tuple[int, MachineState, Axis, Axis | None]] = []
        # Playback time model: feed/rapid rates in mm/min, accel in
        # mm/s^2. Defaults match the raygeo cumulative-time index.
        self._play_params: tuple[float, float, float] = (
            1000.0,
            3000.0,
            1000.0,
        )
        self._sim_time: float = 0.0
        self._playback: tuple[int, float] = (0, 0.0)
        if build_snapshots:
            self._build_snapshots()

    @property
    def snapshots(self):
        """The seek-acceleration snapshots (may be replaced asynchronously)."""
        return self._snapshots

    def set_snapshots(self, snapshots):
        """Replaces the seek-acceleration snapshots from an async build."""
        self._snapshots = snapshots

    def set_playback_params(
        self,
        feed_mm_min: float,
        rapid_mm_min: float,
        accel_mm_s2: float,
    ):
        """Set the machine parameters for the playback time model.

        Feed and rapid rates are in mm/min (the unit stored in the ops),
        acceleration in mm/s^2. These mirror the arguments of the raygeo
        cumulative-time index.
        """
        self._play_params = (
            float(feed_mm_min),
            float(rapid_mm_min),
            float(accel_mm_s2),
        )

    def find_index_at_sim_time(self, t: float) -> int:
        """Command index in effect at simulated time *t* (seconds)."""
        return self._time_ops.find_index_at_time(t, *self._play_params)

    def get_cumulative_time(self, idx: int) -> float:
        """Cumulative simulated time (seconds) up to command *idx*."""
        return self._time_ops.get_cumulative_time_at(idx, *self._play_params)

    def set_sim_time(self, t: float) -> None:
        """Set the simulated playback time and cache the playhead progress.

        The playhead is described as the command currently being
        executed plus the fraction of its duration that has elapsed.
        """
        self._sim_time = float(t)
        self._playback = self._compute_playback_progress(self._sim_time)

    def playback_progress(self) -> tuple[int, float]:
        """Return ``(in_progress_command_index, fraction)`` for the
        current simulated time.

        ``fraction`` is in ``[0, 1]``; it is 0 while the playhead sits
        exactly on a command boundary and 1.0 once everything is done.
        """
        return self._playback

    def _compute_playback_progress(self, t: float) -> tuple[int, float]:
        n = self.ops.len()
        if n == 0:
            return (0, 0.0)
        idx = self.find_index_at_sim_time(t)
        t_end = self.get_cumulative_time(idx)
        if t < t_end:
            # The clamped index has not completed yet (t < cum[0]).
            span = self.get_cumulative_time(0)
            if span <= 0.0:
                return (0, 0.0)
            return (0, max(0.0, min(1.0, t / span)))
        if idx + 1 >= n:
            return (idx, 1.0)
        span = self.get_cumulative_time(idx + 1) - t_end
        if span <= 0.0:
            return (idx + 1, 0.0)
        frac = max(0.0, min(1.0, (t - t_end) / span))
        return (idx + 1, frac)

    def render_state(self) -> MachineState:
        """State for rendering, with axes interpolated within the
        in-progress command during playback.

        ``laser_on`` follows the in-progress command so the laser beam
        fires for the whole duration of a cut (not just after the cut's
        command completes). Returns the plain machine state when paused
        or when the in-progress command does not move (state changes,
        dwells), so stepped frames render exactly like the
        non-interpolated path.
        """
        p, frac = self._playback
        if frac <= 0.0 or p >= self.ops.len():
            return self.state
        if self.ops.category(p) != CommandCategory.MOVING:
            return self.state
        end = self.ops.endpoint(p)
        if p == 0:
            # Nothing has completed yet: the head starts at home.
            start_axes = self._home_axes
        else:
            start_axes = self.state.axes
        axes = dict(start_axes)
        for axis, value in (
            (Axis.X, end[0]),
            (Axis.Y, end[1]),
            (Axis.Z, end[2]),
        ):
            axes[axis] = axes.get(axis, 0.0) + frac * (
                value - axes.get(axis, 0.0)
            )
        ea = self.ops.extra_axes(p)
        if ea:
            for axis, value in ea.items():
                axes[axis] = axes.get(axis, 0.0) + frac * (
                    value - axes.get(axis, 0.0)
                )
        st = MachineState(axis_letters=axes.keys())
        st.axes = axes
        st.laser_on = self.ops.command_type(p) != CommandType.MOVE_TO
        return st

    def _build_snapshots(self):
        self._snapshots = build_snapshots(self.ops, self._machine, self._doc)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def source_axis(self) -> Axis:
        return self._source_axis

    @property
    def rotary_axis(self) -> Axis | None:
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
        index = max(index, 0)

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

        self.advance_to(index)
        self._emit_layer_change()

    def _find_snapshot(self, index: int) -> int | None:
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

    def seek_last_movement(self) -> int | None:
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

    def get_current_layer(self, doc: Doc) -> Layer | None:
        uid = self.state.current_layer_uid
        if uid:
            item = doc.find_descendant_by_uid(uid)
            if isinstance(item, Layer):
                return item
        return None

    def get_effective_layer(self, doc: Doc) -> Layer | None:
        """Return the layer that should drive playback configuration.

        Falls back to the first layer of the document while the player is
        in the preamble (before the first LAYER_START command).
        """
        layer = self.get_current_layer(doc)
        if layer is None and doc.layers:
            layer = doc.layers[0]
        return layer

    def _emit_layer_change(self):
        uid = self.state.current_layer_uid
        if uid != self._prev_layer_uid:
            self._prev_layer_uid = uid
            self.layer_changed.send(self, layer_uid=uid)


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
        self._rotary_axis: Axis | None = None
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
