import math
from bisect import bisect_right

from blinker import Signal
from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.types import CommandCategory, CommandType

from ..core.doc import Doc
from ..core.layer import Layer
from ..machine.kinematic_mapping import resolve_layer_rotary
from ..machine.kinematic_math import KinematicMath
from ..machine.models.machine import Machine
from ..machine.models.rotary_module import RotaryType
from .machine_state import MachineState

_SNAPSHOT_INTERVAL = 1000


def create_home_state(machine: Machine) -> MachineState:
    """Create the machine-origin (home) state for a playback session."""
    state = MachineState.from_axis_set(machine.axes)
    home_x, home_y = machine.panel.machine_point_to_world(0.0, 0.0)
    state.axes[Axis.X] = home_x
    state.axes[Axis.Y] = home_y
    return state


def build_snapshots(
    ops: Ops,
    machine: Machine,
    doc: Doc,
) -> list[tuple[int, MachineState, Axis, Axis | None, float]]:
    """Build the seek-acceleration snapshots for *ops*.

    Returns a fresh list of ``(target, state, source_axis, rotary_axis,
    rotary_dpm)`` tuples spaced every ``_SNAPSHOT_INTERVAL`` commands,
    or an empty list for short op lists. The returned list may be built
    off the main thread and attached to an :class:`OpPlayer` via
    ``set_snapshots``.
    """
    n = ops.len()
    if n <= _SNAPSHOT_INTERVAL:
        return []
    builder = SnapshotBuilder(ops, machine, doc, create_home_state(machine))
    snapshots: list[tuple[int, MachineState, Axis, Axis | None, float]] = []
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
                builder._rotary_dpm,
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
        self._rotary_dpm: float = 0.0
        self._prev_layer_uid: str | None = None
        self.state = self._create_home_state()
        self._home_axes: dict[Axis, float] = dict(self.state.axes)
        self.layer_changed = Signal()
        self._snapshots: list[
            tuple[int, MachineState, Axis, Axis | None, float]
        ] = []
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

    @property
    def sim_time(self) -> float:
        """The current simulated playback time (seconds)."""
        return self._sim_time

    def set_sim_time(self, t: float) -> None:
        """Set the simulated playback time and cache the playhead progress.

        The playhead is described as the command currently being
        executed plus the fraction of its duration that has elapsed.
        Once playback has run to completion the laser is turned off.

        Note that time alone cannot distinguish positions inside a
        cluster of zero-duration commands; use :meth:`set_playhead`
        when the desired position is known as a command index.
        """
        self._sim_time = float(t)
        self._playback = self._compute_playback_progress(self._sim_time)
        if self.is_finished:
            self.state.laser_on = False

    def set_playhead(self, index: int) -> None:
        """Place the playhead on command *index* (that command executed).

        Seeks the machine state to *index*, sets the simulated clock to
        its completion time, and anchors the cached playback progress
        to the *following* command. Anchoring by index keeps stepping
        and external syncing exact even when many commands share the
        same cumulative time (zero-duration state changes).
        """
        n = self.ops.len()
        index = max(0, min(index, n - 1))
        t = self.get_cumulative_time(index)
        self._anchor_progress(index, t, index + 1)

    def set_progress_anchor(self, completed: int, t: float) -> None:
        """Anchor the playhead between *completed* and the next command.

        Seeks the machine state so commands up to *completed* are
        applied, sets the simulated clock to *t*, and caches playback
        progress as command ``completed + 1`` in progress. Used when a
        position is known both as an index boundary and a time (for
        example scrubbing the time-based slider while paused).
        """
        n = self.ops.len()
        completed = max(-1, min(completed, n - 1))
        k = completed + 1
        if completed < 0:
            self.state = self._create_home_state()
            self._current_index = -1
            self._source_axis = Axis.Y
            self._rotary_axis = None
            self._rotary_dpm = 0.0
            self._prev_layer_uid = None
        elif k >= n:
            self.seek(n - 1)
            self._sim_time = self.get_cumulative_time(n - 1)
            self._playback = (n - 1, 1.0)
            return
        else:
            self.seek(completed)
        self._sim_time = float(t)
        self._playback = self._progress_at_index(float(t), k)
        if self.is_finished:
            self.state.laser_on = False

    def _anchor_progress(self, completed: int, t: float, k: int) -> None:
        self.seek(completed)
        self._sim_time = float(t)
        n = self.ops.len()
        if k >= n:
            self._playback = (n - 1, 1.0)
        else:
            self._playback = self._progress_at_index(float(t), k)
        if self.is_finished:
            self.state.laser_on = False

    def _progress_at_index(self, t: float, k: int) -> tuple[int, float]:
        """Playback progress anchored at command *k* for time *t*."""
        start = self.get_cumulative_time(k - 1) if k > 0 else 0.0
        span = self.get_cumulative_time(k) - start
        if span <= 0.0:
            return (k, 0.0)
        frac = max(0.0, min(1.0, (t - start) / span))
        return (k, frac)

    @property
    def is_finished(self) -> bool:
        """True when the playhead has reached the end of the job."""
        p, frac = self._playback
        return p >= self.ops.len() - 1 and frac >= 1.0

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

    def _interpolate_rotary_arc(
        self,
        p: int,
        frac: float,
        start_axes: dict,
        end: tuple,
        axes: dict,
    ) -> bool:
        """Interpolate along a rotary-mode arc, or fall back to linear.

        Rotary mapping rewrites endpoint Y while the arc's stored
        radius (from I/J) still describes the unwrapped surface
        geometry, so the Cartesian angle math cannot use the mapped
        endpoints directly. Instead the planar arc is reconstructed in
        (X, surface) space: the Y travel comes from the rotary axis
        itself (``delta_angle / degrees_per_mm``), which is
        authoritative, and the circle is solved through BOTH
        reconstructed endpoints with the stored radius - so the
        interpolation always lands exactly on the command's target.

        Returns True when the reconstruction succeeded and ``axes``
        holds the interpolated position; False when the caller must
        fall back to linear interpolation.
        """
        i, j, cw = self.ops.arc_params(p)
        radius = math.hypot(i, j)
        if radius <= 1e-9:
            return False
        ea = self.ops.extra_axes(p) or {}
        a_end = ea.get(self._rotary_axis)
        if a_end is None:
            return False
        dpm = self._rotary_dpm
        a_start = start_axes.get(self._rotary_axis, 0.0)
        sx = start_axes.get(Axis.X, 0.0)
        su = a_start / dpm
        ex = end[0]
        eu = su + (a_end - a_start) / dpm

        # Solve the circle through (sx, su) and (ex, eu) with the
        # stored radius. The center lies on the chord's perpendicular
        # bisector; pick the side that agrees with the stored I/J.
        dx, dy = ex - sx, eu - su
        chord = math.hypot(dx, dy)
        if chord > 2.0 * radius:
            return False
        cx0, cu0 = sx + i, su + j
        if chord <= 1e-9:
            # Full turn: the stored center defines the circle.
            cx, cu = cx0, cu0
        else:
            h = math.sqrt(max(radius * radius - (chord * 0.5) ** 2, 0.0))
            mx, mu = sx + dx * 0.5, su + dy * 0.5
            px, pu = -dy / chord, dx / chord
            if (cx0 - mx) * px + (cu0 - mu) * pu >= 0.0:
                cx, cu = mx + px * h, mu + pu * h
            else:
                cx, cu = mx - px * h, mu - pu * h

        angle_start = math.atan2(su - cu, sx - cx)
        angle_end = math.atan2(eu - cu, ex - cx)
        sweep_ccw = (angle_end - angle_start) % (2.0 * math.pi)
        sweep = sweep_ccw - 2.0 * math.pi if cw else sweep_ccw
        angle = angle_start + frac * sweep

        axes[Axis.X] = cx + radius * math.cos(angle)
        u = cu + radius * math.sin(angle)
        axes[self._rotary_axis] = a_start + (u - su) * dpm
        if Axis.Z in axes:
            axes[Axis.Z] = start_axes.get(Axis.Z, 0.0) + frac * (
                end[2] - start_axes.get(Axis.Z, 0.0)
            )
        return True

    def render_state(self) -> MachineState:
        """State for rendering, with axes interpolated within the
        in-progress command during playback.

        ``laser_on`` follows the in-progress command so the laser beam
        fires for the whole duration of a cut (not just after the cut's
        command completes), except once playback has run to completion,
        where the laser is always off. Returns the plain machine state
        when paused or when the in-progress command does not move
        (state changes, dwells), so stepped frames render exactly like
        the non-interpolated path.
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
        is_arc = self.ops.command_type(p) == CommandType.ARC_TO
        rotary_done = False
        spatial_done = False
        if is_arc:
            if self._rotary_axis is not None and self._rotary_dpm != 0.0:
                # Rotary arcs are reconstructed from the rotary axis
                # travel; mapping may have detached the stored circle
                # from the mapped endpoints.
                rotary_done = self._interpolate_rotary_arc(
                    p, frac, start_axes, end, axes
                )
                spatial_done = rotary_done
            else:
                self._interpolate_arc(p, frac, start_axes, end, axes)
                spatial_done = True
        if not spatial_done:
            for axis, value in (
                (Axis.X, end[0]),
                (Axis.Y, end[1]),
                (Axis.Z, end[2]),
            ):
                if axis not in axes:
                    continue
                axes[axis] = start_axes.get(axis, 0.0) + frac * (
                    value - start_axes.get(axis, 0.0)
                )

        ea = self.ops.extra_axes(p)
        if ea:
            for axis, value in ea.items():
                # The rotary axis is already interpolated by
                # _interpolate_rotary_arc when that path succeeded;
                # otherwise it blends linearly like the other extras.
                if rotary_done and axis == self._rotary_axis:
                    continue
                axes[axis] = start_axes.get(axis, 0.0) + frac * (
                    value - start_axes.get(axis, 0.0)
                )
        st = MachineState(axis_letters=axes.keys())
        st.axes = axes
        st.laser_on = (
            self.ops.command_type(p) != CommandType.MOVE_TO
            and not self.is_finished
        )
        return st

    def _interpolate_arc(
        self,
        p: int,
        frac: float,
        start_axes: dict,
        end: tuple,
        axes: dict,
    ) -> None:
        """Interpolate along an arc path for the in-progress command.

        Spatial axes (X) follow the circle geometry.  The rotary axis
        (when active) has its angle computed from the arc's Cartesian
        Y displacement, converted via the stored degrees-per-mm factor.
        """
        i, j, cw = self.ops.arc_params(p)
        sx = start_axes.get(Axis.X, 0.0)
        radius = math.hypot(i, j)
        angle_start = math.atan2(-j, -i)

        # Detect full circle: endpoint X matches start X and the
        # rotary angle (if any) doesn't change between start/end.
        is_full_circle = abs(end[0] - sx) < 1e-6
        if is_full_circle and self._rotary_axis is not None:
            ea = self.ops.extra_axes(p)
            end_rot = ea.get(self._rotary_axis, 0.0) if ea else 0.0
            is_full_circle = (
                abs(start_axes.get(self._rotary_axis, 0.0) - end_rot) < 1e-6
            )

        if is_full_circle:
            sweep = -2.0 * math.pi if cw else 2.0 * math.pi
        else:
            angle_end = math.atan2(
                end[1] - (start_axes.get(Axis.Y, 0.0) + j),
                end[0] - sx - i,
            )
            sweep_ccw = (angle_end - angle_start) % (2.0 * math.pi)
            sweep = sweep_ccw - 2.0 * math.pi if cw else sweep_ccw

        angle = angle_start + frac * sweep

        # X axis always follows the circle geometry.
        axes[Axis.X] = sx + i + radius * math.cos(angle)

        # Rotary axis: convert the arc's Cartesian Y displacement to
        # a rotation angle using the degrees-per-mm factor.
        if self._rotary_axis is not None and self._rotary_dpm != 0.0:
            y_disp = j + radius * math.sin(angle)
            axes[self._rotary_axis] = (
                start_axes.get(self._rotary_axis, 0.0)
                + y_disp * self._rotary_dpm
            )
        elif Axis.Y in axes:
            axes[Axis.Y] = (
                start_axes.get(Axis.Y, 0.0) + j + radius * math.sin(angle)
            )

        if Axis.Z in axes:
            axes[Axis.Z] = start_axes.get(Axis.Z, 0.0) + frac * (
                end[2] - start_axes.get(Axis.Z, 0.0)
            )

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
        if cfg.rotary_axis is not None and cfg.module is not None:
            diameter = (
                layer.rotary_diameter
                if layer and layer.rotary_diameter
                else cfg.module.default_diameter
            )
            ratio = KinematicMath.gear_ratio(
                cfg.module.rotary_type == RotaryType.ROLLERS,
                diameter,
                cfg.module.roller_diameter,
            )
            self._rotary_dpm = KinematicMath.mm_to_degrees(
                1.0, diameter, ratio
            ) * (-1.0 if cfg.module.reverse_axis else 1.0)
        else:
            self._rotary_dpm = 0.0

    def seek(self, index: int):
        if index >= self.ops.len():
            raise IndexError(
                f"Index {index} out of range "
                f"(ops has {self.ops.len()} commands)"
            )
        index = max(index, 0)

        snapshot_idx = self._find_snapshot(index)
        if snapshot_idx is not None:
            (
                snap_index,
                snap_state,
                snap_source,
                snap_rotary,
                snap_dpm,
            ) = self._snapshots[snapshot_idx]
            self.state = snap_state.copy()
            self._current_index = snap_index - 1
            self._source_axis = snap_source
            self._rotary_axis = snap_rotary
            self._rotary_dpm = snap_dpm
        else:
            self.state = self._create_home_state()
            self._current_index = -1
            self._source_axis = Axis.Y
            self._rotary_axis = None
            self._rotary_dpm = 0.0

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

    def sync_state_to_playhead(self) -> None:
        """Move the machine state to the command before the playhead.

        Used during time-based playback, where the playhead is defined
        by the simulated clock instead of explicit seeks. Forward
        movement is incremental, so steady playback only pays for the
        commands crossed since the previous tick; scrubbing backwards
        falls back to :meth:`seek` (snapshot-accelerated).
        """
        p, _frac = self._playback
        target = min(p - 1, self.ops.len() - 1)
        if target < 0 or target == self._current_index:
            return
        if target > self._current_index:
            self.advance_to(target)
        else:
            self.seek(target)
        self._emit_layer_change()

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
        self._rotary_dpm: float = 0.0
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
                if cfg.rotary_axis is not None and cfg.module is not None:
                    diameter = (
                        layer.rotary_diameter
                        if layer and layer.rotary_diameter
                        else cfg.module.default_diameter
                    )
                    ratio = KinematicMath.gear_ratio(
                        cfg.module.rotary_type == RotaryType.ROLLERS,
                        diameter,
                        cfg.module.roller_diameter,
                    )
                    self._rotary_dpm = KinematicMath.mm_to_degrees(
                        1.0, diameter, ratio
                    ) * (-1.0 if cfg.module.reverse_axis else 1.0)
                else:
                    self._rotary_dpm = 0.0
            self.state.apply_command(self.ops, i)
        self._current_index = index
