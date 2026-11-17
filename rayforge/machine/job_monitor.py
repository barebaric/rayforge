from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from blinker import Signal
from raygeo.geo.types import Point3D
from raygeo.ops.types import CommandCategory

from ..shared.tasker import task_mgr
from .driver.driver import DeviceStatus
from .job_status import JobKind, JobOutcome, JobPhase, JobStatus

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ..context import RayforgeContext
    from .models.machine import Machine


logger = logging.getLogger(__name__)

# Minimum wall-clock time between two on_job_updated hook dispatches.
PROGRESS_HOOK_INTERVAL_SECONDS: float = 1.0


class JobMonitor:
    """
    Tracks and reports the progress of a machine job based on Ops data.

    This class calculates the total distance of a job from an Ops object and
    updates the progress as individual operations complete. It emits a signal
    with detailed metrics whenever the progress changes.

    Between begin() and end() it is also the addon-facing job surface:
    it dispatches the job monitoring hooks (see rayforge.core.hooks)
    with JobStatus snapshots, and registers itself on the machine so
    Machine.current_job serves live snapshots.

    The ETA counts down the modeled time remaining at the acknowledged
    command frontier. A pace correction is blended in after a warmup
    period: wall-clock time is compared against the time the model
    predicted for the acknowledged commands, and the remaining
    estimate is scaled by that ratio. This lets the ETA converge to
    the true remaining time when the model is uniformly off (e.g. the
    configured acceleration does not match the firmware) without ever
    reacting to the local speed of the last few commands: a windowed
    speed measurement multiplied by the global remaining distance
    used to swing the displayed ETA between a fraction and a multiple
    of the real remaining time whenever the job mixed fast and slow
    phases.
    """

    # No pace correction before this much wall-clock time has passed.
    # GRBL-style drivers acknowledge commands when the firmware buffers
    # them, so the frontier runs ahead of the tool by the planner depth
    # (several moves' worth of time); the pace ratio only becomes
    # meaningful once the elapsed time dwarfs that lead.
    PACE_WARMUP_SECONDS: float = 120.0

    # Duration over which the pace correction is blended in, so the
    # ETA glides towards the corrected value instead of jumping.
    PACE_RAMP_SECONDS: float = 480.0

    # Bounds for the pace ratio, guarding the display against
    # pathological inputs such as a driver resending its command
    # stream or a machine sitting in a feed hold for a long time.
    MIN_PACE: float = 0.1
    MAX_PACE: float = 10.0

    def __init__(
        self,
        ops: Ops,
        machine: Machine | None = None,
        kind: JobKind = JobKind.SEND,
        context: RayforgeContext | None = None,
        estimated_seconds: float | None = None,
        default_feed_rate: float = 1000.0,
        default_rapid_rate: float = 3000.0,
        acceleration: float = 1000.0,
        clock: Callable[[], float] | None = None,
    ):
        """
        Initializes the JobMonitor.

        Args:
            ops: The Ops object representing the job to be monitored.
            machine: The machine the job runs on, used for pause
                detection and the Machine.current_job pull API.
            kind: Whether the job is a send or frame job.
            context: The application context, providing the plugin
                manager the monitoring hooks are dispatched through.
            estimated_seconds: Total estimated job duration in seconds,
                as produced by ops.estimate_time() with the parameters
                below. Counts down as the ETA while no progress has
                been acknowledged and anchors the pace correction once
                progress arrives. GRBL-style drivers acknowledge
                commands when the firmware buffers them, so short jobs
                complete their streaming long before the machine stops
                moving.
            default_feed_rate: Feed rate assumption for the time model,
                as passed to ops.estimate_time().
            default_rapid_rate: Rapid rate assumption for the time
                model.
            acceleration: Acceleration assumption for the time model.
            clock: Monotonic time source; defaults to time.monotonic.
        """
        self.ops = ops
        self.machine = machine
        self.kind = kind
        self._context = context
        self.total_distance = ops.distance()
        self.estimated_seconds = estimated_seconds
        self.traveled_distance = 0.0
        self._clock = clock or time.monotonic
        self.start_time = self._clock()
        self._time_params = (
            default_feed_rate,
            default_rapid_rate,
            acceleration,
        )
        self._phase = JobPhase.RUNNING
        self._started_wall = time.time()
        self._last_hook_time = 0.0
        self._active = False

        # Modeled execution time (seconds) of the commands up to and
        # including the last acknowledged one.
        self._frontier_seconds = 0.0

        # Create a map from op_index to the distance of that op
        self._distance_map: dict[int, float] = {}
        last_point: Point3D | None = None
        for i in range(ops.len()):
            dist = ops.distance_at(i, last_point)
            self._distance_map[i] = dist
            if ops.category(i) == CommandCategory.MOVING:
                last_point = ops.endpoint(i)

        self.progress_updated = Signal()

    @property
    def status(self) -> JobStatus:
        """Returns an immutable snapshot of the current job status."""
        progress_fraction = (
            self.traveled_distance / self.total_distance
            if self.total_distance > 0
            else 1.0
        )
        return JobStatus(
            machine_id=self.machine.id if self.machine else "",
            machine_name=self.machine.name if self.machine else "",
            kind=self.kind,
            phase=self._phase,
            started_at=self._started_wall,
            elapsed_seconds=self._clock() - self.start_time,
            total_distance=self.total_distance,
            traveled_distance=self.traveled_distance,
            progress_fraction=progress_fraction,
            eta_seconds=self._eta_seconds(),
            estimated_seconds=self.estimated_seconds,
        )

    @property
    def metrics(self) -> dict[str, Any]:
        """Returns the current progress metrics as a dictionary."""
        status = self.status
        return {
            "total_distance": status.total_distance,
            "traveled_distance": status.traveled_distance,
            "progress_fraction": status.progress_fraction,
            "eta_seconds": status.eta_seconds,
        }

    def begin(self) -> None:
        """
        Reports the job as started and starts monitoring.

        Registers the monitor on the machine so Machine.current_job
        serves live snapshots, watches the device state for pause and
        resume transitions, and dispatches the on_job_started hook.
        """
        if self._active:
            return
        self._active = True
        self._started_wall = time.time()
        self._last_hook_time = self._clock() - PROGRESS_HOOK_INTERVAL_SECONDS
        if self.machine is not None:
            self.machine.state_changed.connect(self._on_machine_state_changed)
            self.machine.set_current_job(self)
        self._dispatch("on_job_started", status=self.status)

    def end(
        self,
        outcome: JobOutcome,
        message: str = "",
    ) -> None:
        """
        Reports the job as finished and stops monitoring.

        For a COMPLETED outcome the final snapshot is forced to 100%
        progress. Does nothing when the job was never begun or has
        already ended.

        Args:
            outcome: How the job ended.
            message: Error details when outcome is FAILED.
        """
        if not self._active:
            return
        self._active = False
        if self.machine is not None:
            self.machine.state_changed.disconnect(
                self._on_machine_state_changed
            )
            self.machine.clear_current_job(self)
        if outcome == JobOutcome.COMPLETED:
            self.mark_as_complete()
        self._dispatch(
            "on_job_finished",
            status=self.status,
            outcome=outcome,
            message=message,
        )

    def set_paused(self, paused: bool) -> None:
        """
        Reports the job entering or leaving a feed hold.

        Args:
            paused: True when the job is paused, False when resumed.
        """
        phase = JobPhase.PAUSED if paused else JobPhase.RUNNING
        if not self._active or self._phase == phase:
            return
        self._phase = phase
        self._dispatch(
            "on_job_paused" if paused else "on_job_resumed",
            status=self.status,
        )

    def _on_machine_state_changed(
        self, sender=None, state=None, **kwargs
    ) -> None:
        """Translates device status reports into pause/resume events."""
        if state is None:
            return
        if state.status == DeviceStatus.HOLD:
            self.set_paused(True)
        elif state.status == DeviceStatus.RUN:
            self.set_paused(False)

    def _eta_seconds(self) -> float | None:
        """Estimates the remaining job duration in seconds."""
        total = self.estimated_seconds
        if total is None or total <= 0.0:
            return None
        remaining = max(total - self._frontier_seconds, 0.0)
        elapsed = self._clock() - self.start_time
        if remaining <= 0.0:
            return 0.0
        if self._frontier_seconds <= 0.0:
            # Nothing acknowledged yet: count down the estimate. This
            # also covers drivers that never report granular progress.
            return max(total - elapsed, 0.0)

        pace = elapsed / self._frontier_seconds
        pace = min(max(pace, self.MIN_PACE), self.MAX_PACE)
        ramp = (elapsed - self.PACE_WARMUP_SECONDS) / self.PACE_RAMP_SECONDS
        weight = min(max(ramp, 0.0), 1.0)
        return remaining * (1.0 + weight * (pace - 1.0))

    def _model_time_at(self, op_index: int) -> float:
        """The modeled execution time (s) up to and including op_index."""
        if self.estimated_seconds is None:
            return 0.0
        modeled = self.ops.get_cumulative_time_at(op_index, *self._time_params)
        return min(modeled, self.estimated_seconds)

    def update_progress(self, op_index: int) -> None:
        """
        Updates the progress based on a completed operation.

        Args:
            op_index: The index of the Ops command that has finished.
        """
        logger.debug(f"JobMonitor: progress updated for op_index {op_index}.")

        distance_for_op = self._distance_map.get(op_index, 0.0)

        # Always update traveled_distance (even if distance is 0)
        self.traveled_distance += distance_for_op

        # Clamp to ensure we don't exceed total_distance due to float errors
        self.traveled_distance = min(
            self.traveled_distance, self.total_distance
        )

        # The frontier only moves forward: a repeated or reordered
        # index (e.g. after a connection hiccup) must not rewind it.
        self._frontier_seconds = max(
            self._frontier_seconds, self._model_time_at(op_index)
        )

        logger.debug(
            f"  -> New progress: {self.metrics['progress_fraction']:.2f}"
        )
        # Always emit the signal, even for operations with 0 distance
        self.progress_updated.send(self, metrics=self.metrics)
        self._maybe_report_progress()

    def _maybe_report_progress(self) -> None:
        """Dispatches on_job_updated at a limited rate."""
        if not self._active:
            return
        now = self._clock()
        if now - self._last_hook_time < PROGRESS_HOOK_INTERVAL_SECONDS:
            return
        self._last_hook_time = now
        self._dispatch("on_job_updated", status=self.status)

    def _dispatch(self, hook_name: str, **kwargs: Any) -> None:
        """Schedules an addon hook call on the main thread."""
        task_mgr.schedule_on_main_thread(self._call_hooks, hook_name, **kwargs)

    def _call_hooks(self, hook_name: str, **kwargs: Any) -> None:
        """Calls an addon hook, isolating the job from addon errors."""
        if self._context is None:
            return
        try:
            getattr(self._context.plugin_mgr.hook, hook_name)(**kwargs)
        except Exception:
            logger.exception(f"JobMonitor: addon hook '{hook_name}' failed")

    def mark_as_complete(self) -> None:
        """
        Marks the job as fully complete, setting progress to 100%.
        """
        self.traveled_distance = self.total_distance
        if self.estimated_seconds:
            self._frontier_seconds = self.estimated_seconds
        logger.debug("JobMonitor: marked as complete.")
        self.progress_updated.send(self, metrics=self.metrics)
