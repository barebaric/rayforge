"""
Public, addon-facing types describing the state of a running job.

Addons receive a `JobStatus` snapshot through the job monitoring hooks
(see `rayforge.core.hooks`) and can poll the latest snapshot via
`Machine.current_job`. These types are part of the versioned addon API:
fields are additive only across releases.
"""

from dataclasses import dataclass
from enum import Enum, auto


class JobKind(Enum):
    """The kind of job being executed on a machine."""

    SEND = auto()
    FRAME = auto()


class JobPhase(Enum):
    """The execution phase of a running job."""

    RUNNING = auto()
    PAUSED = auto()


class JobOutcome(Enum):
    """How a job ended."""

    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class JobStatus:
    """
    An immutable snapshot of a job's progress at a point in time.

    Attributes:
        machine_id: Unique id of the machine running the job.
        machine_name: User-visible name of the machine.
        kind: Whether the job is a send (burn) or frame (outline
            trace) job.
        phase: RUNNING or PAUSED.
        started_at: Wall-clock time (seconds since the epoch) at
            which the job started.
        elapsed_seconds: Seconds elapsed since the job started.
        total_distance: Total path length of the job in millimeters.
        traveled_distance: Path length completed so far, in
            millimeters.
        progress_fraction: Completed fraction of the job, 0.0 to 1.0.
        eta_seconds: Estimated remaining seconds, or None when no
            estimate is available.
        estimated_seconds: Total modeled job duration in seconds, or
            None when unknown.
    """

    machine_id: str
    machine_name: str
    kind: JobKind
    phase: JobPhase
    started_at: float
    elapsed_seconds: float
    total_distance: float
    traveled_distance: float
    progress_fraction: float
    eta_seconds: float | None
    estimated_seconds: float | None
