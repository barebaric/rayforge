from __future__ import annotations
from enum import Enum, auto
from typing import TYPE_CHECKING, Set, Callable, Optional

if TYPE_CHECKING:
    from rayforge.pipeline.artifact.key import ArtifactKey
    from rayforge.pipeline.artifact.handle import BaseArtifactHandle


class ContextState(Enum):
    """Represents the state of a GenerationContext."""

    ACTIVE = auto()
    SUPERSEDED = auto()


class GenerationContext:
    """
    Represents a single, isolated execution cycle of the pipeline.

    This object acts as a "container" for all tasks and resources (SHM handles)
    associated with a specific generation. It owns every SHM handle it creates
    and guarantees that the underlying memory remains valid until all tasks in
    this context have completed.

    Key Invariant:
        A GenerationContext owns every SHM handle it creates. It guarantees
        that the underlying memory remains valid until all tasks in this
        context have completed, regardless of whether a newer generation
        has started.
    """

    def __init__(
        self,
        generation_id: int,
        release_callback: Optional[
            Callable[["BaseArtifactHandle"], None]
        ] = None,
    ):
        """
        Initialize a new GenerationContext.

        Args:
            generation_id: A unique integer identifier for this generation.
            release_callback: Optional callback to release resources. If not
                provided, resources will not be released on shutdown.
        """
        self._generation_id = generation_id
        self._task_keys: Set["ArtifactKey"] = set()
        self._resources: Set["BaseArtifactHandle"] = set()
        self._release_callback = release_callback
        self._state = ContextState.ACTIVE
        self._shutdown = False

    @property
    def generation_id(self) -> int:
        """Return the unique generation identifier."""
        return self._generation_id

    @property
    def active_tasks(self) -> Set["ArtifactKey"]:
        """Return a copy of the set of active task keys."""
        return self._task_keys.copy()

    @property
    def resources(self) -> Set["BaseArtifactHandle"]:
        """Return a copy of the set of tracked resources."""
        return self._resources.copy()

    @property
    def is_shutdown(self) -> bool:
        """Return True if the context has been shut down."""
        return self._shutdown

    @property
    def state(self) -> ContextState:
        """Return the current state of this context."""
        return self._state

    def add_task(self, key: "ArtifactKey") -> None:
        """
        Register a task as active for this generation.

        Args:
            key: The ArtifactKey identifying the task.
        """
        self._task_keys.add(key)

    def task_did_finish(self, key: "ArtifactKey") -> None:
        """
        Mark a task as completed for this generation.

        If the context is superseded and has no more active tasks,
        automatically triggers shutdown to release resources.

        Args:
            key: The ArtifactKey identifying the completed task.
        """
        self._task_keys.discard(key)

        if not self.has_active_tasks() and self._is_superseded():
            self.shutdown()

    def has_active_tasks(self) -> bool:
        """
        Check if there are any active tasks remaining.

        Returns:
            True if there are active tasks, False otherwise.
        """
        return len(self._task_keys) > 0

    def _is_superseded(self) -> bool:
        """
        Check if this context has been superseded by a newer generation.

        Returns:
            True if this context is no longer active, False otherwise.
        """
        return self._state == ContextState.SUPERSEDED

    def mark_superseded(self) -> None:
        """
        Mark this context as superseded by a newer generation.
        """
        self._state = ContextState.SUPERSEDED

    def add_resource(self, handle: "BaseArtifactHandle") -> None:
        """
        Register a resource (SHM handle) with this generation.

        The context takes ownership of the resource and will release it
        when shutdown() is called.

        Args:
            handle: The BaseArtifactHandle to track.
        """
        self._resources.add(handle)

    def shutdown(self) -> None:
        """
        Release all tracked resources.

        This method iterates through all tracked resources and releases
        them via the release callback. After shutdown, the context should
        not be used for tracking new resources.
        """
        if self._shutdown:
            return
        self._shutdown = True

        if self._release_callback is not None:
            for handle in self._resources:
                self._release_callback(handle)
        self._resources.clear()
