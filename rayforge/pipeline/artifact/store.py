"""
In-process artifact storage with reference counting.

Replaces the former shared-memory (SHM) store.  All artifacts live as
plain Python objects in a dict keyed by a UUID.  Handles carry the UUID
in their ``key`` field plus any metadata the artifact type needs.

Lifecycle is managed through reference counting via
:meth:`retain` / :meth:`release`.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class ArtifactStore:
    """In-process artifact store with reference-counted handles."""

    def __init__(self):
        self._artifacts: Dict[str, BaseArtifact] = {}
        self._handles: Dict[str, BaseArtifactHandle] = {}

    def shutdown(self):
        for key in list(self._handles):
            self._handles.pop(key, None)
        self._artifacts.clear()

    def _get_or_create_handle(
        self, proto_handle: BaseArtifactHandle
    ) -> BaseArtifactHandle:
        key = proto_handle.key
        canonical = self._handles.get(key)
        if canonical is not None:
            canonical.refcount += 1
            return canonical
        proto_handle.refcount = 1
        self._handles[key] = proto_handle
        return proto_handle

    def put(
        self,
        artifact: BaseArtifact,
        creator_tag: str = "unknown",
        generation_context: Optional[Any] = None,
    ) -> BaseArtifactHandle:
        """Store *artifact* and return a lightweight handle."""
        key = f"rf_{creator_tag}_{uuid.uuid4().hex[:16]}"
        handle = artifact.build_handle(key)
        handle.refcount = 1
        self._handles[key] = handle
        self._artifacts[key] = artifact
        return handle

    def get(self, handle: BaseArtifactHandle) -> BaseArtifact:
        """Return the artifact referenced by *handle*."""
        try:
            return self._artifacts[handle.key]
        except KeyError:
            raise RuntimeError(f"Artifact '{handle.key}' not found in store.")

    def release(self, handle: BaseArtifactHandle) -> None:
        """Decrement refcount; delete artifact when it reaches zero."""
        key = handle.key
        canonical = self._handles.get(key, handle)
        if canonical.refcount > 1:
            canonical.refcount -= 1
            return
        self._handles.pop(key, None)
        self._artifacts.pop(key, None)

    def close_handle(self, handle: BaseArtifactHandle) -> None:
        self.release(handle)

    def retain(self, handle: BaseArtifactHandle) -> bool:
        key = handle.key
        canonical = self._handles.get(key)
        if canonical:
            canonical.refcount += 1
            return True
        return False

    def forget(self, handle: BaseArtifactHandle) -> None:
        self.release(handle)

    @contextmanager
    def checkout_handle(
        self, handle: Optional[BaseArtifactHandle]
    ) -> Generator[Optional[BaseArtifact], None, None]:
        """Retain *handle*, yield its artifact, then release it."""
        if handle is None:
            yield None
            return
        self.retain(handle)
        try:
            yield self.get(handle)
        finally:
            self.release(handle)
