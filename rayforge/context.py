import threading
from typing import Optional
from .pipeline.artifact.store import ArtifactStore

_context_instance: Optional["RayforgeContext"] = None
_context_lock = threading.Lock()


class RayforgeContext:
    """
    A central, singleton context for managing the lifecycle of major
    application services.
    """

    def __init__(self):
        """
        Initializes the context. This constructor is lightweight and safe
        to call from any process.
        """
        self.artifact_store = ArtifactStore()
        # Other managers will be added here in subsequent steps.

    def shutdown(self):
        """
        Shuts down all managed services in the correct order.
        """
        self.artifact_store.shutdown()
        # Other manager shutdowns will be added here.


def get_context() -> RayforgeContext:
    """
    A thread-safe, lazy-initializing accessor for the global RayforgeContext
    singleton.
    """
    global _context_instance
    if _context_instance is None:
        with _context_lock:
            # Double-check lock to prevent race conditions
            if _context_instance is None:
                _context_instance = RayforgeContext()
    return _context_instance
