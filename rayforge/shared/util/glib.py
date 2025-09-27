from typing import Callable, Optional, Tuple, Any
from gi.repository import GLib


def falsify(func, *args, **kwargs):
    """
    Wrapper for GLib.idle_add, as function must return False, otherwise it
    is automatically rescheduled into the event loop.
    """
    func(*args, **kwargs)
    return False


def idle_add(func, *args, **kwargs):
    """
    Wrapper for GLib.idle_add to support multiple args and kwargs.
    """
    GLib.idle_add(lambda: falsify(func, *args, **kwargs))


class DebounceMixin:
    """A mixin to add debouncing capabilities to a class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debounce_timer = 0
        self._debounced_callback: Optional[Callable] = None
        self._debounced_args: Tuple = ()

    def _debounce(self, callback: Callable, *args: Any):
        """
        Schedules a callback to be called after a short delay, cancelling any
        previously scheduled callback.
        """
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
        self._debounced_callback = callback
        self._debounced_args = args
        self._debounce_timer = GLib.timeout_add(
            150, self._commit_debounced_change
        )

    def _commit_debounced_change(self) -> bool:
        """Executes the debounced callback and resets the timer."""
        if self._debounced_callback:
            self._debounced_callback(*self._debounced_args)
        self._debounce_timer = 0
        return GLib.SOURCE_REMOVE
