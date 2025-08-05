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
