from __future__ import annotations
import inspect
import logging
from typing import Optional


def get_caller_stack(depth: int = 4) -> str:
    """
    Return a compact call stack string showing module:function names.

    Args:
        depth: Number of caller frames to include (skips this function
            and its immediate caller).

    Returns:
        A string like "module:func <- module2:func2 <- ..."
    """
    frames = []
    frame = inspect.currentframe()
    try:
        for _ in range(2):
            if frame is None:
                break
            frame = frame.f_back
        for _ in range(depth):
            if frame is None:
                break
            name = frame.f_globals.get("__name__", "?")
            func = frame.f_code.co_name
            frames.append(f"{name.split('.')[-1]}:{func}")
            frame = frame.f_back
    finally:
        del frame
    return " <- ".join(frames) if frames else "?"


def safe_caller_stack(depth: int = 4) -> Optional[str]:
    """
    Get caller stack only if DEBUG-level logging is enabled.

    This avoids the overhead of stack inspection when not debugging.
    Always returns None if the root logger's level is higher than DEBUG.

    Args:
        depth: Number of caller frames to include.

    Returns:
        Stack string if debugging, None otherwise.
    """
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        return get_caller_stack(depth)
    return None
