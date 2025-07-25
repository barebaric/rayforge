"""
A wrapper for user-defined functions to be executed in a separate process.
This wrapper handles communication with the parent process via a queue,
allowing the subprocess to report progress, messages, and results.

WARNING: This file MUST NOT have any imports that cause any other
parts of the application to be initialized. It is designed to
be used during subprocess bootstrapping, where no other parts
of the application should be imported or initialized.
We can also not import any GTK or Adw classes here,
as this would cause the GTK main loop to be initialized,
which is not safe during bootstrapping.
In other words, we cannot use GLib.idle_add or similar.
"""

from multiprocessing import Queue
from typing import Any, Callable


# This wrapper needs to be a top-level function to be pickleable by
# multiprocessing
def process_target_wrapper(
    # The type of queue object will be determined by the multiprocessing
    # context.
    queue: Queue,
    log_level: int,
    user_func: Callable[..., Any],
    user_args: tuple[Any, ...],
    user_kwargs: dict[str, Any],
) -> None:
    """
    A wrapper that runs in the subprocess, calling the user's function
    and communicating status/results back to the parent via a queue.
    """
    print("process_target_wrapper() called")
    import logging

    logger = logging.getLogger("rayforge.tasker.process_target_wrapper")
    logger.setLevel(log_level)
    logger.info(
        f"Subprocess started with user function {user_func.__name__},"
        f"log level {log_level}",
    )

    import traceback
    from queue import Full
    from .proxy import ExecutionContextProxy

    proxy = ExecutionContextProxy(queue, parent_log_level=log_level)
    try:
        result = user_func(proxy, *user_args, **user_kwargs)
        queue.put_nowait(("done", result))
    except Exception:
        error_info = traceback.format_exc()
        try:
            queue.put(("error", error_info), block=True, timeout=1.0)
        except Full:
            logger.error(
                f"Could not report exception to parent process:\n{error_info}"
            )
