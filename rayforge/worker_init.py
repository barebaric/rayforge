import builtins
import logging


def initialize_worker():
    """
    Sets up the minimal environment required for a worker subprocess.

    This function is lightweight and has no dangerous imports. It is the
    designated `worker_initializer` for the TaskManager.
    """
    # Install a fallback gettext translator. This ensures the '_'
    # function exists during the module import phase.
    if not hasattr(builtins, "_"):
        setattr(builtins, "_", lambda s: s)

    logging.getLogger(__name__).debug(
        "Worker process initialized successfully."
    )
