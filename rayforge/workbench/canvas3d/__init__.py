import logging

# This flag can be checked by other parts of the application
# to decide whether to instantiate and show the 3D canvas.
initialized = False

# Store the exception if initialization fails, for better debugging.
initialization_error = None

logger = logging.getLogger(__name__)


def initialize():
    """
    Tries to initialize the required OpenGL bindings.

    This function attempts to import PyOpenGL. A failure indicates that
    the necessary libraries are not available on the system, and the 3D
    canvas cannot be used. It sets the package-level 'initialized' flag
    accordingly. This should be called from the main application entry
    point before any UI is created.
    """
    global initialized, initialization_error
    if initialized or initialization_error:
        return

    try:
        # The import itself triggers platform-specific initialization and will
        # fail if the necessary libraries are not found (e.g., libGL.so).
        from OpenGL import GL  # Imported for side effects

        _ = GL  # Mark as used to silence pyflakes

        logger.info("PyOpenGL imported successfully. 3D canvas is available.")
        initialized = True
    except ImportError as e:
        initialization_error = e
        logger.error(
            "Failed to import PyOpenGL. The 3D canvas will be disabled. "
            "Error: %s",
            e,
        )
        logger.info(
            "This might be due to missing graphics drivers or an "
            "unsupported environment. Please ensure OpenGL libraries are "
            "installed on your system (e.g., 'mesa-libGL' on Linux)."
        )
        initialized = False
    except Exception as e:
        # Catch other potential errors during initial module load.
        initialization_error = e
        logger.error(
            "An unexpected error occurred during OpenGL initialization. "
            "The 3D canvas will be disabled. Error: %s",
            e,
            exc_info=True,
        )
        initialized = False


# Expose the main widget class from the package.
# We wrap this in a try-except to prevent a hard crash if canvas3d.py itself
# has an import error for GL, and define a dummy class so that type hinting
# or unconditional imports elsewhere don't break. The 'initialized' flag
# remains the definitive source of truth for canvas availability.
try:
    from .canvas3d import Canvas3D  # type: ignore
except Exception as e:
    logger.exception(
        "Failed to import Canvas3D. The 3D canvas will not be available."
    )
    initialization_error = f"Canvas3D import failed: {e}"

    class Canvas3D:
        """A placeholder class for when the 3D canvas fails to initialize."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "3D Canvas is not available due to an initialization failure. "
                "Check logs for details."
            )


__all__ = [
    "Canvas3D",
]
