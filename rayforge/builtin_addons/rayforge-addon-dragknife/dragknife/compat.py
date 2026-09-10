"""Availability gate for the raygeo knife transforms.

The knife transformers require raygeo features that newer releases
provide. The addon degrades gracefully when the installed raygeo
predates them: registration is skipped and the tests are skipped.
"""

try:
    from raygeo.ops.transform import (  # noqa: F401 # pyright: ignore[reportMissingImports]
        drag_knife as _drag_knife,
    )
    from raygeo.ops.transform import (  # noqa: F401 # pyright: ignore[reportMissingImports]
        tangential_knife as _tangential_knife,
    )

    KNIFE_TRANSFORMS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on installed raygeo
    KNIFE_TRANSFORMS_AVAILABLE = False
