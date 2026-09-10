"""Availability gate for the raygeo knife transforms.

The knife transformers require raygeo features that newer releases
provide. The addon degrades gracefully when the installed raygeo
predates them: registration is skipped and the tests are ignored.
"""

import importlib

try:
    importlib.import_module("raygeo.ops.transform.drag_knife")
    importlib.import_module("raygeo.ops.transform.tangential_knife")
    KNIFE_TRANSFORMS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on installed raygeo
    KNIFE_TRANSFORMS_AVAILABLE = False
