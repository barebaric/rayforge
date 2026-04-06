from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class AddonManifest:
    """
    Pre-computed addon information for fast worker initialization.

    This manifest is built once in the main process and transmitted to
    worker processes via the shared multiprocessing dict.
    """

    # Maps fully qualified module name to its absolute file path.
    # E.g. "rayforge_addons.my_addon.backend" -> "/path/to/my_addon/backend.py"
    module_paths: Dict[str, str] = field(default_factory=dict)

    # A set of all namespace packages that need to be created in sys.modules
    # for imports to work correctly.
    # E.g., {"rayforge_addons", "rayforge_addons.my_addon"}
    namespaces: Set[str] = field(default_factory=set)

    # A list of fully qualified module names for all ENABLED backend entry
    # points.
    # This tells the worker's lazy loader exactly which modules to import.
    # E.g., ["rayforge_addons.my_addon.backend"]
    enabled_backend_modules: List[str] = field(default_factory=list)
