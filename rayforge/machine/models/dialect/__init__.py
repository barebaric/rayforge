from dataclasses import replace

from .base import GcodeDialect
from .grbl import GRBL_DIALECT
from .grbl_dynamic import GRBL_DYNAMIC_DIALECT
from .grbl_raster import GRBL_RASTER_DIALECT
from .linuxcnc import LINUXCNC_DIALECT
from .mach4_m67 import MACH4_M67_DIALECT
from .marlin import MARLIN_DIALECT
from .smoothieware import SMOOTHIEWARE_DIALECT

BUILTIN_DIALECTS = [
    GRBL_DIALECT,
    GRBL_DYNAMIC_DIALECT,
    GRBL_RASTER_DIALECT,
    LINUXCNC_DIALECT,
    MACH4_M67_DIALECT,
    MARLIN_DIALECT,
    SMOOTHIEWARE_DIALECT,
]

__all__ = [
    "GcodeDialect",
    "replace",
    "GRBL_DIALECT",
    "GRBL_DYNAMIC_DIALECT",
    "GRBL_RASTER_DIALECT",
    "LINUXCNC_DIALECT",
    "MACH4_M67_DIALECT",
    "MARLIN_DIALECT",
    "SMOOTHIEWARE_DIALECT",
    "BUILTIN_DIALECTS",
]
