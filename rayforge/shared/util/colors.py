import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# A fully resolved, render-ready RGBA color.
ColorRGBA = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ColorSet:
    """
    A generic, UI-agnostic container for resolved, render-ready color data.
    It holds pre-calculated lookup tables (LUTs) and RGBA tuples, accessed by
    name.

    This object is immutable and thread-safe.
    """

    _data: Dict[str, Any] = field(default_factory=dict)

    def get_lut(self, name: str) -> np.ndarray:
        """
        Gets a pre-calculated 256x4 color lookup table (LUT) by name.
        Raises a ValueError if the LUT is not found or is invalid.
        """
        lut = self._data.get(name)
        if isinstance(lut, np.ndarray) and lut.shape == (256, 4):
            return lut

        raise ValueError(f"LUT '{name}' not found or invalid.")

    def get_rgba(self, name: str) -> ColorRGBA:
        """
        Gets a resolved RGBA color tuple by name.
        Returns a default magenta color if the name is not found.
        """
        rgba = self._data.get(name)
        if isinstance(rgba, tuple) and len(rgba) == 4:
            return rgba

        logger.warning(
            f"RGBA color '{name}' not found or invalid in ColorSet. "
            f"Returning default."
        )
        return 1.0, 0.0, 1.0, 1.0

    def __repr__(self) -> str:
        keys = sorted(self._data.keys())
        return f"ColorSet(keys={keys})"
