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
        Returns a default magenta LUT if not found or invalid.
        """
        lut = self._data.get(name)
        if isinstance(lut, np.ndarray) and lut.shape == (256, 4):
            return lut

        logger.warning(
            f"LUT '{name}' not found or invalid in ColorSet. "
            f"Returning default."
        )
        # Create a magenta LUT to indicate a missing color
        default_lut = np.zeros((256, 4), dtype=np.float32)
        default_lut[:, 0] = 1.0  # R
        default_lut[:, 2] = 1.0  # B
        default_lut[:, 3] = 1.0  # A
        return default_lut

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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ColorSet to a dictionary."""
        serialized_data: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, np.ndarray):
                serialized_data[key] = {
                    "__type__": "numpy",
                    "data": value.tolist(),
                    "dtype": str(value.dtype),
                }
            else:
                serialized_data[key] = {"__type__": "tuple", "data": value}
        return {"_data": serialized_data}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorSet":
        """Deserializes a ColorSet from a dictionary."""
        deserialized_data: Dict[str, Any] = {}
        source_data = data.get("_data", data)  # Handle both formats
        for key, value in source_data.items():
            if isinstance(value, dict) and "__type__" in value:
                if value["__type__"] == "numpy":
                    deserialized_data[key] = np.array(
                        value["data"], dtype=value["dtype"]
                    )
                else:
                    deserialized_data[key] = tuple(value["data"])
            else:
                deserialized_data[key] = value  # Assume raw data for test
        return cls(_data=deserialized_data)
