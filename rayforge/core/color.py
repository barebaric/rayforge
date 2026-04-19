import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# A fully resolved, render-ready RGBA color.
ColorRGBA = Tuple[float, float, float, float]

ColorAtom = Union[
    str, Tuple[float, float, float], Tuple[float, float, float, float]
]
ColorSpec = Union[ColorAtom, Tuple[ColorAtom, float]]
GradientSpec = Tuple[ColorSpec, ColorSpec]
ColorSpecDict = Dict[str, Union[ColorSpec, GradientSpec]]

OPS_COLOR_SPEC: ColorSpecDict = {
    "cut": ("#ffeeff", "#ff00ff"),
    "engrave": ("#FFFFFF", "#000000"),
    "travel": ("#FF6600", 0.7),
    "zero_power": ("@accent_color", 0.5),
}


def hex_to_rgba(hex_color: str) -> ColorRGBA:
    """Convert a hex color string to an RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b, 1.0)
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
        return (r, g, b, a)
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")


def create_lut_from_color(color: ColorRGBA) -> np.ndarray:
    """Create a 256x4 LUT from a single color (grayscale to color gradient)."""
    r, g, b, a = color
    lut = np.zeros((256, 4), dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        lut[i, 0] = r * t
        lut[i, 1] = g * t
        lut[i, 2] = b * t
        lut[i, 3] = a * t
    return lut


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


COLOR_PALETTE = [
    "#00ccff",
    "#ff6600",
    "#33cc33",
    "#ffcc00",
    "#cc3366",
    "#66cccc",
    "#ff9999",
    "#9966ff",
    "#00cc99",
]


def pick_unused_color(used_colors: set) -> str:
    """Return the first color from COLOR_PALETTE not in used_colors."""
    normalized = {c.upper() for c in used_colors}
    for color in COLOR_PALETTE:
        if color.upper() not in normalized:
            return color
    return COLOR_PALETTE[0]
