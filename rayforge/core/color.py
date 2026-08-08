import logging
import re
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

# A fully resolved, render-ready RGBA color.
ColorRGBA = tuple[float, float, float, float]

ColorAtom = Union[
    str, tuple[float, float, float], tuple[float, float, float, float]
]
ColorSpec = Union[ColorAtom, tuple[ColorAtom, float]]
GradientSpec = tuple[ColorSpec, ColorSpec]
ColorSpecDict = dict[str, ColorSpec | GradientSpec]

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


_CSS_NAMED_COLORS: dict[str, str] = {
    "aliceblue": "f0f8ff",
    "antiquewhite": "faebd7",
    "aqua": "00ffff",
    "aquamarine": "7fffd4",
    "azure": "f0ffff",
    "beige": "f5f5dc",
    "bisque": "ffe4c4",
    "black": "000000",
    "blanchedalmond": "ffebcd",
    "blue": "0000ff",
    "blueviolet": "8a2be2",
    "brown": "a52a2a",
    "burlywood": "deb887",
    "cadetblue": "5f9ea0",
    "chartreuse": "7fff00",
    "chocolate": "d2691e",
    "coral": "ff7f50",
    "cornflowerblue": "6495ed",
    "cornsilk": "fff8dc",
    "crimson": "dc143c",
    "cyan": "00ffff",
    "darkblue": "00008b",
    "darkcyan": "008b8b",
    "darkgoldenrod": "b8860b",
    "darkgray": "a9a9a9",
    "darkgreen": "006400",
    "darkgrey": "a9a9a9",
    "darkkhaki": "bdb76b",
    "darkmagenta": "8b008b",
    "darkolivegreen": "556b2f",
    "darkorange": "ff8c00",
    "darkorchid": "9932cc",
    "darkred": "8b0000",
    "darksalmon": "e9967a",
    "darkseagreen": "8fbc8f",
    "darkslateblue": "483d8b",
    "darkslategray": "2f4f4f",
    "darkslategrey": "2f4f4f",
    "darkturquoise": "00ced1",
    "darkviolet": "9400d3",
    "deeppink": "ff1493",
    "deepskyblue": "00bfff",
    "dimgray": "696969",
    "dimgrey": "696969",
    "dodgerblue": "1e90ff",
    "firebrick": "b22222",
    "floralwhite": "fffaf0",
    "forestgreen": "228b22",
    "fuchsia": "ff00ff",
    "gainsboro": "dcdcdc",
    "ghostwhite": "f8f8ff",
    "gold": "ffd700",
    "goldenrod": "daa520",
    "gray": "808080",
    "grey": "808080",
    "green": "008000",
    "greenyellow": "adff2f",
    "honeydew": "f0fff0",
    "hotpink": "ff69b4",
    "indianred": "cd5c5c",
    "indigo": "4b0082",
    "ivory": "fffff0",
    "khaki": "f0e68c",
    "lavender": "e6e6fa",
    "lavenderblush": "fff0f5",
    "lawngreen": "7cfc00",
    "lemonchiffon": "fffacd",
    "lightblue": "add8e6",
    "lightcoral": "f08080",
    "lightcyan": "e0ffff",
    "lightgoldenrodyellow": "fafad2",
    "lightgray": "d3d3d3",
    "lightgreen": "90ee90",
    "lightgrey": "d3d3d3",
    "lightpink": "ffb6c1",
    "lightsalmon": "ffa07a",
    "lightseagreen": "20b2aa",
    "lightskyblue": "87cefa",
    "lightslategray": "778899",
    "lightslategrey": "778899",
    "lightsteelblue": "b0c4de",
    "lightyellow": "ffffe0",
    "lime": "00ff00",
    "limegreen": "32cd32",
    "linen": "faf0e6",
    "magenta": "ff00ff",
    "maroon": "800000",
    "mediumaquamarine": "66cdaa",
    "mediumblue": "0000cd",
    "mediumorchid": "ba55d3",
    "mediumpurple": "9370db",
    "mediumseagreen": "3cb371",
    "mediumslateblue": "7b68ee",
    "mediumspringgreen": "00fa9a",
    "mediumturquoise": "48d1cc",
    "mediumvioletred": "c71585",
    "midnightblue": "191970",
    "mintcream": "f5fffa",
    "mistyrose": "ffe4e1",
    "moccasin": "ffe4b5",
    "navajowhite": "ffdead",
    "navy": "000080",
    "oldlace": "fdf5e6",
    "olive": "808000",
    "olivedrab": "6b8e23",
    "orange": "ffa500",
    "orangered": "ff4500",
    "orchid": "da70d6",
    "palegoldenrod": "eee8aa",
    "palegreen": "98fb98",
    "paleturquoise": "afeeee",
    "palevioletred": "db7093",
    "papayawhip": "ffefd5",
    "peachpuff": "ffdab9",
    "peru": "cd853f",
    "pink": "ffc0cb",
    "plum": "dda0dd",
    "powderblue": "b0e0e6",
    "purple": "800080",
    "rebeccapurple": "663399",
    "red": "ff0000",
    "rosybrown": "bc8f8f",
    "royalblue": "4169e1",
    "saddlebrown": "8b4513",
    "salmon": "fa8072",
    "sandybrown": "f4a460",
    "seagreen": "2e8b57",
    "seashell": "fff5ee",
    "sienna": "a0522d",
    "silver": "c0c0c0",
    "skyblue": "87ceeb",
    "slateblue": "6a5acd",
    "slategray": "708090",
    "slategrey": "708090",
    "snow": "fffafa",
    "springgreen": "00ff7f",
    "steelblue": "4682b4",
    "tan": "d2b48c",
    "teal": "008080",
    "thistle": "d8bfd8",
    "tomato": "ff6347",
    "turquoise": "40e0d0",
    "violet": "ee82ee",
    "wheat": "f5deb3",
    "white": "ffffff",
    "whitesmoke": "f5f5f5",
    "yellow": "ffff00",
    "yellowgreen": "9acd32",
}

_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_RGB_RE = re.compile(
    r"^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*[\d.]+)?\s*\)$"
)


def normalize_color(color: str | None) -> str | None:
    """
    Normalize a color string to a canonical lowercase 6-digit hex value.

    Accepts ``#rrggbb``, ``#RGB``, ``#rrggbbaa`` (alpha dropped), CSS
    color names and ``rgb(...)`` strings.

    Args:
        color: The color string to normalize.

    Returns:
        The normalized ``#rrggbb`` value, or ``None`` for empty or
        unresolvable input.
    """
    if not color:
        return None
    value = color.strip()
    if not value:
        return None

    m = _HEX_RE.match(value)
    if m:
        h = m.group(1).lower()
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return f"#{h[:6]}"

    m = _RGB_RE.match(value)
    if m:
        r, g, b = (min(int(m.group(i)), 255) for i in (1, 2, 3))
        return f"#{r:02x}{g:02x}{b:02x}"

    name = value.lower().replace(" ", "")
    hexval = _CSS_NAMED_COLORS.get(name)
    if hexval is not None:
        return f"#{hexval}"

    return None


@dataclass(frozen=True)
class ColorSet:
    """
    A generic, UI-agnostic container for resolved, render-ready color data.
    It holds pre-calculated lookup tables (LUTs) and RGBA tuples, accessed by
    name.

    This object is immutable and thread-safe.
    """

    _data: dict[str, Any] = field(default_factory=dict)

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

    def get_lut_argb32(self, name: str) -> np.ndarray:
        """
        Gets the named LUT as a 256×4 ``np.uint8`` array in **pre-multiplied
        ARGB32** order (``[B×α, G×α, R×α, A]``), ready for
        :class:`~raygeo.ops.convert.ViewSpec`.
        """
        lut = self.get_lut(name)
        argb32 = np.empty((256, 4), dtype=np.uint8)
        argb32[:, 0] = np.clip(lut[:, 2] * lut[:, 3] * 255 + 0.5, 0, 255)
        argb32[:, 1] = np.clip(lut[:, 1] * lut[:, 3] * 255 + 0.5, 0, 255)
        argb32[:, 2] = np.clip(lut[:, 0] * lut[:, 3] * 255 + 0.5, 0, 255)
        argb32[:, 3] = np.clip(lut[:, 3] * 255 + 0.5, 0, 255)
        return argb32

    def get_rgba(self, name: str) -> ColorRGBA:
        """
        Gets a resolved RGBA color tuple by name.
        Returns a default magenta color if the name is not found.
        """
        rgba = self._data.get(name)
        if isinstance(rgba, tuple) and len(rgba) == 4:
            return rgba
        if isinstance(rgba, np.ndarray) and rgba.shape == (256, 4):
            return tuple(rgba[255])

        logger.warning(
            f"RGBA color '{name}' not found or invalid in ColorSet. "
            f"Returning default."
        )
        return 1.0, 0.0, 1.0, 1.0

    def get_argb32(self, name: str) -> list:
        """
        Gets a named colour as a 4-element byte list in **pre-multiplied
        ARGB32** order (``[B×α, G×α, R×α, A]``), ready for
        :class:`~raygeo.ops.convert.ViewSpec`.

        Returns a magenta fallback when the name is missing.
        """
        r, g, b, a = self.get_rgba(name)
        return [
            round(b * a * 255),
            round(g * a * 255),
            round(r * a * 255),
            round(a * 255),
        ]

    def __repr__(self) -> str:
        keys = sorted(self._data.keys())
        return f"ColorSet(keys={keys})"

    def to_dict(self) -> dict[str, Any]:
        """Serializes the ColorSet to a dictionary."""
        serialized_data: dict[str, Any] = {}
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
    def from_dict(cls, data: dict[str, Any]) -> "ColorSet":
        """Deserializes a ColorSet from a dictionary."""
        deserialized_data: dict[str, Any] = {}
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
