from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import cairo


@dataclass
class FontConfig:
    """
    Serializable font configuration with forward compatibility.

    This class encapsulates font parameters used for text rendering
    and geometry generation. It supports serialization/deserialization
    and forward compatibility through the extra field.
    """

    font_family: str = "sans-serif"
    font_size: float = 10.0
    bold: bool = False
    italic: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the font configuration to a dictionary.

        Returns:
            A dictionary representation of the font configuration.
        """
        result: Dict[str, Any] = {
            "font_family": self.font_family,
            "font_size": self.font_size,
            "bold": self.bold,
            "italic": self.italic,
        }
        if self.extra:
            result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FontConfig":
        """
        Create a FontConfig instance from a dictionary.

        This method supports forward compatibility by storing any
        unknown fields in the extra dictionary.

        Args:
            data: The dictionary containing font configuration.

        Returns:
            A new FontConfig instance.
        """
        if data is None:
            return cls()

        known_keys = {"font_family", "font_size", "bold", "italic"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            font_family=data.get("font_family", "sans-serif"),
            font_size=float(data.get("font_size", 10.0)),
            bold=bool(data.get("bold", False)),
            italic=bool(data.get("italic", False)),
            extra=extra,
        )

    def copy(self) -> "FontConfig":
        """
        Create a deep copy of the font configuration.

        Returns:
            A new FontConfig instance with the same values.
        """
        return FontConfig(
            font_family=self.font_family,
            font_size=self.font_size,
            bold=self.bold,
            italic=self.italic,
            extra=self.extra.copy(),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FontConfig):
            return NotImplemented
        return (
            self.font_family == other.font_family
            and self.font_size == other.font_size
            and self.bold == other.bold
            and self.italic == other.italic
            and self.extra == other.extra
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.font_family,
                self.font_size,
                self.bold,
                self.italic,
                frozenset(self.extra.items()),
            )
        )

    def get_font_metrics(self) -> Tuple[float, float, float]:
        """
        Gets the font metrics for this font configuration.

        Returns a tuple of (ascent, descent, height) where:
        - ascent: distance from baseline to top of ascenders
        - descent: distance from baseline to bottom of descenders (negative)
        - height: total vertical extent (ascent - descent)

        Returns:
            A tuple (ascent, descent, height).
        """
        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        ctx = cairo.Context(surface)

        slant = (
            cairo.FONT_SLANT_ITALIC if self.italic else cairo.FONT_SLANT_NORMAL
        )
        weight = (
            cairo.FONT_WEIGHT_BOLD if self.bold else cairo.FONT_WEIGHT_NORMAL
        )
        ctx.select_font_face(self.font_family, slant, weight)
        ctx.set_font_size(self.font_size)

        ascent, descent, height, _, _ = ctx.font_extents()

        return ascent, descent, height

    def get_text_width(self, text: str) -> float:
        """
        Gets the width of the text including spaces.

        Unlike geometry-based measurement which ignores spaces, this method
        uses Cairo's text extents to get the actual advance width, which
        properly accounts for whitespace characters.

        Args:
            text: The string to measure.

        Returns:
            The width of the text in geometry units.
        """
        if not text:
            return 0.0

        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        ctx = cairo.Context(surface)

        slant = (
            cairo.FONT_SLANT_ITALIC if self.italic else cairo.FONT_SLANT_NORMAL
        )
        weight = (
            cairo.FONT_WEIGHT_BOLD if self.bold else cairo.FONT_WEIGHT_NORMAL
        )
        ctx.select_font_face(self.font_family, slant, weight)
        ctx.set_font_size(self.font_size)

        x_bearing, y_bearing, width, height, x_advance, y_advance = (
            ctx.text_extents(text)
        )

        return x_advance
