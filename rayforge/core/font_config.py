from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import cairo
from gi.repository import Pango, PangoCairo


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

    def _create_pango_layout(
        self, ctx: Optional[cairo.Context] = None
    ) -> Tuple["Pango.Layout", cairo.Context]:
        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        ctx = ctx or cairo.Context(surface)
        layout = PangoCairo.create_layout(ctx)
        desc = Pango.FontDescription()
        desc.set_family(self.font_family)
        desc.set_size(int(self.font_size * Pango.SCALE))
        if self.bold:
            desc.set_weight(Pango.Weight.BOLD)
        if self.italic:
            desc.set_style(Pango.Style.ITALIC)
        layout.set_font_description(desc)
        return layout, ctx

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
        layout, _ = self._create_pango_layout()
        ctx = layout.get_context()
        desc = layout.get_font_description()
        metrics = ctx.get_metrics(desc)
        ascent = metrics.get_ascent() / Pango.SCALE
        descent = metrics.get_descent() / Pango.SCALE
        return ascent, -descent, ascent + descent

    def _scaled_layout(self, text: str):
        """
        Create a Pango layout at a larger size to avoid hinting,
        returning (layout, scale_factor).
        """
        render_scale = max(1.0, 100.0 / self.font_size)
        render_size = self.font_size * render_scale
        render_config = FontConfig(
            font_family=self.font_family,
            font_size=render_size,
            bold=self.bold,
            italic=self.italic,
        )
        layout, _ = render_config._create_pango_layout()
        layout.set_text(text, -1)
        return layout, render_scale

    def get_text_width(self, text: str) -> float:
        """
        Gets the width of the text including spaces.

        Uses Pango to compute the advance width, which properly
        accounts for kerning and whitespace characters.  Renders at
        a larger internal size to avoid hinting artifacts.

        Args:
            text: The string to measure.

        Returns:
            The width of the text in geometry units.
        """
        if not text:
            return 0.0
        layout, scale = self._scaled_layout(text)
        _, logical = layout.get_extents()
        return logical.width / Pango.SCALE / scale

    def get_text_position(self, text: str, index: int) -> float:
        """
        Get the x-position of a cursor at *index* within *text*,
        accounting for kerning.

        Uses Pango's ``get_cursor_pos`` at an enlarged internal
        size to avoid hinting artifacts.

        Args:
            text: The full string being laid out.
            index: Cursor position (0 = before the first character,
                   len(text) = after the last character).

        Returns:
            The x-coordinate of the cursor in the same coordinate
            space used by :meth:`get_text_width`.
        """
        if not text or index <= 0:
            return 0.0
        layout, scale = self._scaled_layout(text)
        strong, _ = layout.get_cursor_pos(index)
        return strong.x / Pango.SCALE / scale
