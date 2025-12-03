import logging
from typing import Dict, Tuple, Optional, Union, Any, TypeGuard, cast
import numpy as np
from gi.repository import Gdk, Gtk
from ..util.colors import ColorSet, ColorRGBA

logger = logging.getLogger(__name__)


# --- Type Definitions for GTK-side Color Specifications ---
ColorAtom = Union[
    str, Tuple[float, float, float], Tuple[float, float, float, float]
]
ColorSpec = Union[ColorAtom, Tuple[ColorAtom, float]]
GradientSpec = Tuple[ColorSpec, ColorSpec]
ColorSpecDict = Dict[str, Union[ColorSpec, GradientSpec]]


def _is_gradient_spec(val: Any) -> TypeGuard[GradientSpec]:
    """
    Checks if a value conforms to the GradientSpec type.

    A ColorSpec-with-alpha is `(ColorAtom, float)`, while a gradient is
    `(ColorSpec, ColorSpec)`. A ColorSpec itself is never a bare float/int,
    so checking the type of the second element is a reliable way to distinguish
    the two forms of 2-element tuples.
    """
    return (
        isinstance(val, tuple)
        and len(val) == 2
        and not isinstance(val[1], (int, float))
    )


def _is_spec_with_alpha(val: Any) -> TypeGuard[Tuple[ColorAtom, float]]:
    """
    Checks if a value conforms to the (ColorAtom, float) variant of ColorSpec.
    """
    return (
        isinstance(val, tuple)
        and len(val) == 2
        and isinstance(val[1], (float, int))
    )


class GtkColorResolver:
    """
    A GTK-specific resolver that converts a generic ColorSpecDict into a
    render-ready, UI-agnostic ColorSet using a Gtk.StyleContext.
    """

    def __init__(self, context: Gtk.StyleContext):
        self.context = context
        self._color_cache: Dict[ColorAtom, Gdk.RGBA] = {}

    def resolve(self, spec_dict: ColorSpecDict) -> ColorSet:
        """
        Performs the conversion from a specification dict to a resolved
        color set.
        """
        resolved_data = {}
        for name, spec in spec_dict.items():
            if _is_gradient_spec(spec):
                # The TypeGuard guarantees `spec` is GradientSpec here.
                resolved_data[name] = self._create_lut_from_gradient(spec)
            else:
                # The TypeGuard guarantees `spec` is not a GradientSpec,
                # thus it must be a ColorSpec.
                resolved_data[name] = self._resolve_color_spec_to_rgba(
                    cast(ColorSpec, spec)
                )

        return ColorSet(_data=resolved_data)

    def _create_lut_from_gradient(self, gradient: GradientSpec) -> np.ndarray:
        """Generates a 256x4 NumPy array (LUT) from a gradient spec."""
        start_spec, end_spec = gradient
        start_rgba = self._resolve_color_spec_to_rgba(start_spec)
        end_rgba = self._resolve_color_spec_to_rgba(end_spec)

        s = np.array(start_rgba, dtype=np.float32)
        e = np.array(end_rgba, dtype=np.float32)

        t = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, np.newaxis]
        lut = s * (1 - t) + e * t
        return lut

    def _resolve_color_spec_to_rgba(self, spec: ColorSpec) -> ColorRGBA:
        """
        Resolves a single ColorSpec into a concrete (r, g, b, a) tuple.
        """
        alpha_override: Optional[float] = None
        atom: ColorAtom
        if _is_spec_with_alpha(spec):  # Tuple[ColorAtom, float]
            atom, alpha_override = spec
        else:  # ColorAtom
            atom = cast(ColorAtom, spec)

        if atom in self._color_cache:
            rgba = self._color_cache[atom]
        elif isinstance(atom, str):
            if atom.startswith("@"):
                color_name = atom[1:]
                found, color = self.context.lookup_color(color_name)
                if not found:
                    logger.warning(f"Theme color '{color_name}' not found.")
                    color = Gdk.RGBA(red=1.0, green=0.0, blue=1.0, alpha=1.0)
            else:
                color = Gdk.RGBA()
                if not color.parse(atom):
                    logger.warning(f"Could not parse color string: '{atom}'")
                    color = Gdk.RGBA(red=1.0, green=0.0, blue=1.0, alpha=1.0)
            self._color_cache[atom] = color
            rgba = color
        elif isinstance(atom, tuple) and len(atom) in [3, 4]:
            # Gdk.RGBA expects floats from 0.0-1.0. If integers (0-255) are
            # provided, normalize them.
            r, g, b = atom[0], atom[1], atom[2]
            if isinstance(r, int):
                r, g, b = r / 255.0, g / 255.0, b / 255.0
                a = atom[3] / 255.0 if len(atom) == 4 else 1.0
            else:
                a = atom[3] if len(atom) == 4 else 1.0
            return (r, g, b, a)
        else:
            raise ValueError(f"Invalid ColorAtom: {atom}")

        return (
            rgba.red,
            rgba.green,
            rgba.blue,
            alpha_override if alpha_override is not None else rgba.alpha,
        )
