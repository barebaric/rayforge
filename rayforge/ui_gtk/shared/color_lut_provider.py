"""
Shared colour LUT assembly for the renderers.

Builds the per-laser and fallback colour lookup tables consumed by the
3D ops, ring buffer, and texture renderers, so that the canvases no
longer assemble raw arrays themselves.
"""

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ...core.color import ColorSet
from ...image.util.srgb import create_lut_from_color
from ...machine.models.colors import OpsColorSet
from ...machine.models.laser import LaserHead

if TYPE_CHECKING:
    from ...machine.models.machine import Machine


class ColorLutProvider:
    """
    Provides colour LUTs for the power-based renderers.

    Encapsulates the per-laser ``ColorSet`` resolution from the machine
    and the assembly of the 1D/2D LUT arrays passed to the renderers'
    ``update_color_lut`` methods.
    """

    def __init__(
        self,
        color_set: ColorSet,
        laser_color_sets: Dict[str, ColorSet],
    ):
        self._color_set = color_set
        self._laser_color_sets = laser_color_sets

    @classmethod
    def from_machine(
        cls,
        machine: Optional["Machine"],
        color_set: ColorSet,
    ) -> "ColorLutProvider":
        """
        Build a provider from a machine's laser heads and a theme ColorSet.
        """
        laser_color_sets: Dict[str, ColorSet] = {}
        if machine is not None:
            for laser in machine.heads:
                if not isinstance(laser, LaserHead):
                    continue
                laser_color_set = OpsColorSet.from_laser(laser, color_set)
                laser_color_sets[laser.uid] = laser_color_set.to_color_set()
        return cls(color_set, laser_color_sets)

    @property
    def color_set(self) -> ColorSet:
        """The resolved base theme ColorSet."""
        return self._color_set

    @property
    def laser_color_sets(self) -> Dict[str, ColorSet]:
        """Per-laser colour sets keyed by laser UID."""
        return self._laser_color_sets

    @property
    def has_lasers(self) -> bool:
        """True if per-laser colour sets have been resolved."""
        return bool(self._laser_color_sets)

    @property
    def num_lasers(self) -> int:
        """Number of resolved lasers (at least 1)."""
        return len(self._laser_color_sets) or 1

    def cut_lut(self) -> np.ndarray:
        """LUT for cut/engraved lines, dimmed by power."""
        if self.has_lasers:
            lut = np.zeros((self.num_lasers, 256, 4), dtype=np.float32)
            for row_idx, uid in enumerate(self._laser_color_sets):
                lut[row_idx] = self._laser_color_sets[uid].get_lut("cut")
            return lut
        return create_lut_from_color(self._color_set.get_rgba("cut"))

    def engrave_lut_2d(self) -> np.ndarray:
        """LUT for texture/engrave rendering."""
        if self.has_lasers:
            lut = np.zeros((self.num_lasers, 256, 4), dtype=np.float32)
            for row_idx, uid in enumerate(self._laser_color_sets):
                lut[row_idx] = self._laser_color_sets[uid].get_lut("engrave")
            return lut
        return self._color_set.get_lut("engrave")

    def ring_lut_2d(self) -> np.ndarray:
        """
        LUT for the scanline overlay ring buffer.

        The overlay dims by power too, so each laser gets a brightness
        ramp rather than a flat colour.
        """
        if self.has_lasers:
            lut = np.zeros((self.num_lasers, 256, 4), dtype=np.float32)
            for row_idx, uid in enumerate(self._laser_color_sets):
                cs = self._laser_color_sets[uid]
                engrave_rgba = tuple(cs.get_lut("engrave")[255])
                lut[row_idx] = create_lut_from_color(engrave_rgba)
            return lut
        return create_lut_from_color((1.0, 1.0, 1.0, 1.0))
