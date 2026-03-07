from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING
import numpy as np

from ...shared.util.colors import (
    ColorSet,
    ColorRGBA,
    hex_to_rgba,
    create_lut_from_color,
)

if TYPE_CHECKING:
    from .laser import Laser


@dataclass(frozen=True)
class OpsColorSet:
    """
    ColorSet for a specific laser's operations.

    This class bridges the Laser model's color settings with the rendering
    pipeline's ColorSet format.
    """

    laser_uid: str
    cut_lut: np.ndarray
    engrave_lut: np.ndarray
    travel_rgba: ColorRGBA
    zero_power_rgba: ColorRGBA

    @classmethod
    def from_laser(
        cls,
        laser: "Laser",
        theme_colors: ColorSet,
    ) -> "OpsColorSet":
        """
        Create an OpsColorSet from a Laser model and theme colors.

        Args:
            laser: The Laser model with cut_color and raster_color properties
            theme_colors: The theme's ColorSet for travel and zero_power colors

        Returns:
            An OpsColorSet with laser-specific cut/raster LUTs
        """
        cut_rgba = hex_to_rgba(laser.cut_color)
        raster_rgba = hex_to_rgba(laser.raster_color)

        cut_lut = create_lut_from_color(cut_rgba)
        engrave_lut = create_lut_from_color(raster_rgba)

        travel_rgba = theme_colors.get_rgba("travel")
        zero_power_rgba = theme_colors.get_rgba("zero_power")

        return cls(
            laser_uid=laser.uid,
            cut_lut=cut_lut,
            engrave_lut=engrave_lut,
            travel_rgba=travel_rgba,
            zero_power_rgba=zero_power_rgba,
        )

    def to_color_set(self) -> ColorSet:
        """
        Convert to a standard ColorSet for use in the rendering pipeline.

        Returns:
            A ColorSet with cut, engrave, travel, and zero_power entries
        """
        data: Dict[str, Any] = {
            "cut": self.cut_lut,
            "engrave": self.engrave_lut,
            "travel": self.travel_rgba,
            "zero_power": self.zero_power_rgba,
        }
        return ColorSet(_data=data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the OpsColorSet to a dictionary."""
        return {
            "laser_uid": self.laser_uid,
            "cut_lut": {
                "__type__": "numpy",
                "data": self.cut_lut.tolist(),
                "dtype": str(self.cut_lut.dtype),
            },
            "engrave_lut": {
                "__type__": "numpy",
                "data": self.engrave_lut.tolist(),
                "dtype": str(self.engrave_lut.dtype),
            },
            "travel_rgba": {
                "__type__": "tuple",
                "data": self.travel_rgba,
            },
            "zero_power_rgba": {
                "__type__": "tuple",
                "data": self.zero_power_rgba,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpsColorSet":
        """Deserialize an OpsColorSet from a dictionary."""
        return cls(
            laser_uid=data["laser_uid"],
            cut_lut=np.array(
                data["cut_lut"]["data"], dtype=data["cut_lut"]["dtype"]
            ),
            engrave_lut=np.array(
                data["engrave_lut"]["data"], dtype=data["engrave_lut"]["dtype"]
            ),
            travel_rgba=tuple(data["travel_rgba"]["data"]),
            zero_power_rgba=tuple(data["zero_power_rgba"]["data"]),
        )
