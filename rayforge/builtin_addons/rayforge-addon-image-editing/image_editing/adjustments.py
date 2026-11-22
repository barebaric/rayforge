from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rayforge.core.image_edit import ImageEditStep

from .ops import SharpenOp, ToneOp


@dataclass(frozen=True)
class ImageAdjustments:
    """
    The values of the Adjust Image dialog. They map to a "sharpen" and a
    "tone" edit; neutral values mean the edit is not needed.
    """

    brightness: float = 0.0
    contrast: float = 0.0
    gamma: float = 1.0
    sharpen_amount: float = 0.0
    sharpen_radius_mm: float = 0.2

    @property
    def has_tone(self) -> bool:
        return (
            self.brightness != 0.0 or self.contrast != 0.0 or self.gamma != 1.0
        )

    @property
    def has_sharpen(self) -> bool:
        return self.sharpen_amount > 0.0 and self.sharpen_radius_mm > 0.0

    def to_edits(self) -> dict[str, ImageEditStep | None]:
        """
        The edits by op name, sharpen first so the tone curve works on
        the sharpened image; ``None`` removes that op's edit.
        """
        sharpen = ImageEditStep(
            SharpenOp.NAME,
            {
                "amount": self.sharpen_amount,
                "radius_mm": self.sharpen_radius_mm,
            },
        )
        tone = ImageEditStep(
            ToneOp.NAME,
            {
                "brightness": self.brightness,
                "contrast": self.contrast,
                "gamma": self.gamma,
            },
        )
        return {
            SharpenOp.NAME: sharpen if self.has_sharpen else None,
            ToneOp.NAME: tone if self.has_tone else None,
        }

    @classmethod
    def from_edits(cls, edits: Iterable[ImageEditStep]) -> ImageAdjustments:
        """Reads the values back from a workpiece's edit stack."""
        defaults = cls()
        values = {}
        for edit in edits:
            if edit.op == ToneOp.NAME:
                values["brightness"] = edit.params.get("brightness", 0.0)
                values["contrast"] = edit.params.get("contrast", 0.0)
                values["gamma"] = edit.params.get("gamma", 1.0)
            elif edit.op == SharpenOp.NAME:
                values["sharpen_amount"] = edit.params.get("amount", 0.0)
                values["sharpen_radius_mm"] = edit.params.get(
                    "radius_mm", defaults.sharpen_radius_mm
                )
        return cls(**values)
