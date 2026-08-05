from enum import Enum


class UnitSystem(Enum):
    """
    The unit system a machine operates in.

    All internal values in Rayforge are stored in millimeters
    (the application base unit). ``UnitSystem`` describes the unit
    system of the *machine* — the system used to communicate with
    the device and to emit G-code. Conversion to and from the base
    unit happens only at driver/encoder boundaries.
    """

    METRIC = "metric"
    IMPERIAL = "imperial"

    @property
    def scale_from_mm(self) -> float:
        """Multiplier to convert a millimeter value into this unit
        system. Metric is ``1.0``; imperial is ``1/25.4``."""
        return 1.0 if self is UnitSystem.METRIC else 1.0 / 25.4
