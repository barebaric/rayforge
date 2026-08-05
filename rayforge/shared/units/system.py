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
