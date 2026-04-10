from enum import IntFlag


class Axis(IntFlag):
    """Enum for machine axes"""

    X = 1
    Y = 2
    Z = 4
    A = 8
    B = 16
    C = 32
    U = 64

    def assert_single_axis(self) -> None:
        if self.value.bit_count() != 1:
            raise ValueError(
                f"{self!r} combines multiple axes, expected a single axis"
            )
