from .base import SpinRow


class AngleSpinRow(SpinRow):
    """
    A spin row for angle values in degrees.

    Builds on :class:`SpinRow` with degree-appropriate defaults: a full
    rotation range of -360..360 degrees, whole-degree stepping and one
    decimal place. Pass ``lower``/``upper``/``digits`` to override
    (e.g. a 0..180 half-turn or an integer-degree field).
    """

    __gtype_name__ = "RayforgeAngleSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: str | None = None,
        *,
        lower: float = -360.0,
        upper: float = 360.0,
        digits: int = 1,
        **kwargs,
    ):
        super().__init__(
            title,
            subtitle,
            lower=lower,
            upper=upper,
            digits=digits,
            **kwargs,
        )
