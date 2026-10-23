from .unit_spin_row import UnitSpinRow


class LengthSpinRow(UnitSpinRow):
    """Unit-aware spin row for the ``length`` quantity (base unit mm)."""

    __gtype_name__ = "RayforgeLengthSpinRow"

    DEFAULT_STEP_IN_BASE = 0.1

    def __init__(
        self,
        title: str,
        subtitle: str | None = None,
        *,
        step_in_base: float | None = DEFAULT_STEP_IN_BASE,
        **kwargs,
    ):
        super().__init__(
            title,
            subtitle,
            quantity="length",
            step_in_base=step_in_base,
            **kwargs,
        )
