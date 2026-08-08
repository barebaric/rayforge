from .unit_spin_row import UnitSpinRow


class LengthSpinRow(UnitSpinRow):
    """Unit-aware spin row for the ``length`` quantity (base unit mm)."""

    __gtype_name__ = "RayforgeLengthSpinRow"

    def __init__(self, title: str, subtitle: str | None = None, **kwargs):
        super().__init__(title, subtitle, quantity="length", **kwargs)
