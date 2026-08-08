from .unit_spin_row import UnitSpinRow


class AccelerationSpinRow(UnitSpinRow):
    """Unit-aware spin row for ``acceleration`` (base mm/s^2)."""

    __gtype_name__ = "RayforgeAccelerationSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: str | None = None,
        *,
        step_increment: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            title,
            subtitle,
            quantity="acceleration",
            step_increment=step_increment,
            **kwargs,
        )
