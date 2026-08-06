from typing import Optional

from .unit_spin_row import UnitSpinRow


class SpeedSpinRow(UnitSpinRow):
    """Unit-aware spin row for the ``speed`` quantity (base mm/min)."""

    __gtype_name__ = "RayforgeSpeedSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        *,
        step_increment: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            title,
            subtitle,
            quantity="speed",
            step_increment=step_increment,
            **kwargs,
        )
