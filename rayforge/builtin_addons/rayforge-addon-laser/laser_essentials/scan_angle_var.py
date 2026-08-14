"""The raster scan-angle VarSet variable."""

from gettext import gettext as _

from rayforge.core.varset import SliderFloatVar


class ScanAngleVar(SliderFloatVar):
    """A SliderFloatVar for the raster scan angle in degrees.

    Hints the UI to render a slider whose suffix carries a
    :class:`DirectionPreview` visualizing the scan direction. The
    preview also reflects the ``cross_hatch`` sibling setting, which
    the adapter picks up via :meth:`update_from_values`.
    """

    def __init__(
        self,
        key: str = "scan_angle",
        label: str = _("Angle"),
        description: str | None = _("Angle of scan lines in degrees"),
        default: float = 0.0,
        value: float | None = None,
    ):
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            min_val=0.0,
            max_val=360.0,
            show_value=True,
            format_suffix="°",
        )
