"""Dialogs that create an Interval Test or a Focus Test."""

from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from gi.repository import Adw, Gtk

from rayforge.core.varset import (
    BoolVar,
    IntVar,
    LabeledChoiceVar,
    LengthVar,
    SliderFloatVar,
    SpeedVar,
    VarSet,
)
from rayforge.ui_gtk.shared.patched_dialog_window import PatchedDialogWindow
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget

from ..calibration_tests import (
    MAX_FOCUS_OFFSET_MM,
    FocusMode,
    FocusTestParams,
    IntervalTestParams,
)

DEFAULT_WIDTH = 600

_FOCUS_HELP = {
    FocusMode.Z_AXIS: _(
        "Focus the laser as usual first. The job moves the head to the "
        "first offset, engraves one line per offset and returns the head "
        "to the starting height. A positive offset means more distance "
        "between the head and the material."
    ),
    FocusMode.MANUAL: _(
        "Focus the laser as usual first. The labels are engraved at that "
        "height. Before each line the job pauses: set the head to the "
        "first offset, then move it by one step at every following "
        "pause, and press Resume. A positive offset means more distance "
        "between the head and the material."
    ),
    FocusMode.RAMP: _(
        "Prop up one end of a flat strip so it rises evenly under the "
        "line. The thinnest part of the line marks the focus; the ticks "
        "show the distance along the line, so the height there follows "
        "from the rise."
    ),
}


def _labels_active(values: dict[str, Any]) -> bool:
    return bool(values.get("include_labels"))


def _is_ramp(values: dict[str, Any]) -> bool:
    return values.get("mode") == FocusMode.RAMP.value


def _is_not_ramp(values: dict[str, Any]) -> bool:
    return not _is_ramp(values)


def _laser_varset(defaults) -> VarSet:
    return VarSet(
        title=_("Laser"),
        vars=[
            SliderFloatVar(
                key="power_percent",
                label=_("Power"),
                default=defaults.power_percent,
                min_val=0.0,
                max_val=100.0,
                show_value=True,
                format_suffix="%",
            ),
            SpeedVar(
                key="speed",
                label=_("Speed"),
                default=int(defaults.speed),
                min_val=1,
            ),
        ],
    )


def _labels_varset(defaults) -> VarSet:
    return VarSet(
        title=_("Labels"),
        vars=[
            BoolVar(
                key="include_labels",
                label=_("Include Labels"),
                default=defaults.include_labels,
            ),
            SliderFloatVar(
                key="label_power_percent",
                label=_("Label Engrave Power (%)"),
                default=defaults.label_power_percent,
                min_val=0.0,
                max_val=100.0,
                show_value=True,
                format_suffix="%",
                sensitive_when=_labels_active,
            ),
            SpeedVar(
                key="label_speed",
                label=_("Label Engrave Speed"),
                default=int(defaults.label_speed),
                min_val=1,
                sensitive_when=_labels_active,
            ),
        ],
    )


def _interval_varset(defaults: IntervalTestParams) -> VarSet:
    return VarSet(
        title=_("Cells"),
        description=_(
            "Engraves a row of filled squares, each with its own line "
            "interval, at the same power and speed."
        ),
        vars=[
            IntVar(
                key="count",
                label=_("Number of Cells"),
                default=defaults.count,
                min_val=2,
                max_val=20,
            ),
            LengthVar(
                key="min_interval",
                label=_("Smallest Interval"),
                default=defaults.min_interval,
                min_val=0.01,
                max_val=5.0,
                digits=3,
            ),
            LengthVar(
                key="max_interval",
                label=_("Largest Interval"),
                default=defaults.max_interval,
                min_val=0.01,
                max_val=5.0,
                digits=3,
            ),
            LengthVar(
                key="cell_size",
                label=_("Cell Size"),
                default=defaults.cell_size,
                min_val=1.0,
                max_val=100.0,
            ),
            LengthVar(
                key="spacing",
                label=_("Spacing"),
                default=defaults.spacing,
                min_val=0.0,
                max_val=50.0,
            ),
        ],
    )


def _focus_varset(defaults: FocusTestParams, modes: list[FocusMode]) -> VarSet:
    limit = MAX_FOCUS_OFFSET_MM
    return VarSet(
        title=_("Focus Steps"),
        vars=[
            LabeledChoiceVar(
                key="mode",
                label=_("Method"),
                choices=[(m.label, m.value) for m in modes],
                default=modes[0].value,
                allow_none=False,
            ),
            LengthVar(
                key="start_offset",
                label=_("First Offset"),
                default=defaults.start_offset,
                min_val=-limit,
                max_val=limit,
                digits=2,
                visible_when=_is_not_ramp,
            ),
            LengthVar(
                key="step",
                label=_("Step"),
                default=defaults.step,
                min_val=-limit,
                max_val=limit,
                digits=2,
                visible_when=_is_not_ramp,
            ),
            IntVar(
                key="count",
                label=_("Number of Lines"),
                default=defaults.count,
                min_val=2,
                max_val=41,
                visible_when=_is_not_ramp,
            ),
            LengthVar(
                key="line_length",
                label=_("Line Length"),
                default=defaults.line_length,
                min_val=1.0,
                max_val=200.0,
                visible_when=_is_not_ramp,
            ),
            LengthVar(
                key="spacing",
                label=_("Spacing"),
                default=defaults.spacing,
                min_val=0.5,
                max_val=50.0,
                visible_when=_is_not_ramp,
            ),
            LengthVar(
                key="ramp_length",
                label=_("Ramp Length"),
                default=defaults.ramp_length,
                min_val=10.0,
                max_val=1000.0,
                visible_when=_is_ramp,
            ),
        ],
    )


class _CalibrationTestDialog(PatchedDialogWindow):
    """Shared frame: a header with Cancel/Create and one page."""

    def __init__(
        self,
        parent: Gtk.Window | None,
        title: str,
        on_create: Callable[[Any], object],
        varsets: list[VarSet],
    ):
        super().__init__(transient_for=parent, modal=True)
        self.set_title(title)
        self.set_default_size(DEFAULT_WIDTH, -1)
        self._on_create = on_create

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)
        self.set_content(toolbar)

        cancel = Gtk.Button(label=_("Cancel"))
        cancel.connect("clicked", lambda _w: self.close())
        header.pack_start(cancel)
        self.create_button = Gtk.Button(label=_("Create"))
        self.create_button.add_css_class("suggested-action")
        self.create_button.connect("clicked", self._on_create_clicked)
        header.pack_end(self.create_button)

        self.banner = Adw.Banner()
        toolbar.add_top_bar(self.banner)
        page = Adw.PreferencesPage()
        toolbar.set_content(page)

        self.varset_widgets: list[VarSetWidget] = []
        for var_set in varsets:
            widget = VarSetWidget()
            widget.populate(var_set)
            page.add(widget)
            self.varset_widgets.append(widget)

    def values(self) -> dict[str, Any]:
        """The values of every row, keyed like the params fields."""
        values: dict[str, Any] = {}
        for widget in self.varset_widgets:
            values.update(widget.get_values())
        return values

    def params(self) -> object:
        raise NotImplementedError

    def _on_create_clicked(self, _button):
        try:
            self._on_create(self.params())
        except ValueError as e:
            self.banner.set_title(str(e))
            self.banner.set_revealed(True)
            return
        self.close()


class IntervalTestDialog(_CalibrationTestDialog):
    """Asks for the settings of an Interval Test."""

    def __init__(
        self,
        parent: Gtk.Window | None,
        on_create: Callable[[IntervalTestParams], object],
    ):
        defaults = IntervalTestParams()
        super().__init__(
            parent,
            _("Create Interval Test"),
            on_create,
            [
                _interval_varset(defaults),
                _laser_varset(defaults),
                _labels_varset(defaults),
            ],
        )

    def params(self) -> IntervalTestParams:
        return IntervalTestParams(**self.values())


class FocusTestDialog(_CalibrationTestDialog):
    """Asks for the settings of a Focus Test."""

    def __init__(
        self,
        parent: Gtk.Window | None,
        on_create: Callable[[FocusTestParams], object],
        modes: list[FocusMode],
    ):
        defaults = FocusTestParams()
        super().__init__(
            parent,
            _("Create Focus Test"),
            on_create,
            [
                _focus_varset(defaults, modes),
                _laser_varset(defaults),
                _labels_varset(defaults),
            ],
        )
        self.steps_widget = self.varset_widgets[0]
        self.steps_widget.data_changed.connect(self._on_steps_changed)
        self._update_help()

    @property
    def mode(self) -> FocusMode:
        return FocusMode(self.steps_widget.get_values()["mode"])

    def _on_steps_changed(self, sender, key: str):
        if key == "mode":
            self._update_help()

    def _update_help(self):
        self.steps_widget.set_description(_FOCUS_HELP[self.mode])

    def params(self) -> FocusTestParams:
        values = self.values()
        values["mode"] = FocusMode(values["mode"])
        return FocusTestParams(**values)
