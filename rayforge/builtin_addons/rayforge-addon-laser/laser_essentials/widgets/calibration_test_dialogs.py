"""Dialogs that create an Interval Test or a Focus Test."""

from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from gi.repository import Adw, Gtk

from rayforge.ui_gtk.shared.patched_dialog_window import PatchedDialogWindow
from rayforge.ui_gtk.shared.pref_rows import (
    LengthSpinRow,
    SpeedSpinRow,
    SpinRow,
)

from ..calibration_tests import (
    MAX_FOCUS_OFFSET_MM,
    FocusMode,
    FocusTestParams,
    IntervalTestParams,
)

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


class _CalibrationTestDialog(PatchedDialogWindow):
    """Shared frame: a header with Cancel/Create and one page."""

    def __init__(
        self,
        parent: Gtk.Window | None,
        title: str,
        on_create: Callable[[Any], object],
    ):
        super().__init__(transient_for=parent, modal=True)
        self.set_title(title)
        self.set_default_size(520, 640)
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
        self.page = Adw.PreferencesPage()
        toolbar.set_content(self.page)

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

    @staticmethod
    def _power_row(title: str, value: float) -> SpinRow:
        return SpinRow(
            title,
            _("Percent of the maximum power"),
            lower=0.1,
            upper=100.0,
            step_increment=1.0,
            digits=1,
            value=value,
        )

    def _add_label_group(self, defaults) -> None:
        group = Adw.PreferencesGroup(title=_("Labels"))
        self.labels_row = Adw.SwitchRow(title=_("Include Labels"))
        self.labels_row.set_active(defaults.include_labels)
        group.add(self.labels_row)
        self.label_power_row = self._power_row(
            _("Label Power"), defaults.label_power_percent
        )
        group.add(self.label_power_row)
        self.label_speed_row = SpeedSpinRow(
            _("Label Speed"),
            lower=1,
            upper=60000,
            value_in_base=defaults.label_speed,
        )
        group.add(self.label_speed_row)
        self.labels_row.connect(
            "notify::active", lambda *_a: self._sync_label_rows()
        )
        self._sync_label_rows()
        self.page.add(group)

    def _sync_label_rows(self):
        active = self.labels_row.get_active()
        self.label_power_row.set_sensitive(active)
        self.label_speed_row.set_sensitive(active)


class IntervalTestDialog(_CalibrationTestDialog):
    """Asks for the settings of an Interval Test."""

    def __init__(
        self,
        parent: Gtk.Window | None,
        on_create: Callable[[IntervalTestParams], object],
        max_speed: float = 60000.0,
    ):
        super().__init__(parent, _("Create Interval Test"), on_create)
        d = IntervalTestParams()
        group = Adw.PreferencesGroup(
            title=_("Cells"),
            description=_(
                "Engraves a row of filled squares, each with its own line "
                "interval, at the same power and speed."
            ),
        )
        self.count_row = SpinRow(
            _("Number of Cells"), lower=2, upper=20, value=d.count
        )
        self.min_row = LengthSpinRow(
            _("Smallest Interval"),
            lower=0.01,
            upper=5.0,
            step_increment=0.01,
            digits=3,
            value_in_base=d.min_interval,
        )
        self.max_row = LengthSpinRow(
            _("Largest Interval"),
            lower=0.01,
            upper=5.0,
            step_increment=0.01,
            digits=3,
            value_in_base=d.max_interval,
        )
        self.size_row = LengthSpinRow(
            _("Cell Size"),
            lower=1.0,
            upper=100.0,
            value_in_base=d.cell_size,
        )
        self.spacing_row = LengthSpinRow(
            _("Spacing"), lower=0.0, upper=50.0, value_in_base=d.spacing
        )
        for row in (
            self.count_row,
            self.min_row,
            self.max_row,
            self.size_row,
            self.spacing_row,
        ):
            group.add(row)
        self.page.add(group)

        laser = Adw.PreferencesGroup(title=_("Laser"))
        self.power_row = self._power_row(_("Power"), d.power_percent)
        self.speed_row = SpeedSpinRow(
            _("Speed"),
            lower=1,
            upper=max_speed,
            value_in_base=min(d.speed, max_speed),
        )
        laser.add(self.power_row)
        laser.add(self.speed_row)
        self.page.add(laser)
        self._add_label_group(d)

    def params(self) -> IntervalTestParams:
        return IntervalTestParams(
            min_interval=self.min_row.get_value_in_base_units(),
            max_interval=self.max_row.get_value_in_base_units(),
            count=self.count_row.get_int_value(),
            cell_size=self.size_row.get_value_in_base_units(),
            spacing=self.spacing_row.get_value_in_base_units(),
            power_percent=self.power_row.get_value(),
            speed=self.speed_row.get_value_in_base_units(),
            include_labels=self.labels_row.get_active(),
            label_power_percent=self.label_power_row.get_value(),
            label_speed=self.label_speed_row.get_value_in_base_units(),
        )


class FocusTestDialog(_CalibrationTestDialog):
    """Asks for the settings of a Focus Test."""

    def __init__(
        self,
        parent: Gtk.Window | None,
        on_create: Callable[[FocusTestParams], object],
        modes: list[FocusMode],
        max_speed: float = 60000.0,
    ):
        super().__init__(parent, _("Create Focus Test"), on_create)
        d = FocusTestParams()
        self._modes = modes
        group = Adw.PreferencesGroup(title=_("Focus Steps"))
        self.steps_group = group
        self.mode_row = Adw.ComboRow(
            title=_("Method"),
            model=Gtk.StringList.new([m.label for m in modes]),
        )
        if d.mode in modes:
            self.mode_row.set_selected(modes.index(d.mode))
        self.mode_row.connect("notify::selected", lambda *_a: self._sync())
        group.add(self.mode_row)

        limit = MAX_FOCUS_OFFSET_MM
        self.start_row = LengthSpinRow(
            _("First Offset"),
            lower=-limit,
            upper=limit,
            step_increment=0.1,
            digits=2,
            value_in_base=d.start_offset,
        )
        self.step_row = LengthSpinRow(
            _("Step"),
            lower=-limit,
            upper=limit,
            step_increment=0.1,
            digits=2,
            value_in_base=d.step,
        )
        self.count_row = SpinRow(
            _("Number of Lines"), lower=2, upper=41, value=d.count
        )
        self.length_row = LengthSpinRow(
            _("Line Length"),
            lower=1.0,
            upper=200.0,
            value_in_base=d.line_length,
        )
        self.spacing_row = LengthSpinRow(
            _("Spacing"), lower=0.5, upper=50.0, value_in_base=d.spacing
        )
        self.ramp_row = LengthSpinRow(
            _("Ramp Length"),
            lower=10.0,
            upper=1000.0,
            value_in_base=d.ramp_length,
        )
        self._step_rows = (
            self.start_row,
            self.step_row,
            self.count_row,
            self.length_row,
            self.spacing_row,
        )
        for row in (*self._step_rows, self.ramp_row):
            group.add(row)
        self.page.add(group)

        laser = Adw.PreferencesGroup(title=_("Laser"))
        self.power_row = self._power_row(_("Power"), d.power_percent)
        self.speed_row = SpeedSpinRow(
            _("Speed"),
            lower=1,
            upper=max_speed,
            value_in_base=min(d.speed, max_speed),
        )
        laser.add(self.power_row)
        laser.add(self.speed_row)
        self.page.add(laser)
        self._add_label_group(d)
        self._sync()

    @property
    def mode(self) -> FocusMode:
        index = self.mode_row.get_selected()
        if 0 <= index < len(self._modes):
            return self._modes[index]
        return FocusMode.RAMP

    def _sync(self):
        mode = self.mode
        self.steps_group.set_description(_FOCUS_HELP[mode])
        is_ramp = mode == FocusMode.RAMP
        for row in self._step_rows:
            row.set_visible(not is_ramp)
        self.ramp_row.set_visible(is_ramp)

    def params(self) -> FocusTestParams:
        return FocusTestParams(
            mode=self.mode,
            start_offset=self.start_row.get_value_in_base_units(),
            step=self.step_row.get_value_in_base_units(),
            count=self.count_row.get_int_value(),
            line_length=self.length_row.get_value_in_base_units(),
            spacing=self.spacing_row.get_value_in_base_units(),
            ramp_length=self.ramp_row.get_value_in_base_units(),
            power_percent=self.power_row.get_value(),
            speed=self.speed_row.get_value_in_base_units(),
            include_labels=self.labels_row.get_active(),
            label_power_percent=self.label_power_row.get_value(),
            label_speed=self.label_speed_row.get_value_in_base_units(),
        )
