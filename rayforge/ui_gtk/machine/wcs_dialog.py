from gettext import gettext as _

from gi.repository import Adw

from ...machine.models.machine import Machine
from ...shared.tasker import task_mgr
from ..shared.unit_spin_row import LengthSpinRow


class WcsDialog(Adw.MessageDialog):
    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            heading=_("Edit Work Offsets"),
            body=_(
                "Enter the offset from Machine Zero to Work Zero for "
                "the active WCS."
            ),
            **kwargs,
        )
        self.machine = machine
        self.add_response("cancel", _("Cancel"))
        self.add_response("save", _("Save"))
        self.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)
        self.set_default_response("save")
        self.set_close_response("cancel")

        off_x, off_y, off_z = machine.get_active_wcs_offset()
        wcs_label = machine.get_wcs_label(machine.active_wcs)

        group = Adw.PreferencesGroup()

        self._label_row = Adw.EntryRow(title=_("Label"), text=wcs_label)
        group.add(self._label_row)

        self._row_x = LengthSpinRow(
            _("X Offset"),
            lower=-10000,
            upper=10000,
            min_value_in_base=-10000.0,
            max_value_in_base=10000.0,
            value_in_base=off_x,
        )
        group.add(self._row_x)

        self._row_y = LengthSpinRow(
            _("Y Offset"),
            lower=-10000,
            upper=10000,
            min_value_in_base=-10000.0,
            max_value_in_base=10000.0,
            value_in_base=off_y,
        )
        group.add(self._row_y)

        self._row_z = LengthSpinRow(
            _("Z Offset"),
            lower=-10000,
            upper=10000,
            min_value_in_base=-10000.0,
            max_value_in_base=10000.0,
            value_in_base=off_z,
        )
        group.add(self._row_z)

        self.set_extra_child(group)

        self.connect("response", self._on_response)

    def _on_response(self, dlg, response):
        if response == "save":
            label = self._label_row.get_text()
            nx = self._row_x.get_value_in_base_units()
            ny = self._row_y.get_value_in_base_units()
            nz = self._row_z.get_value_in_base_units()
            self.machine.set_wcs_label(self.machine.active_wcs, label)
            task_mgr.add_coroutine(
                lambda ctx: self.machine.set_work_origin(nx, ny, nz)
            )
