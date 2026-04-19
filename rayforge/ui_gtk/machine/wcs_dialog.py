from gettext import gettext as _

from gi.repository import Adw

from ...machine.models.machine import Machine
from ...shared.tasker import task_mgr


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

        self._row_x = Adw.SpinRow.new_with_range(-10000, 10000, 0.1)
        self._row_x.set_title("X Offset")
        self._row_x.set_value(off_x)
        group.add(self._row_x)

        self._row_y = Adw.SpinRow.new_with_range(-10000, 10000, 0.1)
        self._row_y.set_title("Y Offset")
        self._row_y.set_value(off_y)
        group.add(self._row_y)

        self._row_z = Adw.SpinRow.new_with_range(-10000, 10000, 0.1)
        self._row_z.set_title("Z Offset")
        self._row_z.set_value(off_z)
        group.add(self._row_z)

        self.set_extra_child(group)

        self.connect("response", self._on_response)

    def _on_response(self, dlg, response):
        if response == "save":
            label = self._label_row.get_text()
            nx = self._row_x.get_value()
            ny = self._row_y.get_value()
            nz = self._row_z.get_value()
            self.machine.set_wcs_label(self.machine.active_wcs, label)
            task_mgr.add_coroutine(
                lambda ctx: self.machine.set_work_origin(nx, ny, nz)
            )
