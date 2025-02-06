import gi
from .groupbox import GroupBox
from .draglist import DragListBox
from .models.workpiece import WorkPiece
from .models.workstep import WorkStep
from .workstepsettings import WorkStepSettingsDialog

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402


class WorkStepBox(GroupBox):
    def __init__(self, workstep: WorkStep):
        # Hint: possible icon names can be found using gtk3-icon-browser
        super().__init__(workstep.name,
                         workstep.get_summary(),
                         icon_name='applications-engineering')
        self.workstep = workstep
        self.listbox = DragListBox()
        self.add_child(self.listbox)

        for workpiece in workstep.workpieces:
            pass  # TODO: self.add_workpiece(workpiece)

        self.button.connect('clicked', self.on_button_clicked)

    def on_workstep_changed(self, sender, **kwargs):
        self.title_label.set_label(self.workstep.name)
        self.subtitle_label.set_label(self.workstep.get_summary())

    def add_workpiece(self, workpiece: WorkPiece):
        label = Gtk.Label(label=workpiece.name)
        label.set_xalign(0)
        row = Gtk.ListBoxRow()
        row.set_child(label)
        self.listbox.add_row(row)

    def on_button_clicked(self, button):
        dialog = WorkStepSettingsDialog(self.workstep)
        dialog.present()
        dialog.changed.connect(self.on_workstep_changed)


if __name__ == "__main__":
    class TestWindow(Gtk.ApplicationWindow):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            workstep = WorkStep('My test workstep')
            workstep.add_workpiece(WorkPiece('Item one'))

            box = WorkStepBox(workstep)
            self.set_child(box)
            self.set_default_size(300, 200)

    def on_activate(app):
        win = TestWindow(application=app)
        win.present()

    app = Gtk.Application(application_id="org.example.groupviewexample")
    app.connect('activate', on_activate)
    app.run()
