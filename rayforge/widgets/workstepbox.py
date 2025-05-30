from gi.repository import Gtk
from blinker import Signal
from ..models.workpiece import WorkPiece
from ..models.workplan import WorkStep
from ..util.resources import get_icon_path
from .groupbox import GroupBox
from .workstepsettings import WorkStepSettingsDialog


class WorkStepBox(GroupBox):
    def __init__(self, workstep: WorkStep, prefix=''):
        super().__init__(workstep.name, workstep.get_summary())
        self.workstep = workstep
        self.prefix = prefix
        self.delete_clicked = Signal()

        self.visibility_on_icon = Gtk.Image.new_from_file(
            get_icon_path('visibility-on')
        )
        self.visibility_off_icon = Gtk.Image.new_from_file(
            get_icon_path('visibility-off')
        )
        button = Gtk.ToggleButton()
        button.set_active(workstep.visible)
        button.set_child(self.visibility_on_icon)
        self.add_button(button)
        button.connect('clicked', self.on_button_view_click)
        self.on_button_view_click(button)

        icon = Gtk.Image.new_from_file(get_icon_path('settings'))
        button = Gtk.Button()
        button.set_child(icon)
        self.add_button(button)
        button.connect('clicked', self.on_button_properties_clicked)

        icon = Gtk.Image.new_from_file(get_icon_path('delete'))
        button = Gtk.Button()
        button.set_child(icon)
        self.add_button(button)
        button.connect('clicked', self.on_button_delete_clicked)

        self.on_workstep_changed(self.workstep)   # trigger label update
        # TODO: self.add_child(thumbnail)

    def set_prefix(self, prefix):
        self.prefix = prefix

    def on_workstep_changed(self, sender, **kwargs):
        self.title_label.set_label(f"{self.prefix}{self.workstep.name}")
        self.subtitle_label.set_label(self.workstep.get_summary())

    def on_button_view_click(self, button):
        self.workstep.set_visible(button.get_active())
        if button.get_active():
            button.set_child(self.visibility_on_icon)
        else:
            button.set_child(self.visibility_off_icon)

    def on_button_properties_clicked(self, button):
        dialog = WorkStepSettingsDialog(self.workstep)
        dialog.present(self)
        dialog.changed.connect(self.on_workstep_changed)

    def on_button_delete_clicked(self, button):
        self.delete_clicked.send(self, workstep=self.workstep)


if __name__ == "__main__":
    from typing import cast
    from ..opsproducer import OpsProducer
    from ..render import Renderer

    class TestWindow(Gtk.ApplicationWindow):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            producer = cast(OpsProducer, object)  # dummy
            renderer = cast(Renderer, object)  # dummy
            workstep = WorkStep(producer, 'My test workstep')
            workstep.add_workpiece(WorkPiece('Item one', b'', renderer))

            box = WorkStepBox(workstep)
            self.set_child(box)
            self.set_default_size(300, 200)

    def on_activate(app):
        win = TestWindow(application=app)
        win.present()

    app = Gtk.Application(application_id="org.example.groupviewexample")
    app.connect('activate', on_activate)
    app.run()
