from gi.repository import Gtk, Gdk
from ..models.workplan import WorkPlan, WorkStep
from .draglist import DragListBox
from .workstepbox import WorkStepBox
from .stepselector import WorkStepSelector
from .roundbutton import RoundButton


css = """
.workplan {
    background-color: #ffffff;
    border-radius: 8px;
    margin: 0 0 9px 0;
    box-shadow: 0 8px 8px rgba(0, 0, 0, 0.1);
}
"""


class WorkPlanView(Gtk.ScrolledWindow):
    def __init__(self, workplan: WorkPlan, **kwargs):
        super().__init__(**kwargs)
        self.add_css_class("workplan")
        self.apply_css()
        self.workplan = workplan

        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_child(self.box)

        self.draglist = DragListBox()
        self.draglist.reordered.connect(self.on_workplan_reordered)
        self.box.append(self.draglist)
        self.workplan.changed.connect(self.on_workplan_changed)

        # Add "+" button
        button = RoundButton("+")
        button.connect("clicked", self.on_button_add_clicked)
        self.box.append(button)

        self.update()

    def apply_css(self):
        provider = Gtk.CssProvider()
        provider.load_from_data(css.encode())
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def update(self):
        self.draglist.remove_all()
        for seq, step in enumerate(self.workplan, start=1):
            row = Gtk.ListBoxRow()
            row.data = step
            self.draglist.add_row(row)
            workstepbox = WorkStepBox(step, prefix=f"Step {seq}: ")
            workstepbox.delete_clicked.connect(self.on_button_delete_clicked)
            row.set_child(workstepbox)

    def on_button_add_clicked(self, button):
        popup = WorkStepSelector(WorkStep.__subclasses__())
        popup.set_parent(button)
        popup.popup()
        popup.connect("closed", self.on_add_dialog_response)
        return

    def on_add_dialog_response(self, popup):
        if popup.selected:
            self.workplan.add_workstep(popup.selected())

    def on_button_delete_clicked(self, sender, workstep, **kwargs):
        self.workplan.remove_workstep(workstep)

    def on_workplan_changed(self, sender, **kwargs):
        self.update()

    def on_workplan_reordered(self, sender, **kwargs):
        self.workplan.set_worksteps([row.data for row in self.draglist])
