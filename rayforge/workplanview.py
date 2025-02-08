import gi
from .draglist import DragListBox
from .workstepbox import WorkStepBox
from .roundbutton import RoundButton
from .models.workplan import WorkPlan, WorkStep

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk  # noqa: E402


css = """
.workplan {
    background-color: #ffffff;
    border-radius: 8px;
    margin: 0;
    box-shadow: 0 8px 8px rgba(0, 0, 0, 0.1);
}
"""


class WorkPlanView(Gtk.ScrolledWindow):
    def __init__(self, workplan: WorkPlan):
        super().__init__()
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
        button.connect("clicked", self.on_button_clicked)
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

    def on_button_clicked(self, button):
        self.workplan.add_workstep(WorkStep.for_outline())

    def on_button_delete_clicked(self, sender, workstep, **kwargs):
        self.workplan.remove_workstep(workstep)

    def on_workplan_changed(self, sender, **kwargs):
        self.update()

    def on_workplan_reordered(self, sender, **kwargs):
        self.workplan.set_worksteps([row.data for row in self.draglist])
