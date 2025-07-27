from gi.repository import Gtk, Gdk  # type: ignore
from ..models.doc import Doc
from ..models.workplan import WorkPlan, WorkStep
from ..undo.list_cmd import ListItemCommand, ReorderListCommand
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
    def __init__(self, doc: Doc, workplan: WorkPlan, **kwargs):
        super().__init__(**kwargs)
        self.add_css_class("workplan")
        self.apply_css()
        self.doc = doc
        self.workplan = workplan

        self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
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
        provider.load_from_string(css)
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
            workstepbox = WorkStepBox(
                self.doc, step, prefix=_("Step {seq}: ").format(seq=seq)
            )
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
            workstep_cls = popup.selected
            new_step = self.workplan.create_workstep(workstep_cls)

            # Create an undoable command for adding the workstep
            command = ListItemCommand(
                owner_obj=self.workplan,
                item=new_step,
                undo_command="remove_workstep",
                redo_command="add_workstep",
                name=_("Add workstep '{name}'").format(name=new_step.name)
            )
            self.doc.history_manager.execute(command)

    def on_button_delete_clicked(self, sender, workstep, **kwargs):
        # Create an undoable command for removing the workstep by
        # creating a new list without the specified step.
        new_list = [s for s in self.workplan.worksteps if s is not workstep]
        command = ReorderListCommand(
            target_obj=self.workplan,
            list_property_name="worksteps",
            new_list=new_list,
            setter_method_name="set_worksteps",
            name=_("Remove workstep '{name}'").format(name=workstep.name)
        )
        self.doc.history_manager.execute(command)

    def on_workplan_changed(self, sender, **kwargs):
        self.update()

    def on_workplan_reordered(self, sender, **kwargs):
        # Create an undoable command for reordering the worksteps
        new_order = [row.data for row in self.draglist]
        command = ReorderListCommand(
            target_obj=self.workplan,
            list_property_name="worksteps",
            new_list=new_order,
            setter_method_name="set_worksteps",
            name=_("Reorder worksteps")
        )
        self.doc.history_manager.execute(command)
