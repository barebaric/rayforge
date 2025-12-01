import logging
from typing import Optional, List, Callable, cast, TYPE_CHECKING
from gi.repository import Gtk
from ...icons import get_icon
from ...core.workflow import Workflow
from ...undo.models.list_cmd import ListItemCommand, ReorderListCommand
from ...shared.ui.draglist import DragListBox
from ...shared.ui.expander import Expander
from .step_box import StepBox
from ...shared.ui.popover_menu import PopoverMenu

if TYPE_CHECKING:
    from ..editor import DocEditor

logger = logging.getLogger(__name__)


class WorkflowView(Expander):
    """
    A widget that displays a collapsible, reorderable list of Steps
    for a given Workflow.
    """

    def __init__(
        self,
        editor: "DocEditor",
        workflow: Workflow,
        step_factories: List[Callable],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.workflow: Optional[Workflow] = None  # Will be set by set_workflow
        self.step_factories = step_factories
        self.editor = editor
        self.set_expanded(True)

        # A container for all content that will be revealed by the expander
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(content_box)

        # The reorderable list of steps goes inside the content box
        self.draglist = DragListBox()
        self.draglist.reordered.connect(self.on_workflow_reordered)
        content_box.append(self.draglist)

        # A Gtk.Button, styled as a card, serves as our "Add" button
        add_button = Gtk.Button()
        add_button.add_css_class("darkbutton")
        add_button.connect("clicked", self.on_button_add_clicked)
        content_box.append(add_button)

        # The button's content is a box with an icon and a label.
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_box.set_margin_top(10)
        button_box.set_margin_end(12)
        button_box.set_margin_bottom(10)
        button_box.set_margin_start(12)

        add_icon = get_icon("add-symbolic")
        button_box.append(add_icon)

        lbl = _("Add New Step...")
        add_label = Gtk.Label()
        add_label.set_markup(f"<span weight='normal'>{lbl}</span>")
        add_label.set_xalign(0)  # Left-align the label
        button_box.append(add_label)

        add_button.set_child(button_box)

        # Set initial workflow
        self.set_workflow(workflow)

    def set_workflow(self, workflow: Optional[Workflow]):
        """Sets the view to display a different workflow."""
        if self.workflow:
            try:
                # Disconnect old handlers
                self.workflow.updated.disconnect(self.on_workflow_changed)
                self.workflow.descendant_added.disconnect(
                    self.on_workflow_changed
                )
                self.workflow.descendant_removed.disconnect(
                    self.on_workflow_changed
                )
            except (TypeError, ValueError):
                pass

        self.workflow = workflow
        self.set_visible(bool(self.workflow))

        if self.workflow:
            # Connect to signals that indicate a change in the workflow's
            # properties or its list of children.
            self.workflow.updated.connect(self.on_workflow_changed)
            self.workflow.descendant_added.connect(self.on_workflow_changed)
            self.workflow.descendant_removed.connect(self.on_workflow_changed)
            # Trigger initial full population and metadata update
            self.on_workflow_changed(self.workflow)

    def on_workflow_changed(self, sender, **kwargs):
        """
        Handles any change to the workflow (structural or property) by
        updating the UI completely.
        """
        if not self.workflow:
            return

        # Update metadata
        count = len(self.workflow.steps)
        self.set_title(self.workflow.name)
        self.set_subtitle(
            _("{count} step").format(count=count)
            if count == 1
            else _("{count} steps").format(count=count)
        )

        # Rebuild the list of step widgets
        self.update_list()

    def update_list(self):
        """
        Re-populates the draglist to match the state of the workflow's steps.
        """
        if not self.workflow or not self.workflow.doc:
            return

        # Check if the list of steps is already in sync to avoid unnecessary
        # rebuilds.
        current_steps = [row.data for row in self.draglist]  # type: ignore
        if current_steps == self.workflow.steps:
            # The list structure is the same, just tell each stepbox to
            # update its summary.
            for i, row in enumerate(self.draglist):
                row = cast(Gtk.ListBoxRow, row)
                hbox = row.get_child()
                assert hbox, "Failed to get hbox from draglist row"
                stepbox = hbox.get_last_child()
                if isinstance(stepbox, StepBox):
                    stepbox.set_prefix(_("Step {seq}: ").format(seq=i + 1))
                    stepbox.on_step_changed(stepbox.step)
            return

        # If the list structure has changed, rebuild it completely.
        self.draglist.remove_all()
        for seq, step in enumerate(self.workflow, start=1):
            row = Gtk.ListBoxRow()
            row.data = step  # type: ignore # Store model for reordering
            stepbox = StepBox(
                self.editor,
                step,
                prefix=_("Step {seq}: ").format(seq=seq),
            )
            stepbox.delete_clicked.connect(self.on_button_delete_clicked)
            row.set_child(stepbox)
            self.draglist.add_row(row)

    def on_button_add_clicked(self, button):
        """Shows a popup to select and add a new step type."""
        if not self.workflow or not self.workflow.doc:
            return

        popup = PopoverMenu(
            step_factories=self.step_factories, context=self.editor.context
        )
        popup.set_parent(button)
        popup.popup()
        popup.connect("closed", self.on_add_dialog_response)

    def on_add_dialog_response(self, popup: PopoverMenu):
        """Handles the creation of a new step after the popup closes."""
        if not self.workflow or not self.workflow.doc:
            return
        if popup.selected_item:
            step_factory = popup.selected_item
            new_step = step_factory(self.editor.context)

            # Apply best recipe using helper method
            self.editor.step.apply_best_recipe_to_step(new_step)

            command = ListItemCommand(
                owner_obj=self.workflow,
                item=new_step,
                undo_command="remove_step",
                redo_command="add_step",
                name=_("Add step '{name}'").format(name=new_step.name),
            )
            self.workflow.doc.history_manager.execute(command)

    def on_button_delete_clicked(self, sender, step, **kwargs):
        """Handles deletion of a step with an undoable command."""
        if not self.workflow or not self.workflow.doc:
            return
        new_list = [s for s in self.workflow.steps if s is not step]
        command = ReorderListCommand(
            target_obj=self.workflow,
            list_property_name="steps",
            new_list=new_list,
            setter_method_name="set_steps",
            name=_("Remove step '{name}'").format(name=step.name),
        )
        self.workflow.doc.history_manager.execute(command)

    def on_workflow_reordered(self, sender, **kwargs):
        """Handles reordering of steps with an undoable command."""
        if not self.workflow or not self.workflow.doc:
            return
        new_order = [row.data for row in self.draglist]  # type: ignore
        command = ReorderListCommand(
            target_obj=self.workflow,
            list_property_name="steps",
            new_list=new_order,
            setter_method_name="set_steps",
            name=_("Reorder steps"),
        )
        self.workflow.doc.history_manager.execute(command)
