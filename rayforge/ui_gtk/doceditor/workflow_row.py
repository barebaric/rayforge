from gettext import gettext as _
from typing import TYPE_CHECKING, Optional

from gi.repository import GObject, Gdk, Gtk

from ...core.capability import ENGRAVE
from ...core.step_registry import step_registry
from ...core.undo.list_cmd import ListItemCommand, ReorderListCommand
from ...context import get_context
from ..icons import get_icon
from ..shared.gtk import apply_css
from ..shared.popover_menu import PopoverMenu
from .step_settings_dialog import StepSettingsDialog

if TYPE_CHECKING:
    from ...core.layer import Layer
    from ...core.workflow import Workflow
    from ...doceditor.editor import DocEditor

css = """
.workflow-row {
    min-height: 36px;
    padding: 0px 3px;
    margin-bottom: 3px;
    background-color: alpha(@theme_fg_color, 0.04);
    border-bottom: 1px solid @borders;
}
.workflow-step-button {
    min-width: 28px;
    min-height: 28px;
    padding: 0px;
    margin: 2px;
    border-radius: 6px;
}
.workflow-step-button:hover {
    background-color: alpha(@theme_fg_color, 0.08);
}
.workflow-arrow {
    margin: 0 -1px;
}
.workflow-drop-indicator {
    min-width: 2px;
    min-height: 24px;
    background-color: @accent_color;
    border-radius: 1px;
}
"""

_FALLBACK_ICON = "laser-path-symbolic"


class WorkflowRow(Gtk.Box):
    def __init__(
        self,
        editor: "DocEditor",
        layer: "Layer",
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)
        apply_css(css)
        self.add_css_class("workflow-row")
        self.set_hexpand(True)

        self.editor = editor
        self.layer = layer
        self._workflow: Optional["Workflow"] = None
        self._drag_source_uid: Optional[str] = None
        self._potential_drop_index: int = -1
        self._step_buttons: list = []
        self._btn_uids: dict = {}
        self._drop_indicator: Optional[Gtk.Box] = None

        self._setup_drag_source()
        self._setup_drop_target()

        self._connect_machine()
        self._connect_workflow()
        self._rebuild()

    def _setup_drag_source(self):
        self._drag_source = Gtk.DragSource()
        self._drag_source.set_actions(Gdk.DragAction.MOVE)
        self._drag_source.connect("prepare", self._on_drag_prepare)
        self._drag_source.connect("drag-end", self._on_drag_end)
        self._drag_source.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self.add_controller(self._drag_source)

    def _setup_drop_target(self):
        self._drop_target = Gtk.DropTarget.new(
            GObject.TYPE_STRING, Gdk.DragAction.MOVE
        )
        self._drop_target.connect("drop", self._on_drop)
        self._drop_target.connect("motion", self._on_drop_motion)
        self._drop_target.connect("leave", self._on_drop_leave)
        self.add_controller(self._drop_target)

    def _connect_machine(self):
        machine = get_context().machine
        if machine:
            machine.changed.connect(self._on_machine_changed)

    def _disconnect_machine(self):
        machine = get_context().machine
        if machine:
            try:
                machine.changed.disconnect(self._on_machine_changed)
            except (TypeError, ValueError):
                pass

    def _on_machine_changed(self, sender, **kwargs):
        self._rebuild()

    def _connect_workflow(self):
        self._disconnect_workflow()
        self._workflow = self.layer.workflow
        if self._workflow:
            self._workflow.descendant_added.connect(self._on_workflow_changed)
            self._workflow.descendant_removed.connect(
                self._on_workflow_changed
            )
            self._workflow.descendant_updated.connect(
                self._on_workflow_changed
            )
            self._workflow.updated.connect(self._on_workflow_changed)

    def _disconnect_workflow(self):
        if not self._workflow:
            return
        try:
            self._workflow.descendant_added.disconnect(
                self._on_workflow_changed
            )
            self._workflow.descendant_removed.disconnect(
                self._on_workflow_changed
            )
            self._workflow.descendant_updated.disconnect(
                self._on_workflow_changed
            )
            self._workflow.updated.disconnect(self._on_workflow_changed)
        except (TypeError, ValueError):
            pass
        self._workflow = None

    def refresh(self):
        current_workflow = self.layer.workflow
        if current_workflow is not self._workflow:
            self._connect_workflow()
            self._rebuild()

    def _on_workflow_changed(self, sender, **kwargs):
        self._rebuild()

    def _get_step_icon(self, step) -> str:
        return step.ICON or _FALLBACK_ICON

    def _get_step_color(self, step) -> Optional[str]:
        if not step.visible:
            return None
        machine = get_context().machine
        if not machine or not machine.heads:
            return None
        try:
            laser = step.get_selected_laser(machine)
        except ValueError:
            return None
        if ENGRAVE in step.capabilities:
            return laser.raster_color
        return laser.cut_color

    def _apply_step_color(self, button: Gtk.Button, step):
        color = self._get_step_color(step)
        if not color:
            return
        class_name = f"step-color-{color.lstrip('#').lower()}"
        color_css = (
            f".workflow-step-button.{class_name} {{"
            f"  border: 2px solid {color};"
            "}"
        )
        apply_css(color_css)
        button.add_css_class(class_name)

    def _rebuild(self):
        self._drag_source_uid = None
        self._potential_drop_index = -1
        self._step_buttons = []
        self._btn_uids = {}

        child = self.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self.remove(child)
            child = next_child

        workflow = self._workflow
        if not workflow or not workflow.steps:
            label = Gtk.Label(label=_("No Operations"))
            label.add_css_class("dim-label")
            label.add_css_class("caption")
            label.set_margin_start(6)
            self.append(label)
        else:
            for i, step in enumerate(workflow.steps):
                if i > 0:
                    arrow = get_icon("go-next-symbolic")
                    arrow.add_css_class("workflow-arrow")
                    arrow.set_valign(Gtk.Align.CENTER)
                    self.append(arrow)

                icon = get_icon(self._get_step_icon(step))
                icon.set_pixel_size(18)

                button = Gtk.Button(child=icon)
                button.add_css_class("workflow-step-button")
                button.add_css_class("flat")
                button.set_tooltip_text(step.name)
                button.set_valign(Gtk.Align.CENTER)
                button.connect(
                    "clicked", self._make_step_clicked_handler(step)
                )
                self._apply_step_color(button, step)
                self._btn_uids[id(button)] = step.uid
                self._step_buttons.append(button)
                self.append(button)

        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        self.append(spacer)

        add_icon = get_icon("list-add-symbolic")
        add_btn = Gtk.Button(child=add_icon)
        add_btn.add_css_class("flat")
        add_btn.set_tooltip_text(_("Add Step"))
        add_btn.set_valign(Gtk.Align.CENTER)
        add_btn.connect("clicked", self._on_add_step_clicked)
        self.append(add_btn)

    def _step_uid_at(self, x: float) -> Optional[str]:
        for btn in self._step_buttons:
            alloc = btn.get_allocation()
            if alloc.x <= x < alloc.x + alloc.width:
                return self._btn_uids.get(id(btn))
        return None

    def _btn_at(self, x: float) -> Optional[Gtk.Button]:
        for btn in self._step_buttons:
            alloc = btn.get_allocation()
            if alloc.x <= x < alloc.x + alloc.width:
                return btn
        return None

    def _step_index_at(self, x: float) -> int:
        for i, btn in enumerate(self._step_buttons):
            alloc = btn.get_allocation()
            if x < alloc.x + alloc.width / 2:
                return i
        return len(self._step_buttons)

    def _on_drag_prepare(self, source, x, y):
        uid = self._step_uid_at(x)
        if not uid:
            return None

        btn = self._btn_at(x)
        if btn:
            snapshot = Gtk.Snapshot()
            Gtk.Widget.do_snapshot(btn, snapshot)
            paintable = snapshot.to_paintable()
            if paintable:
                btn_alloc = btn.get_allocation()
                source.set_icon(
                    paintable, btn_alloc.width / 2, btn_alloc.height / 2
                )

        self._drag_source_uid = uid
        self._potential_drop_index = -1
        return Gdk.ContentProvider.new_for_value(uid)

    def _on_drag_end(self, source, drag, delete_data):
        if delete_data and self._potential_drop_index != -1:
            self._commit_reorder()
        self._drag_source_uid = None
        self._potential_drop_index = -1
        self._remove_drop_indicator()

    def _on_drop(self, drop_target, value, x, y):
        if not self._drag_source_uid:
            return False
        return self._potential_drop_index != -1

    def _on_drop_motion(self, drop_target, x, y):
        if not self._drag_source_uid:
            return Gdk.DragAction(0)
        self._potential_drop_index = self._step_index_at(x)
        self._show_drop_indicator(self._potential_drop_index)
        return Gdk.DragAction.MOVE

    def _on_drop_leave(self, drop_target):
        self._potential_drop_index = -1
        self._remove_drop_indicator()

    def _show_drop_indicator(self, index: int):
        self._remove_drop_indicator()
        if index < 0:
            return
        indicator = Gtk.Box()
        indicator.add_css_class("workflow-drop-indicator")
        indicator.set_valign(Gtk.Align.CENTER)
        self._drop_indicator = indicator
        if index >= len(self._step_buttons):
            last_btn = self._step_buttons[-1] if self._step_buttons else None
            if last_btn:
                self.insert_child_after(indicator, last_btn)
            else:
                self.prepend(indicator)
        else:
            btn = self._step_buttons[index]
            prev = btn.get_prev_sibling()
            if prev:
                self.insert_child_after(indicator, prev)
            else:
                self.prepend(indicator)

    def _remove_drop_indicator(self):
        if self._drop_indicator:
            self.remove(self._drop_indicator)
            self._drop_indicator = None

    def _commit_reorder(self):
        workflow = self._workflow
        if not workflow or not workflow.doc:
            return
        steps = list(workflow.steps)
        source_index = None
        for i, s in enumerate(steps):
            if s.uid == self._drag_source_uid:
                source_index = i
                break
        if source_index is None:
            return
        target_index = self._potential_drop_index
        if source_index == target_index:
            return
        new_order = list(steps)
        moved = new_order.pop(source_index)
        insert_at = target_index
        if source_index < target_index:
            insert_at -= 1
        new_order.insert(insert_at, moved)
        command = ReorderListCommand(
            target_obj=workflow,
            list_property_name="steps",
            new_list=new_order,
            setter_method_name="set_steps",
            name=_("Reorder steps"),
        )
        workflow.doc.history_manager.execute(command)

    def _make_step_clicked_handler(self, step):
        def handler(button):
            parent_window = self.get_root()
            dialog = StepSettingsDialog(
                self.editor,
                step,
                transient_for=parent_window,
            )
            dialog.present()

        return handler

    def _on_add_step_clicked(self, button):
        workflow = self._workflow
        if not workflow or not workflow.doc:
            return
        popup = PopoverMenu(
            step_factories=step_registry.get_factories(),
            context=self.editor.context,
        )
        popup.set_parent(self)
        popup.popup()
        popup.connect("closed", self._on_add_step_dialog_response)

    def _on_add_step_dialog_response(self, popup):
        workflow = self._workflow
        if not workflow or not workflow.doc:
            return
        if popup.selected_item:
            step_factory = popup.selected_item
            new_step = step_factory(self.editor.context)
            self.editor.step.apply_best_recipe_to_step(new_step)
            command = ListItemCommand(
                owner_obj=workflow,
                item=new_step,
                undo_command="remove_step",
                redo_command="add_step",
                name=_("Add step '{name}'").format(name=new_step.name),
            )
            workflow.doc.history_manager.execute(command)
            parent_window = self.get_root()
            dialog = StepSettingsDialog(
                self.editor,
                new_step,
                transient_for=parent_window,
            )
            dialog.present()

    def do_destroy(self):
        self._disconnect_machine()
        self._disconnect_workflow()
