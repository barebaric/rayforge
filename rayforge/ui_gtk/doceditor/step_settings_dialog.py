from typing import TYPE_CHECKING, Dict, Optional, Tuple
from gettext import gettext as _

from blinker import Signal
from gi.repository import Adw, Gtk

from ...context import get_context
from ...core.step import Step
from ...core.undo import ChangePropertyCommand, HistoryManager
from ...core.varset import Var, VarSet
from ...core.capability import LaserHeadVar
from ...pipeline.producer import OpsProducer
from ...pipeline.producer.placeholder import PlaceholderProducer
from ...pipeline.transformer import OpsTransformer
from ...pipeline.transformer.placeholder import PlaceholderTransformer
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..shared.preferences_page import TrackedPreferencesPage
from ..varset.varsetwidget import VarSetWidget
from .recipe_control_widget import RecipeControlWidget
from .step_settings.placeholder import PlaceholderSettingsWidget

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


def _merged_varset(caps: Tuple) -> VarSet:
    merged: Dict[str, Var] = {}
    for cap in caps:
        for var in cap.varset:
            merged[var.key] = var
    return VarSet(vars=list(merged.values()))


class GeneralStepSettingsView(TrackedPreferencesPage):
    """A view for the general and producer settings of a Step."""

    changed = Signal()

    def __init__(self, editor: "DocEditor", step: Step):
        super().__init__()
        self.editor = editor
        self.doc = editor.doc
        self.step = step
        producer_type = (
            step.opsproducer_dict.get("type", "unknown")
            if step.opsproducer_dict
            else "unknown"
        )
        self.key = producer_type.lower().replace("producer", "")
        self.path_prefix = "/step-settings/"
        self.history_manager: HistoryManager = self.doc.history_manager

        # 1. Producer Settings
        producer_dict = self.step.opsproducer_dict
        producer = None
        if producer_dict:
            producer_name = producer_dict.get("type")
            if producer_name:
                producer = OpsProducer.from_dict(producer_dict)

        context = get_context()
        if context:
            context.plugin_mgr.hook.step_settings_loaded(
                dialog=self, step=self.step, producer=producer
            )

        # Add placeholder widget if producer is not available
        if isinstance(producer, PlaceholderProducer) and producer_dict:
            self.add(
                PlaceholderSettingsWidget(
                    self.editor,
                    producer.label,
                    producer,
                    self,
                    self.step,
                )
            )

        # Build settings UI from capability VarSet.
        # VarSetWidget IS the general group — no visual split.
        varset = _merged_varset(
            step.get_effective_capabilities(get_context().machine)
        )
        self.varset_widget = VarSetWidget(
            title=_("General Settings"),
            description=_(
                "Power, speed, and laser head selection for this operation."
            ),
            debounce_ms=300,
        )

        self.recipe_control = RecipeControlWidget(self.editor, self.step)
        self.recipe_control.recipe_applied.connect(self._sync_widgets_to_model)
        self.varset_widget.add(self.recipe_control)
        self.recipe_control.set_visible(
            producer.show_recipe_settings if producer else False
        )

        self.varset_widget.populate(varset)
        self.varset_widget.data_changed.connect(self._on_data_changed)
        self.add(self.varset_widget)

        # Post-process: connect laser head selector for kerf transaction
        if "selected_laser_uid" in self.varset_widget.widget_map:
            row, var = self.varset_widget.widget_map["selected_laser_uid"]
            if isinstance(row, Adw.ComboRow):
                row.connect("notify::selected", self._on_laser_selected)

        self._sync_widgets_to_model()

    def _cleanup(self):
        pass

    def _sync_widgets_to_model(self, sender=None, **kwargs):
        """Updates all widgets to reflect the current state of the Step."""
        values = {}
        for key in self.varset_widget.widget_map:
            if key == "selected_laser_uid":
                continue
            values[key] = getattr(self.step, key, None)
        self.varset_widget.set_values(values)

        # Sync laser head selector using UID-to-name mapping
        if "selected_laser_uid" in self.varset_widget.widget_map:
            __, var = self.varset_widget.widget_map["selected_laser_uid"]
            if isinstance(var, LaserHeadVar):
                self.varset_widget.set_values(
                    {"selected_laser_uid": self.step.selected_laser_uid}
                )

    def _on_laser_selected(self, combo_row, pspec):
        """Handles laser head changes with kerf transaction."""
        machine = get_context().machine
        if not machine or not machine.heads:
            return

        selected_index = combo_row.get_selected()
        selected_laser = machine.heads[selected_index]
        new_uid = selected_laser.uid

        if self.step.selected_laser_uid == new_uid:
            return

        with self.history_manager.transaction(_("Change Laser")) as t:
            t.execute(
                ChangePropertyCommand(
                    target=self.step,
                    property_name="selected_laser_uid",
                    new_value=new_uid,
                    setter_method_name="set_selected_laser_uid",
                )
            )
            new_kerf = selected_laser.spot_size_mm[0]
            t.execute(
                ChangePropertyCommand(
                    target=self.step,
                    property_name="kerf_mm",
                    new_value=new_kerf,
                    setter_method_name="set_kerf_mm",
                )
            )
            if "kerf_mm" in self.varset_widget.widget_map:
                kerf_row, __ = self.varset_widget.widget_map["kerf_mm"]
                if isinstance(kerf_row, Adw.SpinRow):
                    kerf_row.set_value(new_kerf)

        self.changed.send(self)

    def _on_data_changed(self, sender, **kwargs):
        key = kwargs.get("key")
        if not key or key == "selected_laser_uid":
            return
        self._commit_change(key)

    def _commit_change(self, key: str):
        values = self.varset_widget.get_values()
        new_value = values.get(key)
        if new_value is None:
            return

        current_value = getattr(self.step, key, None)
        if current_value is not None and new_value == current_value:
            return

        setter = getattr(self.step, f"set_{key}", None)
        if not setter:
            return

        command = ChangePropertyCommand(
            target=self.step,
            property_name=key,
            new_value=new_value,
            setter_method_name=f"set_{key}",
            name=_("Change {key}").format(key=key.replace("_", " ")),
        )
        self.history_manager.execute(command)
        self.changed.send(self)


class PostProcessingSettingsView(TrackedPreferencesPage):
    """A view for the post-processing transformers of a Step."""

    use_expanders = True

    def __init__(self, editor: "DocEditor", step: Step):
        super().__init__()
        self.editor = editor
        self.step = step
        producer_type = (
            step.opsproducer_dict.get("type", "unknown")
            if step.opsproducer_dict
            else "unknown"
        )
        producer_key = producer_type.lower().replace("producer", "")
        self.key = f"{producer_key}/post-processing"
        self.path_prefix = "/step-settings/"

        self._main_group = Adw.PreferencesGroup()
        super().add(self._main_group)
        self._has_expanders = False

        all_transformer_dicts = (
            step.per_workpiece_transformers_dicts or []
        ) + (step.per_step_transformers_dicts or [])

        # Deduplicate by object identity (same dict can be in both lists)
        seen_ids = set()
        unique_transformer_dicts = []
        for t_dict in all_transformer_dicts:
            dict_id = id(t_dict)
            if dict_id not in seen_ids:
                seen_ids.add(dict_id)
                unique_transformer_dicts.append(t_dict)

        context = get_context()
        if context:
            for t_dict in unique_transformer_dicts:
                transformer = OpsTransformer.from_dict(t_dict)
                context.plugin_mgr.hook.transformer_settings_loaded(
                    dialog=self, step=step, transformer=transformer
                )
                # Add placeholder widget if transformer is not available
                if isinstance(transformer, PlaceholderTransformer):
                    self.add(
                        PlaceholderSettingsWidget(
                            editor,
                            transformer.label,
                            transformer,
                            self,
                            step,
                        )
                    )

        if not self._has_expanders:
            placeholder_label = Gtk.Label(
                label=_("No post-processing options available for this step."),
                halign=Gtk.Align.CENTER,
                margin_top=24,
                margin_bottom=24,
                wrap=True,
            )
            placeholder_label.add_css_class("dim-label")
            self._main_group.add(placeholder_label)

    def add(self, group):
        rows = getattr(group, "_rows", None)
        if rows is None:
            super().add(group)
            return

        title = group.get_title()
        subtitle = group.get_description()

        expander = Adw.ExpanderRow(title=title or "")
        if subtitle:
            expander.set_subtitle(subtitle)
        expander.set_expanded(False)

        enable_switch_row = None
        for row in rows:
            if isinstance(row, Adw.SwitchRow) and enable_switch_row is None:
                enable_switch_row = row
                switch = Gtk.Switch()
                switch.set_active(row.get_active())
                switch.set_valign(Gtk.Align.CENTER)
                expander.add_suffix(switch)

                def _on_header_toggled(sw, pspec, orig=row):
                    if orig.get_active() != sw.get_active():
                        orig.set_active(sw.get_active())

                switch.connect("notify::active", _on_header_toggled)

                def _on_orig_toggled(r, pspec, sw=switch):
                    if sw.get_active() != r.get_active():
                        sw.set_active(r.get_active())

                row.connect("notify::active", _on_orig_toggled)
            else:
                expander.add_row(row)

        self._main_group.add(expander)
        self._has_expanders = True


class StepSettingsDialog(PatchedDialogWindow):
    _open_dialogs: Dict[int, "StepSettingsDialog"] = {}

    def __init__(
        self,
        editor: "DocEditor",
        step: Step,
        **kwargs,
    ):
        super().__init__(skip_usage_tracking=True, **kwargs)
        self.editor = editor
        self.step = step
        self.set_title(_("{name} Settings").format(name=step.name))

        # Adw.ToolbarView provides areas for a header, content, and bottom bar.
        main_view = Adw.ToolbarView()
        self.set_content(main_view)

        # A HeaderBar provides the window decorations (close button, etc.)
        header = Adw.HeaderBar()
        main_view.add_top_bar(header)

        # Gtk.Stack holds the pages.
        self.stack = Gtk.Stack()
        main_view.set_content(self.stack)

        # Set a reasonable default size to avoid being too narrow
        self.set_default_size(600, 750)

        # Destroy window on close to prevent leaks
        self.set_hide_on_close(False)
        self.connect("close-request", self._on_close_request)

        # --- Page 1: Main Step Settings ---
        self.general_view = GeneralStepSettingsView(self.editor, self.step)
        scrolled_page1 = Gtk.ScrolledWindow(
            child=self.general_view,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        self.stack.add_named(scrolled_page1, "step-settings")

        # --- Page 2: Post Processing Settings ---
        self.post_processing_view = PostProcessingSettingsView(
            self.editor, self.step
        )
        scrolled_page2 = Gtk.ScrolledWindow(
            child=self.post_processing_view,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        self.stack.add_named(scrolled_page2, "post-processing")

        # --- Build the custom switcher ---
        switcher_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        switcher_box.add_css_class("linked")
        header.set_title_widget(switcher_box)

        self.btn_step_settings = Gtk.ToggleButton()
        self.btn_step_settings.set_child(
            self._create_tab_title(_("Step Settings"), "laser-path-symbolic")
        )
        self.btn_step_settings.connect(
            "toggled", self._on_tab_toggled, self.stack, "step-settings"
        )
        switcher_box.append(self.btn_step_settings)

        self.btn_post_processing = Gtk.ToggleButton(
            group=self.btn_step_settings
        )
        self.btn_post_processing.set_child(
            self._create_tab_title(
                _("Post Processing"), "post-processor-symbolic"
            )
        )
        self.btn_post_processing.connect(
            "toggled", self._on_tab_toggled, self.stack, "post-processing"
        )
        switcher_box.append(self.btn_post_processing)

        has_post_processors = bool(
            step.per_step_transformers_dicts
            or step.per_workpiece_transformers_dicts
        )
        self.btn_post_processing.set_visible(has_post_processors)

        # Default to step-settings page
        self.btn_step_settings.set_active(True)
        self.general_view._sync_widgets_to_model()

    @classmethod
    def present_for_step(
        cls,
        editor: "DocEditor",
        step: Step,
        parent_window: Optional[Gtk.Root],
    ) -> "StepSettingsDialog":
        existing = cls._open_dialogs.get(id(step))
        if existing:
            existing.present()
            return existing
        dialog = cls(editor, step, transient_for=parent_window)
        cls._open_dialogs[id(step)] = dialog
        dialog.connect("close-request", cls._on_dialog_closed)
        dialog.present()
        return dialog

    @classmethod
    def _on_dialog_closed(cls, dialog: "StepSettingsDialog", *args) -> bool:
        cls._open_dialogs.pop(id(dialog.step), None)
        return False

    def set_initial_page(self, page: str):
        """Set the initial visible page after dialog construction."""
        if page == "post-processing":
            self.btn_post_processing.set_active(True)
        else:
            self.btn_step_settings.set_active(True)

    def _on_tab_toggled(self, button, stack, page_name):
        """Callback to switch the Gtk.Stack page."""
        if button.get_active():
            stack.set_visible_child_name(page_name)

    def _create_tab_title(self, title_str: str, icon_name: str) -> Gtk.Widget:
        """Creates a box with an icon and a label for a tab button."""
        icon = get_icon(icon_name)
        label = Gtk.Label(label=title_str)
        box = Gtk.Box(spacing=6, orientation=Gtk.Orientation.HORIZONTAL)
        box.append(icon)
        box.append(label)
        return box

    def _on_close_request(self, window):
        # Clean up the debounce timer in the general view when the window is
        # closed to prevent a GLib warning.
        self.general_view._cleanup()
        return False  # Allow the window to close
