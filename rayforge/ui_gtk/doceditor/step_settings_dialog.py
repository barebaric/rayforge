from typing import TYPE_CHECKING, Dict, Optional, Tuple
from gettext import gettext as _

from blinker import Signal
from gi.repository import Adw, GLib, Gtk

from ...context import get_context
from ...core.capability import CUT
from ...core.step import Step
from ...core.undo import ChangePropertyCommand, HistoryManager
from ...pipeline.producer import OpsProducer
from ...pipeline.producer.placeholder import PlaceholderProducer
from ...pipeline.transformer import OpsTransformer
from ...pipeline.transformer.placeholder import PlaceholderTransformer
from ..icons import get_icon
from ..shared.adwfix import get_spinrow_float
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..shared.preferences_page import TrackedPreferencesPage
from ..shared.slider import create_slider_row
from ..shared.unit_spin_row import UnitSpinRowHelper
from .recipe_control_widget import RecipeControlWidget
from .step_settings.placeholder import PlaceholderSettingsWidget

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


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

        # Used to delay updates from continuous-change widgets like sliders
        # to avoid excessive updates.
        self._debounce_timer = 0
        self._debounced_callback = None
        self._debounced_args: Tuple = ()

        # Safely get machine properties with sensible fallbacks
        machine = get_context().machine
        if machine:
            max_cut_speed = machine.max_cut_speed
            max_travel_speed = machine.max_travel_speed
        else:
            # Provide sensible defaults if no machine is configured
            max_cut_speed = 3000  # mm/min
            max_travel_speed = 3000  # mm/min

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

        general_group = Adw.PreferencesGroup(
            title=_("General Settings"),
            description=_(
                "Power, speed, and laser head selection for this operation."
            ),
        )
        self.add(general_group)

        # Recipe Control Widget
        self.recipe_control = RecipeControlWidget(self.editor, self.step)
        self.recipe_control.recipe_applied.connect(self._sync_widgets_to_model)
        general_group.add(self.recipe_control)
        self.recipe_control.set_visible(
            producer.show_recipe_settings if producer else False
        )

        # Laser Head Selector
        if machine and machine.heads:
            laser_names = [head.name for head in machine.heads]
            string_list = Gtk.StringList.new(laser_names)
            self.laser_row = Adw.ComboRow(
                title=_("Laser Head"), model=string_list
            )
            self.laser_row.connect("notify::selected", self.on_laser_selected)
            general_group.add(self.laser_row)

        # Power Slider
        self.power_adjustment = Gtk.Adjustment(
            upper=100, step_increment=0.1, page_increment=10
        )
        power_row, power_scale = create_slider_row(
            title=_("Power"),
            adjustment=self.power_adjustment,
            digits=1,
            format_suffix="%",
            on_value_changed=lambda s: self._debounce(
                self.on_power_changed, s
            ),
        )
        general_group.add(power_row)
        # Set power row visibility based on producer capability
        power_row.set_visible(producer.supports_power if producer else False)

        # Tab Power Slider
        self.tab_power_adjustment = Gtk.Adjustment(
            upper=100, step_increment=0.1, page_increment=10
        )
        tab_power_row, tab_power_scale = create_slider_row(
            title=_("Tab Power"),
            subtitle=_("Laser power at tab positions (% of cut power)"),
            adjustment=self.tab_power_adjustment,
            digits=1,
            format_suffix="%",
            on_value_changed=lambda s: self._debounce(
                self.on_tab_power_changed, s
            ),
        )
        general_group.add(tab_power_row)
        # Set tab power row visibility: only for steps with CutCapability
        tab_power_row.set_visible(
            CUT in self.step.capabilities
            and (producer.supports_power if producer else False)
        )
        self.tab_power_row = tab_power_row

        # Add a spin row for cut speed
        cut_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=max_cut_speed,
            step_increment=10,
            page_increment=100,
        )
        cut_speed_row = Adw.SpinRow(
            title=_("Cut Speed"),
            subtitle=_("Max: {max_speed}"),
            adjustment=cut_speed_adjustment,
        )
        self.cut_speed_helper = UnitSpinRowHelper(
            spin_row=cut_speed_row,
            quantity="speed",
            max_value_in_base=max_cut_speed,
        )
        self.cut_speed_helper.changed.connect(
            self._on_cut_speed_changed_wrapper
        )
        general_group.add(cut_speed_row)

        # Set cut speed row visibility based on producer capability
        cut_speed_row.set_visible(
            producer.supports_cut_speed if producer else False
        )

        # Add a spin row for travel speed
        travel_speed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=max_travel_speed,
            step_increment=10,
            page_increment=100,
        )
        travel_speed_row = Adw.SpinRow(
            title=_("Travel Speed"),
            subtitle=_("Max: {max_speed}"),
            adjustment=travel_speed_adjustment,
        )
        self.travel_speed_helper = UnitSpinRowHelper(
            spin_row=travel_speed_row,
            quantity="speed",
            max_value_in_base=max_travel_speed,
        )
        self.travel_speed_helper.changed.connect(
            self._on_travel_speed_changed_wrapper
        )
        general_group.add(travel_speed_row)

        # Add a switch for air assist
        self.air_assist_row = Adw.SwitchRow()
        self.air_assist_row.set_title(_("Air Assist"))
        self.air_assist_row.connect(
            "notify::active", self.on_air_assist_changed
        )
        general_group.add(self.air_assist_row)

        # Kerf Setting (conditionally visible)
        kerf_adj = Gtk.Adjustment(
            lower=0.0,
            upper=2.0,
            step_increment=0.01,
            page_increment=0.1,
        )
        self.kerf_row = Adw.SpinRow(
            title=_("Beam Width (Kerf)"),
            subtitle=_("Effective laser cut width in machine units"),
            adjustment=kerf_adj,
            digits=3,
        )
        self.kerf_row.connect(
            "changed", lambda r: self._debounce(self._on_kerf_changed, r)
        )
        general_group.add(self.kerf_row)

        # Set kerf row visibility based on producer capability
        self.kerf_row.set_visible(
            producer.supports_kerf if producer else False
        )

    def _cleanup(self):
        """Clean up resources like the debounce timer."""
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0

    def _sync_widgets_to_model(self, sender=None, **kwargs):
        """
        Updates all widgets to reflect the current state of the Step model.
        """
        machine = get_context().machine

        # Sync Laser Head
        if machine and machine.heads:
            initial_index = 0
            if self.step.selected_laser_uid:
                try:
                    initial_index = next(
                        i
                        for i, head in enumerate(machine.heads)
                        if head.uid == self.step.selected_laser_uid
                    )
                except StopIteration:
                    pass  # Fallback to index 0
            self.laser_row.set_selected(initial_index)

        # Sync Power
        power_percent = self.step.power * 100.0
        self.power_adjustment.set_value(power_percent)

        # Sync Speeds
        self.cut_speed_helper.set_value_in_base_units(self.step.cut_speed)
        self.travel_speed_helper.set_value_in_base_units(
            self.step.travel_speed
        )

        # Sync Air Assist
        self.air_assist_row.set_active(self.step.air_assist)

        # Sync Kerf
        self.kerf_row.get_adjustment().set_value(self.step.kerf_mm)

        # Sync Tab Power
        tab_power_percent = self.step.tab_power * 100.0
        self.tab_power_adjustment.set_value(tab_power_percent)

    def on_laser_selected(self, combo_row, pspec):
        """Handles changes in the laser head selection."""
        machine = get_context().machine
        if not machine or not machine.heads:
            return

        selected_index = combo_row.get_selected()
        selected_laser = machine.heads[selected_index]
        new_uid = selected_laser.uid

        if self.step.selected_laser_uid == new_uid:
            return

        # Use a transaction to group laser and kerf changes into one
        # undoable action.
        with self.history_manager.transaction(_("Change Laser")) as t:
            # Command for the laser UID
            t.execute(
                ChangePropertyCommand(
                    target=self.step,
                    property_name="selected_laser_uid",
                    new_value=new_uid,
                    setter_method_name="set_selected_laser_uid",
                )
            )

            # Command for the kerf, using the new laser's spot size
            new_kerf = selected_laser.spot_size_mm[0]
            t.execute(
                ChangePropertyCommand(
                    target=self.step,
                    property_name="kerf_mm",
                    new_value=new_kerf,
                    setter_method_name="set_kerf_mm",
                )
            )
            # Update the UI to reflect the new model state
            self.kerf_row.set_value(new_kerf)

        self.changed.send(self)

    def _debounce(self, callback, *args):
        """
        Schedules a callback to be executed after a short delay, canceling any
        previously scheduled callback. This prevents excessive updates from
        widgets like sliders.
        """
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)

        self._debounced_callback = callback
        self._debounced_args = args
        # Debounce requests by 150ms
        self._debounce_timer = GLib.timeout_add(
            150, self._commit_debounced_change
        )

    def _commit_debounced_change(self):
        """Executes the debounced callback."""
        if self._debounced_callback:
            self._debounced_callback(*self._debounced_args)

        self._debounce_timer = 0
        self._debounced_callback = None
        self._debounced_args = ()
        return GLib.SOURCE_REMOVE

    def _on_kerf_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        if abs(new_value - self.step.kerf_mm) > 1e-6:
            command = ChangePropertyCommand(
                target=self.step,
                property_name="kerf_mm",
                new_value=new_value,
                setter_method_name="set_kerf_mm",
                name=_("Change Kerf"),
            )
            self.history_manager.execute(command)
            self.changed.send(self)

    def on_power_changed(self, scale):
        new_value = scale.get_value() / 100.0
        command = ChangePropertyCommand(
            target=self.step,
            property_name="power",
            new_value=new_value,
            setter_method_name="set_power",
            name=_("Change laser power"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_tab_power_changed(self, scale):
        new_value = scale.get_value() / 100.0
        command = ChangePropertyCommand(
            target=self.step,
            property_name="tab_power",
            new_value=new_value,
            setter_method_name="set_tab_power",
            name=_("Change tab power"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def _on_cut_speed_changed_wrapper(self, helper: UnitSpinRowHelper):
        """Wrapper method that debounces the cut speed change."""
        self._debounce(self.on_cut_speed_changed, helper)

    def _on_travel_speed_changed_wrapper(self, helper: UnitSpinRowHelper):
        """Wrapper method that debounces the travel speed change."""
        self._debounce(self.on_travel_speed_changed, helper)

    def on_cut_speed_changed(self, helper: UnitSpinRowHelper):
        new_value = helper.get_value_in_base_units()
        if new_value == self.step.cut_speed:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="cut_speed",
            new_value=new_value,
            setter_method_name="set_cut_speed",
            name=_("Change cut speed"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_travel_speed_changed(self, helper: UnitSpinRowHelper):
        new_value = helper.get_value_in_base_units()
        if new_value == self.step.travel_speed:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="travel_speed",
            new_value=new_value,
            setter_method_name="set_travel_speed",
            name=_("Change Travel Speed"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_air_assist_changed(self, row, pspec):
        new_value = row.get_active()
        if new_value == self.step.air_assist:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="air_assist",
            new_value=new_value,
            setter_method_name="set_air_assist",
            name=_("Toggle air assist"),
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
    def _on_dialog_closed(cls, dialog: "StepSettingsDialog", *_) -> bool:
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
