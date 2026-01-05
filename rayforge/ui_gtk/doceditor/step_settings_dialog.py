from typing import Tuple, TYPE_CHECKING
from gi.repository import Gtk, Adw, GLib, Gdk
from blinker import Signal
from ...context import get_context
from ...core.step import Step
from ...core.undo import HistoryManager, ChangePropertyCommand
from ..shared.adwfix import get_spinrow_float
from ..shared.unit_spin_row import UnitSpinRowHelper
from ...pipeline.producer import OpsProducer
from ...pipeline.transformer import OpsTransformer
from ..icons import get_icon
from .recipe_control_widget import RecipeControlWidget
from .step_settings import WIDGET_REGISTRY

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


class GeneralStepSettingsView(Adw.PreferencesPage):
    """A view for the general and producer settings of a Step."""

    changed = Signal()

    def __init__(self, editor: "DocEditor", step: Step):
        super().__init__()
        self.editor = editor
        self.doc = editor.doc
        self.step = step
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
                # Instantiate producer to check its properties later
                producer = OpsProducer.from_dict(producer_dict)
                WidgetClass = WIDGET_REGISTRY.get(producer_name)
                if WidgetClass:
                    widget = WidgetClass(
                        self.editor,
                        self.step.typelabel,
                        producer_dict,
                        self,
                        self.step,
                    )
                    self.add(widget)

        general_group = Adw.PreferencesGroup(title=_("General Settings"))
        self.add(general_group)

        # Recipe Control Widget
        self.recipe_control = RecipeControlWidget(self.editor, self.step)
        self.recipe_control.recipe_applied.connect(self._sync_widgets_to_model)
        general_group.add(self.recipe_control)

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
        power_row = Adw.ActionRow(title=_("Power (%)"))
        self.power_adjustment = Gtk.Adjustment(
            upper=100, step_increment=1, page_increment=10
        )
        power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self.power_adjustment,
            digits=0,
            draw_value=True,
        )
        power_scale.set_size_request(300, -1)
        power_scale.connect(
            "value-changed",
            lambda scale: self._debounce(self.on_power_changed, scale),
        )
        power_row.add_suffix(power_scale)
        general_group.add(power_row)
        # Set power row visibility based on producer capability
        power_row.set_visible(producer.supports_power if producer else False)

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


class PostProcessingSettingsView(Adw.PreferencesPage):
    """A view for the post-processing transformers of a Step."""

    def __init__(self, editor: "DocEditor", step: Step):
        super().__init__()

        content_added = False

        # 3. Path Post-Processing Transformers
        if step.per_workpiece_transformers_dicts:
            for t_dict in step.per_workpiece_transformers_dicts:
                transformer_name = t_dict.get("name")
                if transformer_name:
                    WidgetClass = WIDGET_REGISTRY.get(transformer_name)
                    if WidgetClass:
                        transformer = OpsTransformer.from_dict(t_dict)
                        widget = WidgetClass(
                            editor,
                            transformer.label,
                            t_dict,
                            self,
                            step,
                        )
                        self.add(widget)
                        content_added = True

        # 4. Post-Step (Assembly) Transformers
        if step.per_step_transformers_dicts:
            for t_dict in step.per_step_transformers_dicts:
                transformer_name = t_dict.get("name")
                if transformer_name:
                    WidgetClass = WIDGET_REGISTRY.get(transformer_name)
                    if WidgetClass:
                        transformer = OpsTransformer.from_dict(t_dict)
                        widget = WidgetClass(
                            editor,
                            transformer.label,
                            t_dict,
                            self,
                            step,
                        )
                        self.add(widget)
                        content_added = True

        if not content_added:
            # Add a placeholder to ensure this page is never empty. An empty
            # page can cause layout issues.
            placeholder_group = Adw.PreferencesGroup()
            placeholder_label = Gtk.Label(
                label=_("No post-processing options available for this step."),
                halign=Gtk.Align.CENTER,
                margin_top=24,
                margin_bottom=24,
                wrap=True,
            )
            placeholder_label.add_css_class("dim-label")
            placeholder_group.add(placeholder_label)
            self.add(placeholder_group)


class StepSettingsDialog(Adw.Window):
    def __init__(
        self,
        editor: "DocEditor",
        step: Step,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        stack = Gtk.Stack()
        main_view.set_content(stack)

        # Set a reasonable default size to avoid being too narrow
        self.set_default_size(600, 750)

        # Destroy window on close to prevent leaks
        self.set_hide_on_close(False)
        self.connect("close-request", self._on_close_request)

        # Add a key controller to close the dialog on Escape press
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        # --- Page 1: Main Step Settings ---
        self.general_view = GeneralStepSettingsView(self.editor, self.step)
        scrolled_page1 = Gtk.ScrolledWindow(
            child=self.general_view,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        stack.add_named(scrolled_page1, "step-settings")

        # --- Page 2: Post Processing Settings ---
        self.post_processing_view = PostProcessingSettingsView(
            self.editor, self.step
        )
        scrolled_page2 = Gtk.ScrolledWindow(
            child=self.post_processing_view,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        stack.add_named(scrolled_page2, "post-processing")

        # --- Build the custom switcher ---
        switcher_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        switcher_box.add_css_class("linked")
        header.set_title_widget(switcher_box)

        btn1 = Gtk.ToggleButton()
        btn1.set_child(
            self._create_tab_title(_("Step Settings"), "laser-path-symbolic")
        )
        btn1.connect("toggled", self._on_tab_toggled, stack, "step-settings")
        switcher_box.append(btn1)

        btn2 = Gtk.ToggleButton(group=btn1)
        btn2.set_child(
            self._create_tab_title(
                _("Post Processing"), "post-processor-symbolic"
            )
        )
        btn2.connect("toggled", self._on_tab_toggled, stack, "post-processing")
        switcher_box.append(btn2)

        # Set initial state
        btn1.set_active(True)
        self.general_view._sync_widgets_to_model()

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

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events, closing the dialog on Escape or Ctrl+W."""
        has_ctrl = state & Gdk.ModifierType.CONTROL_MASK

        if keyval == Gdk.KEY_Escape or (has_ctrl and keyval == Gdk.KEY_w):
            self.close()
            return True
        return False

    def _on_close_request(self, window):
        # Clean up the debounce timer in the general view when the window is
        # closed to prevent a GLib warning.
        self.general_view._cleanup()
        return False  # Allow the window to close
