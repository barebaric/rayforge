import logging
from gettext import gettext as _
from typing import Any, Dict, List, Optional, Type

from blinker import Signal
from gi.repository import Adw, Gtk

from ...context import get_context
from ...machine.device.profile import DeviceProfile
from ...machine.driver import drivers
from ...machine.driver.driver import Driver
from ...shared.tasker import task_mgr
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..shared.unit_spin_row import UnitSpinRowHelper
from ..varset.varsetwidget import VarSetWidget

logger = logging.getLogger(__name__)

_PROBING_DRIVERS = [d for d in drivers if d.supports_probing]


class ConfigWizard(PatchedDialogWindow):
    """
    Driver-agnostic configuration wizard for probing devices.

    Shows all probing-capable drivers in a combo row on the connect
    page.  The connection parameters update dynamically when the
    driver changes.  After probing, the review page lets the user
    adjust values before creating the machine.

    Emits ``profile_created`` with the resulting DeviceProfile.
    """

    profile_created = Signal()

    def __init__(self, **kwargs):
        super().__init__(
            transient_for=kwargs.pop("transient_for", None),
            modal=True,
            default_width=700,
            default_height=600,
            title=_("Configure Device"),
            **kwargs,
        )

        self._driver_cls: Optional[Type[Driver]] = None
        self._profile: Optional[DeviceProfile] = None
        self._warnings: List[str] = []
        self._warning_rows: List[Adw.ActionRow] = []

        self._setup_ui()

    def _setup_ui(self):
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(content)

        header = Adw.HeaderBar()
        content.append(header)

        self._main_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=24,
            margin_bottom=24,
            margin_start=24,
            margin_end=24,
        )
        content.append(self._main_box)

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self._stack.set_vexpand(True)
        self._main_box.append(self._stack)

        self._setup_connect_page()
        self._setup_discover_page()
        self._setup_review_page()

        self._button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.END,
            margin_top=12,
        )
        self._main_box.append(self._button_box)

        self._back_btn = Gtk.Button(label=_("Back"))
        self._back_btn.add_css_class("flat")
        self._back_btn.connect("clicked", self._on_back_clicked)
        self._back_btn.set_visible(False)
        self._button_box.append(self._back_btn)

        self._cancel_btn = Gtk.Button(label=_("Cancel"))
        self._cancel_btn.add_css_class("flat")
        self._cancel_btn.connect("clicked", lambda _: self.close())
        self._button_box.append(self._cancel_btn)

        self._next_btn = Gtk.Button(label=_("Next"))
        self._next_btn.add_css_class("suggested-action")
        self._next_btn.connect("clicked", self._on_next_clicked)
        self._button_box.append(self._next_btn)

        self._create_btn = Gtk.Button(label=_("Create Machine"))
        self._create_btn.add_css_class("suggested-action")
        self._create_btn.connect("clicked", self._on_create_clicked)
        self._create_btn.set_visible(False)
        self._button_box.append(self._create_btn)

        self._retry_btn = Gtk.Button(label=_("Retry"))
        self._retry_btn.connect("clicked", self._on_retry_clicked)
        self._retry_btn.set_visible(False)
        self._button_box.append(self._retry_btn)

        self._stack.connect("notify::visible-child", self._on_page_changed)

        if _PROBING_DRIVERS:
            self._select_driver(0)

    def _setup_connect_page(self):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)

        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        scrolled.set_child(page_box)

        driver_group = Adw.PreferencesGroup(
            title=_("Connection"),
            description=_(
                "Select a driver and enter the connection parameters "
                "for your device."
            ),
        )
        page_box.append(driver_group)

        self._driver_store = Gtk.StringList()
        for d in _PROBING_DRIVERS:
            self._driver_store.append(d.label)

        self._combo_row = Adw.ComboRow(
            title=_("Select driver"),
            model=self._driver_store,
        )
        self._combo_row.set_use_subtitle(True)
        driver_group.add(self._combo_row)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_factory_setup)
        factory.connect("bind", self._on_factory_bind)
        self._combo_row.set_factory(factory)

        self._connect_widget = VarSetWidget()
        page_box.append(self._connect_widget)

        self._combo_row.connect("notify::selected", self._on_driver_changed)

        if _PROBING_DRIVERS:
            self._combo_row.set_selected(0)

        self._stack.add_named(scrolled, "connect")

    def _on_factory_setup(self, factory, list_item):
        row = Adw.ActionRow()
        list_item.set_child(row)

    def _on_factory_bind(self, factory, list_item):
        index = list_item.get_position()
        driver_cls = _PROBING_DRIVERS[index]
        row = list_item.get_child()
        row.set_title(driver_cls.label)
        row.set_subtitle(driver_cls.subtitle)

    def _on_driver_changed(self, combo_row, _param):
        self._select_driver(combo_row.get_selected())

    def _select_driver(self, index: int):
        if index < 0 or index >= len(_PROBING_DRIVERS):
            self._driver_cls = None
            self._combo_row.set_title(_("Select driver"))
            self._combo_row.set_subtitle("")
            self._connect_widget.clear_dynamic_rows()
            self._next_btn.set_sensitive(False)
            return

        driver_cls = _PROBING_DRIVERS[index]
        self._driver_cls = driver_cls
        self._combo_row.set_title(driver_cls.label)
        self._combo_row.set_subtitle(driver_cls.subtitle)

        var_set = driver_cls.get_setup_vars()
        self._connect_widget.populate(var_set)
        self._next_btn.set_sensitive(True)

    def _setup_discover_page(self):
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        status_group = Adw.PreferencesGroup(
            title=_("Discovering Device"),
            description=_(
                "Connecting to the device and reading its configuration."
            ),
        )
        page_box.append(status_group)

        self._error_row = Adw.ActionRow()
        self._error_row.set_visible(False)
        self._error_row.add_css_class("error")
        status_group.add(self._error_row)

        spinner_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=24,
        )
        self._discover_spinner = Gtk.Spinner()
        self._discover_spinner.set_halign(Gtk.Align.CENTER)
        self._discover_spinner.set_size_request(32, 32)
        self._discover_spinner.start()
        spinner_box.append(self._discover_spinner)
        page_box.append(spinner_box)

        self._stack.add_named(page_box, "discover")

    def _setup_review_page(self):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)

        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        scrolled.set_child(page_box)

        info_group = Adw.PreferencesGroup(title=_("Device Info"))
        page_box.append(info_group)

        self._info_name_row = Adw.ActionRow(title=_("Device Name"))
        info_group.add(self._info_name_row)

        self._info_desc_row = Adw.ActionRow(title=_("Description"))
        info_group.add(self._info_desc_row)

        self._info_fw_row = Adw.ActionRow(title=_("Firmware Version"))
        info_group.add(self._info_fw_row)

        self._info_rx_row = Adw.ActionRow(title=_("RX Buffer Size"))
        info_group.add(self._info_rx_row)

        self._info_arc_row = Adw.ActionRow(title=_("Arc Tolerance"))
        info_group.add(self._info_arc_row)

        area_group = Adw.PreferencesGroup(
            title=_("Working Area"),
            description=_("Axis travel in machine units."),
        )
        page_box.append(area_group)

        x_row = Adw.SpinRow(
            title=_("X Travel"),
            adjustment=Gtk.Adjustment(
                lower=1,
                upper=10000,
                step_increment=10,
                page_increment=100,
            ),
        )
        self._x_helper = UnitSpinRowHelper(
            spin_row=x_row,
            quantity="length",
            max_value_in_base=10000.0,
        )
        area_group.add(x_row)

        y_row = Adw.SpinRow(
            title=_("Y Travel"),
            adjustment=Gtk.Adjustment(
                lower=1,
                upper=10000,
                step_increment=10,
                page_increment=100,
            ),
        )
        self._y_helper = UnitSpinRowHelper(
            spin_row=y_row,
            quantity="length",
            max_value_in_base=10000.0,
        )
        area_group.add(y_row)

        speed_group = Adw.PreferencesGroup(
            title=_("Speed"),
            description=_("Speed limits in machine units per minute."),
        )
        page_box.append(speed_group)

        travel_row = Adw.SpinRow(
            title=_("Max Travel Speed"),
            adjustment=Gtk.Adjustment(
                lower=0,
                upper=60000,
                step_increment=100,
                page_increment=1000,
            ),
        )
        self._travel_speed_helper = UnitSpinRowHelper(
            spin_row=travel_row,
            quantity="speed",
            max_value_in_base=60000.0,
        )
        speed_group.add(travel_row)

        cut_row = Adw.SpinRow(
            title=_("Max Cut Speed"),
            adjustment=Gtk.Adjustment(
                lower=0,
                upper=60000,
                step_increment=100,
                page_increment=1000,
            ),
        )
        self._cut_speed_helper = UnitSpinRowHelper(
            spin_row=cut_row,
            quantity="speed",
            max_value_in_base=60000.0,
        )
        speed_group.add(cut_row)

        accel_group = Adw.PreferencesGroup(
            title=_("Acceleration"),
            description=_("Machine units per second squared."),
        )
        page_box.append(accel_group)

        accel_row = Adw.SpinRow(
            title=_("Acceleration"),
            adjustment=Gtk.Adjustment(
                lower=0,
                upper=10000,
                step_increment=10,
                page_increment=100,
            ),
        )
        self._accel_helper = UnitSpinRowHelper(
            spin_row=accel_row,
            quantity="acceleration",
            max_value_in_base=10000.0,
        )
        accel_group.add(accel_row)

        laser_group = Adw.PreferencesGroup(title=_("Laser"))
        page_box.append(laser_group)

        self._power_row = Adw.SpinRow(
            title=_("Max Power (S-value)"),
            subtitle=_("Maximum spindle speed / S-value range"),
            adjustment=Gtk.Adjustment(
                lower=1,
                upper=100000,
                step_increment=100,
                page_increment=1000,
            ),
        )
        laser_group.add(self._power_row)

        behavior_group = Adw.PreferencesGroup(title=_("Behavior"))
        page_box.append(behavior_group)

        self._home_row = Adw.SwitchRow(
            title=_("Home on Start"),
            subtitle=_("Run homing cycle when machine connects"),
        )
        behavior_group.add(self._home_row)

        self._single_axis_row = Adw.SwitchRow(
            title=_("Single-Axis Homing"),
            subtitle=_("Allow homing individual axes"),
        )
        behavior_group.add(self._single_axis_row)

        self._warning_group = Adw.PreferencesGroup(title=_("Warnings"))
        self._warning_group.set_visible(False)
        page_box.append(self._warning_group)

        self._stack.add_named(scrolled, "review")

    def _populate_review_page(
        self,
        profile: DeviceProfile,
        warnings: List[str],
    ):
        mc = profile.machine_config
        dc: Dict[str, Any] = mc.driver_config or {}

        self._info_name_row.set_subtitle(profile.meta.name or "")
        self._info_desc_row.set_subtitle(profile.meta.description or "")
        self._info_fw_row.set_subtitle(
            dc.get("firmware_version", _("Unknown"))
        )
        rx = dc.get("rx_buffer_size")
        self._info_rx_row.set_subtitle(
            str(rx) if rx is not None else _("Unknown")
        )
        arc = dc.get("arc_tolerance")
        self._info_arc_row.set_subtitle(
            f"{arc}" if arc is not None else _("Unknown")
        )

        if mc.axis_extents:
            self._x_helper.set_value_in_base_units(mc.axis_extents[0])
            self._y_helper.set_value_in_base_units(mc.axis_extents[1])
        else:
            self._x_helper.set_value_in_base_units(0)
            self._y_helper.set_value_in_base_units(0)

        self._travel_speed_helper.set_value_in_base_units(
            mc.max_travel_speed or 0
        )
        self._cut_speed_helper.set_value_in_base_units(mc.max_cut_speed or 0)
        self._accel_helper.set_value_in_base_units(mc.acceleration or 0)

        if mc.heads:
            self._power_row.set_value(mc.heads[0].get("max_power", 1000))
        else:
            self._power_row.set_value(1000)

        self._home_row.set_active(mc.home_on_start or False)
        self._single_axis_row.set_active(
            mc.single_axis_homing_enabled or False
        )

        self._populate_warnings(warnings)

    def _populate_warnings(self, warnings: List[str]):
        for row in self._warning_rows:
            self._warning_group.remove(row)
        self._warning_rows.clear()

        if not warnings:
            self._warning_group.set_visible(False)
            return

        for text in warnings:
            row = Adw.ActionRow(title=text)
            row.add_css_class("warning")
            self._warning_group.add(row)
            self._warning_rows.append(row)
        self._warning_group.set_visible(True)

    def _build_profile_from_ui(self) -> DeviceProfile:
        assert self._profile is not None
        mc = self._profile.machine_config

        x = self._x_helper.get_value_in_base_units()
        y = self._y_helper.get_value_in_base_units()
        mc.axis_extents = (x, y) if x > 0 and y > 0 else None

        mc.max_travel_speed = (
            int(self._travel_speed_helper.get_value_in_base_units()) or None
        )
        mc.max_cut_speed = (
            int(self._cut_speed_helper.get_value_in_base_units()) or None
        )
        mc.acceleration = (
            int(self._accel_helper.get_value_in_base_units()) or None
        )

        max_power = int(self._power_row.get_value())
        mc.heads = [{"max_power": max_power}] if max_power else None

        mc.home_on_start = self._home_row.get_active() or None
        mc.single_axis_homing_enabled = (
            self._single_axis_row.get_active() or None
        )

        return self._profile

    def _on_page_changed(self, stack, pspec):
        page = stack.get_visible_child_name()
        self._back_btn.set_visible(page == "review")
        self._cancel_btn.set_visible(True)
        self._next_btn.set_visible(page == "connect")
        self._create_btn.set_visible(page == "review")
        self._retry_btn.set_visible(False)

    def _on_back_clicked(self, button):
        self._stack.set_visible_child_name("connect")

    def _on_next_clicked(self, button):
        if self._stack.get_visible_child_name() == "connect":
            self._start_probe()

    def _on_retry_clicked(self, button):
        self._start_probe()

    def _on_create_clicked(self, button):
        if self._profile is None:
            return
        profile = self._build_profile_from_ui()
        self.profile_created.send(self, profile=profile)
        self.close()

    def _start_probe(self):
        if self._driver_cls is None:
            return

        self._error_row.set_visible(False)
        self._discover_spinner.set_visible(True)
        self._stack.set_visible_child_name("discover")

        params = self._connect_widget.get_values()

        try:
            self._driver_cls.precheck(**params)
        except Exception as e:
            self._show_probe_error(str(e))
            return

        assert self._driver_cls is not None
        driver_cls = self._driver_cls
        context = get_context()

        async def _probe_coroutine(exec_ctx):
            return await driver_cls.probe(context, **params)

        task_mgr.add_coroutine(
            _probe_coroutine,
            key="config_wizard_probe",
            when_done=self._on_probe_done,
        )

    def _on_probe_done(self, task):
        def _update_ui():
            if task.get_status() != "completed":
                try:
                    task.result()
                    msg = _("Probe failed")
                except Exception as e:
                    msg = str(e)
                self._show_probe_error(msg)
                return

            profile, warnings = task.result()
            self._profile = profile
            self._warnings = warnings
            self._populate_review_page(profile, warnings)
            self._stack.set_visible_child_name("review")

        task_mgr.schedule_on_main_thread(_update_ui)

    def _show_probe_error(self, message: str):
        self._discover_spinner.set_visible(False)
        self._error_row.set_title(_("Error: {}").format(message))
        self._error_row.set_visible(True)
        self._retry_btn.set_visible(True)

    def close(self):
        super().close()

    def do_close_request(self, *args) -> bool:
        return False
