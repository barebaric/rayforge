"""
Add/edit dialog for a single tool.

Follows the recipe-editor convention: a :class:`PatchedDialogWindow`
with an ``Adw.ToolbarView`` header, a custom toggle-button tab bar in
the header, and an ``Adw.ViewStack`` of ``Adw.PreferencesPage`` tabs.
The geometry rows are rebuilt from
:data:`~tool_library.tool.CATEGORY_PARAMS` when the category changes, so
the editor only asks for the attributes relevant to the selected shape.
Length fields use
:class:`~rayforge.ui_gtk.shared.pref_rows.length_choice_spin_row.LengthChoiceSpinRow`
so they are stored in base mm but shown in a per-row unit chosen by the
user via the dropdown (defaulting to their preferred unit).
"""

from gettext import gettext as _
from typing import Optional, cast

from blinker import Signal
from gi.repository import Adw, Gtk
from raygeo.cnc.tool import ToolModel

from rayforge.ui_gtk.icons import get_icon
from rayforge.ui_gtk.shared.patched_dialog_window import PatchedDialogWindow
from rayforge.ui_gtk.shared.pref_rows import (
    LengthChoiceSpinRow,
    SpinRow,
)

from .tool import (
    CATEGORY_BY_NAME,
    CATEGORY_LABELS,
    CATEGORY_NAMES,
    CATEGORY_PARAMS,
    TOOL_MATERIAL_BY_NAME,
    TOOL_MATERIAL_LABELS,
    TOOL_MATERIAL_NAMES,
    ParamSpec,
    Tool,
    category_to_name,
    tool_material_to_name,
)


class AddEditToolDialog(PatchedDialogWindow):
    """Add or edit a :class:`Tool`."""

    def __init__(
        self,
        parent: Optional[Gtk.Window],
        tool: Optional[Tool] = None,
    ):
        super().__init__(transient_for=parent, modal=True)
        self.response = Signal()
        self._tool = tool
        is_editing = tool is not None
        self.set_title(_("Edit Tool") if is_editing else _("Add Tool"))
        self.set_default_size(560, 500)
        self._positive = "save" if is_editing else "add"

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)
        self.set_content(toolbar)

        cancel_btn = Gtk.Button(label=_("Cancel"))
        cancel_btn.connect("clicked", lambda _w: self._send_response("cancel"))
        header.pack_start(cancel_btn)

        save_btn = Gtk.Button(label=_("Save") if is_editing else _("Add"))
        save_btn.add_css_class("suggested-action")
        save_btn.connect(
            "clicked", lambda _w: self._send_response(self._positive)
        )
        header.pack_end(save_btn)

        self.view_stack = Adw.ViewStack()
        toolbar.set_content(self.view_stack)

        self.switcher_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.switcher_box.add_css_class("linked")
        header.set_title_widget(self.switcher_box)
        self._tab_buttons: dict[str, Gtk.ToggleButton] = {}

        self._params = tool.model.get_parameters() if tool else {}

        self._build_general_page(tool)
        self._build_geometry_page(tool)
        self._build_setup_page(tool)

        self._tab_buttons["general"].set_active(True)

    # --- Tab wiring -----------------------------------------------------

    def _create_tab_child(self, text: str, icon_name: str) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.append(get_icon(icon_name))
        box.append(Gtk.Label(label=text))
        return box

    def _add_page(
        self,
        page: Gtk.Widget,
        name: str,
        title: str,
        icon_name: str,
    ):
        group = self._tab_buttons["general"] if self._tab_buttons else None
        button = Gtk.ToggleButton(group=group) if group else Gtk.ToggleButton()
        button.set_child(self._create_tab_child(title, icon_name))
        button.connect("toggled", self._on_tab_toggled, name)
        self.switcher_box.append(button)
        self.view_stack.add_named(page, name)
        self._tab_buttons[name] = button

    def _on_tab_toggled(self, button: Gtk.ToggleButton, page_name: str):
        if button.get_active():
            self.view_stack.set_visible_child_name(page_name)

    # --- Pages ----------------------------------------------------------

    def _build_general_page(self, tool: Optional[Tool]) -> None:
        page = Adw.PreferencesPage()
        group = Adw.PreferencesGroup(
            title=_("General"),
            description=_("Identity and coating of the tool."),
        )
        page.add(group)
        self._name = Adw.EntryRow(title=_("Name"))
        self._label = Adw.EntryRow(title=_("Label"))
        self._coating = Adw.EntryRow(title=_("Coating (optional)"))
        group.add(self._name)
        group.add(self._label)
        group.add(self._coating)

        if tool is not None:
            self._name.set_text(tool.name)
            self._label.set_text(tool.label)
            if tool.coating:
                self._coating.set_text(tool.coating)

        for entry in (self._name, self._label, self._coating):
            entry.connect(
                "apply", lambda _w: self._send_response(self._positive)
            )

        self._add_page(page, "general", _("General"), "general-symbolic")
        self._name.grab_focus()

    def _build_geometry_page(self, tool: Optional[Tool]) -> None:
        page = Adw.PreferencesPage()
        self._geom = Adw.PreferencesGroup(
            title=_("Geometry"),
            description=_("Cutting shape, tool material, and dimensions."),
        )
        page.add(self._geom)

        category_labels = [CATEGORY_LABELS[n] for n in CATEGORY_NAMES]
        self._category = Adw.ComboRow(
            title=_("Category"),
            subtitle=_("Classification used for operation compatibility"),
            model=Gtk.StringList.new(category_labels),
        )
        self._category.set_selected(
            CATEGORY_NAMES.index(
                category_to_name(tool.category) if tool else CATEGORY_NAMES[0]
            )
        )
        self._geom.add(self._category)

        material_labels = [
            TOOL_MATERIAL_LABELS[n] for n in TOOL_MATERIAL_NAMES
        ]
        self._tool_material = Adw.ComboRow(
            title=_("Tool Material"),
            subtitle=_("Substrate the tool is made of"),
            model=Gtk.StringList.new(material_labels),
        )
        self._tool_material.set_selected(
            TOOL_MATERIAL_NAMES.index(
                tool_material_to_name(tool.tool_material)
                if tool
                else TOOL_MATERIAL_NAMES[0]
            )
        )
        self._geom.add(self._tool_material)

        self._param_rows: dict[str, Gtk.Widget] = {}
        self._param_helpers: dict[str, LengthChoiceSpinRow] = {}
        self._rebuild_params(self._params)
        self._category.connect(
            "notify::selected", lambda *_: self._on_category_changed()
        )

        self._add_page(page, "geometry", _("Geometry"), "tool-change-symbolic")

    def _build_setup_page(self, tool: Optional[Tool]) -> None:
        page = Adw.PreferencesPage()
        setup = Adw.PreferencesGroup(
            title=_("Setup"),
            description=_("Holder protrusion and spindle limits."),
        )
        page.add(setup)
        self._stickout_row, self._stickout_helper = self._add_length(
            setup,
            _("Stickout"),
            _("Protrusion from the holder"),
            200.0,
            tool.stickout if tool else None,
        )
        self._max_rpm = self._add_plain(
            setup,
            _("Max RPM"),
            _("Spindle speed cap; steps clamp to this"),
            60000.0,
            tool.max_rpm if tool else None,
            0,
        )

        self._add_page(page, "setup", _("Setup"), "step-settings-symbolic")

    # --- Geometry param rows --------------------------------------------

    def _on_category_changed(self) -> None:
        self._rebuild_params(self._read_params())

    def _rebuild_params(self, values: dict[str, float]) -> None:
        for row in self._param_rows.values():
            self._geom.remove(row)
        self._param_rows.clear()
        self._param_helpers.clear()

        cat = CATEGORY_NAMES[self._category.get_selected()]
        for spec in CATEGORY_PARAMS.get(cat, []):
            value = values.get(spec.key)
            row, helper = self._make_param_row(spec, value)
            self._geom.add(row)
            self._param_rows[spec.key] = row
            if helper is not None:
                self._param_helpers[spec.key] = helper

    def _make_param_row(
        self,
        spec: ParamSpec,
        value: Optional[float],
    ) -> tuple[Gtk.Widget, Optional[LengthChoiceSpinRow]]:
        if spec.quantity == "length":
            return self._add_length(
                None, spec.title, spec.subtitle, spec.upper, value
            )
        return (
            self._add_plain(
                None,
                spec.title,
                spec.subtitle,
                spec.upper,
                value,
                spec.digits,
                spec.is_int,
            ),
            None,
        )

    def _add_length(
        self,
        group: Optional[Adw.PreferencesGroup],
        title: str,
        subtitle: str,
        upper: float,
        value: Optional[float] = None,
    ) -> tuple[LengthChoiceSpinRow, LengthChoiceSpinRow]:
        row = LengthChoiceSpinRow(
            title=title,
            subtitle=subtitle,
            upper=upper,
        )
        if value is not None:
            row.set_value_in_base_units(float(value))
        if group is not None:
            group.add(row)
        return row, row

    def _add_plain(
        self,
        group: Optional[Adw.PreferencesGroup],
        title: str,
        subtitle: str,
        upper: float,
        value: Optional[float] = None,
        digits: int = 1,
        is_int: bool = False,
    ) -> SpinRow:
        step = 1.0 if is_int or digits == 0 else 0.1
        row = SpinRow(
            title,
            subtitle,
            lower=0.0,
            upper=upper,
            step_increment=step,
            digits=digits,
            numeric=True,
            value=float(value) if value is not None else None,
        )
        if group is not None:
            group.add(row)
        return row

    # --- Result ----------------------------------------------------------

    def param_keys(self) -> list[str]:
        """Return the geometry param keys shown for the current category."""
        return list(self._param_rows)

    def is_length_param(self, key: str) -> bool:
        """True if ``key`` is a unit-aware (length) geometry field."""
        return key in self._param_helpers

    def _read_params(self) -> dict[str, float]:
        params: dict[str, float] = {}
        for key, row in self._param_rows.items():
            helper = self._param_helpers.get(key)
            if helper is not None:
                params[key] = helper.get_value_in_base_units()
            else:
                params[key] = float(cast(SpinRow, row).get_value())
        return params

    def _send_response(self, response_id: str) -> None:
        self.response.send(self, response_id=response_id)

    def get_tool(self) -> Tool:
        """Build a :class:`Tool` from the dialog fields."""
        params = self._read_params()
        return Tool(
            uid=self._tool.uid if self._tool else Tool.create_default().uid,
            name=self._name.get_text().strip() or _("Unnamed Tool"),
            max_rpm=float(self._max_rpm.get_value()),
            label=self._label.get_text().strip(),
            category=CATEGORY_BY_NAME[
                CATEGORY_NAMES[self._category.get_selected()]
            ],
            tool_material=TOOL_MATERIAL_BY_NAME[
                TOOL_MATERIAL_NAMES[self._tool_material.get_selected()]
            ],
            stickout=self._stickout_helper.get_value_in_base_units(),
            coating=self._coating.get_text().strip() or None,
            model=ToolModel(**params),
        )
