"""A multi-select dialog for choosing step types (recipe targeting)."""

from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...core.step_registry import step_registry
from ..icons import get_icon
from ..shared.gtk import apply_css

css = """
.step-type-selector-list {
    background: none;
}
"""


class StepTypeSelectionDialog(Adw.MessageDialog):
    """A searchable, multi-select list of registered step types.

    Mirrors the look of :class:`RecipeSelectorDialog`. Each row shows
    the step's icon and ``TYPELABEL`` with a check button suffix. The
    selection is returned (in list order) via ``on_select_callback``.
    """

    class _StepTypeRow(Adw.ActionRow):
        def __init__(self, step_class_name: str, **kwargs):
            super().__init__(**kwargs)
            self.step_type = step_class_name

    def __init__(
        self,
        parent: Gtk.Window | None,
        selected: set[str],
        on_select_callback: Callable[[list[str]], None],
    ):
        super().__init__(transient_for=parent)
        self.on_select_callback = on_select_callback
        self._selected: set[str] = set(selected)

        self.set_heading(_("Select Step Types"))
        self.set_body(_("Choose which step types this recipe applies to."))

        apply_css(css)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_top(12)
        self.set_extra_child(content_box)

        self.search_entry = Gtk.SearchEntry(placeholder_text=_("Search..."))
        self.search_entry.connect(
            "search-changed", lambda *_: self._filter_list()
        )
        content_box.append(self.search_entry)

        scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            min_content_height=300,
            vexpand=True,
        )
        scrolled_window.add_css_class("card")
        content_box.append(scrolled_window)

        self.list_box = Gtk.ListBox()
        self.list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.list_box.add_css_class("step-type-selector-list")
        scrolled_window.set_child(self.list_box)

        self._all_rows: list[StepTypeSelectionDialog._StepTypeRow] = []
        self._populate()

        self.add_response("cancel", _("Cancel"))
        self.add_response("apply", _("Apply"))
        self.set_response_appearance("apply", Adw.ResponseAppearance.SUGGESTED)
        self.set_default_response("apply")
        self.connect("response", self._on_response)

    def _populate(self):
        step_classes = [
            cls for cls in step_registry.all_steps().values() if not cls.HIDDEN
        ]
        step_classes.sort(key=lambda c: c.TYPELABEL or c.__name__)

        for cls in step_classes:
            name = cls.__name__
            row = self._StepTypeRow(
                step_class_name=name,
                title=cls.TYPELABEL or name,
            )

            icon = get_icon(cls.ICON or "step-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            row.add_prefix(icon)

            check = Gtk.CheckButton()
            check.set_active(name in self._selected)
            check.set_valign(Gtk.Align.CENTER)
            check.connect("notify::active", self._on_check_toggled, name)
            row.add_suffix(check)

            self.list_box.append(row)
            self._all_rows.append(row)

    def _on_check_toggled(self, check: Gtk.CheckButton, _pspec, name: str):
        if check.get_active():
            self._selected.add(name)
        else:
            self._selected.discard(name)

    def _filter_list(self):
        search_text = self.search_entry.get_text().lower()
        for row in self._all_rows:
            label = row.get_title().lower()
            row.set_visible(not search_text or search_text in label)

    def _on_response(self, _dialog, response_id: str):
        if response_id == "apply":
            # Preserve the visible list order.
            ordered = [
                row.step_type
                for row in self._all_rows
                if row.step_type in self._selected
            ]
            self.on_select_callback(ordered)
        self.close()
