from gi.repository import Gtk, Adw

from gettext import gettext as _

from rayforge.core.varset.hostnamevar import HostnameVar
from rayforge.core.varset.octoprintauthflowvar import OctoprintAuthFlowVar
from rayforge.core.varset.var import Var
from blinker import Signal


def escape_title(text: str) -> str:
    """Escape ampersands for GTK/Adwaita title display."""
    return text.replace("&", "&&")


class OctoprintAuthFlowRow(Adw.PreferencesGroup):
    def __init__(self, auth_flow_var: OctoprintAuthFlowVar):
        super().__init__(title=_("Authorization"))
        self.auth_flow_var = auth_flow_var
        self._add_host_port_rows()
        self._add_button_row()
        self.data_changed = Signal()

    def _add_host_port_rows(self):
        self.host_row = self._create_hostname_row(
            self.auth_flow_var.host_var, "value"
        )
        self.port_row = self._create_integer_row(
            self.auth_flow_var.port_var, "value"
        )
        self.add(self.host_row)
        self.add(self.port_row)
        self.add(self._create_raw_value_row())

    def _create_raw_value_row(self):
        row = Adw.ActionRow(title=_("API Key"))

        raw = self.auth_flow_var.raw_value or ""
        truncated = (raw[:20] + "...") if len(raw) > 20 else raw

        self.raw_value_label = Gtk.Label(
            label=truncated or _("Not set")
        )  # store ref here
        self.raw_value_label.set_selectable(False)
        self.raw_value_label.add_css_class("dim-label")
        self.raw_value_label.set_max_width_chars(24)
        self.raw_value_label.set_tooltip_text(raw)

        row.add_suffix(self.raw_value_label)
        self.raw_value_row = row
        return row

    def update_raw_value(self):
        raw = self.auth_flow_var.raw_value or ""
        truncated = (raw[:20] + "...") if len(raw) > 20 else raw
        label = self.raw_value_label
        label.set_text(truncated or _("Not set"))
        label.set_tooltip_text(raw)

    # From RowFactory
    def _create_hostname_row(self, var: HostnameVar, target_property: str):
        row = Adw.EntryRow(title=escape_title(var.label))
        if var.description:
            row.set_tooltip_text(var.description)
        row.set_show_apply_button(True)
        initial_val = getattr(var, target_property)
        if initial_val is not None:
            row.set_text(str(initial_val))
        row.connect("changed", lambda row: self._on_hostname_changed(var, row))
        return row

    def _on_hostname_changed(self, var: HostnameVar, row: Adw.EntryRow):
        var.value = row.get_text()
        self.data_changed.send(self)

    def _create_integer_row(self, var: Var, target_property: str):
        # Use getattr to safely handle generic Vars that don't have min/max
        min_val = getattr(var, "min_val", None)
        max_val = getattr(var, "max_val", None)
        lower = min_val if min_val is not None else -2147483647
        upper = max_val if max_val is not None else 2147483647

        initial_val = getattr(var, target_property)

        adj = Gtk.Adjustment(
            value=int(initial_val) if initial_val is not None else 0,
            lower=lower,
            upper=upper,
            step_increment=1,
        )
        row = Adw.SpinRow(adjustment=adj, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        row.connect(
            "notify::value",
            lambda r, _: self._on_port_changed(var, row),
        )
        return row

    def _on_port_changed(self, var: Var, row: Adw.SpinRow):
        var.value = int(row.get_value())
        self.data_changed.send(self)

    def _add_button_row(self):
        row = Adw.ActionRow(title="Authenticate with Octoprint")
        button = Gtk.Button(label="Start Auth Flow")
        button.connect("clicked", lambda _: self.on_authorize_clicked())
        row.add_suffix(button)
        row.set_activatable_widget(button)
        self.add(row)

    def on_authorize_clicked(self):
        self.auth_flow_var._on_click(self.on_auth_finished)

    def on_auth_finished(self):
        self.data_changed.send(self)
        self.update_raw_value()
