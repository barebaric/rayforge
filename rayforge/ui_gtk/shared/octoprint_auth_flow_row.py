from gi.repository import Gtk, Adw

from gettext import gettext as _
from rayforge.core.varset.hostnamevar import HostnameVar
from rayforge.core.varset.octoprintauthflowvar import OctoprintAuthFlowVar
from rayforge.core.varset.var import Var


def escape_title(text: str) -> str:
    """Escape ampersands for GTK/Adwaita title display."""
    return text.replace("&", "&&")


class OctoprintAuthFlowRow(Adw.PreferencesGroup):
    def __init__(self, auth_flow_var: OctoprintAuthFlowVar):
        super().__init__(title=_("Authorization"))
        self.auth_flow_var = auth_flow_var
        self._add_host_port_rows()
        self._add_button_row()

    def _add_host_port_rows(self):
        host_row = self._create_hostname_row(
            self.auth_flow_var.host_var, "value"
        )
        port_row = self._create_integer_row(
            self.auth_flow_var.port_var, "value"
        )
        self.add(host_row)
        self.add(port_row)

    # From RowFactory
    def _create_hostname_row(self, var: HostnameVar, target_property: str):
        row = Adw.EntryRow(title=escape_title(var.label))
        if var.description:
            row.set_tooltip_text(var.description)
        row.set_show_apply_button(True)
        initial_val = getattr(var, target_property)
        if initial_val is not None:
            row.set_text(str(initial_val))
        row.connect(
            "apply", lambda _: setattr(var, target_property, row.get_text())
        )
        return row

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
            lambda _, __: setattr(var, target_property, int(adj.get_value())),
        )
        return row

    def _add_button_row(self):
        row = Adw.ActionRow(title="Authenticate with Octoprint")
        button = Gtk.Button(label="Start Auth Flow")
        button.connect("clicked", lambda _: self.on_authorize_clicked())
        row.add_suffix(button)
        row.set_activatable_widget(button)
        self.add(row)

    def on_authorize_clicked(self):
        self.auth_flow_var._on_click()
