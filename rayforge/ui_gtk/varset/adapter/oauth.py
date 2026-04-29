import json
import logging
from gettext import gettext as _
from typing import Any, Dict, Optional, Tuple, Union

from gi.repository import Adw, GLib, Gtk

from ....core.varset import OAuthFlowVar, Var
from ....shared.oauth.flow import OAuthFlow, OAuthFlowConfig
from .base import RowAdapter, escape_title, register_adapter

logger = logging.getLogger(__name__)


@register_adapter(OAuthFlowVar)
class OAuthFlowAdapter(RowAdapter):
    """
    Adapter that renders an OAuthFlowVar.

    When all URL fields are provided, uses a simple ActionRow with
    the auth status as subtitle and a Sign In / Sign Out button as
    suffix.  When URL fields are ``None``, uses an ExpanderRow so
    the user can fill in the missing values before authenticating.
    """

    has_natural_commit = True

    _FIELD_DEFS = (
        ("authorize_url", _("Authorize URL")),
        ("token_url", _("Token URL")),
        ("client_id", _("Client ID")),
    )

    def __init__(
        self,
        row: Union[Adw.ActionRow, Adw.ExpanderRow],
        sign_in_btn: Gtk.Button,
        sign_out_btn: Gtk.Button,
        var: OAuthFlowVar,
        dynamic_entries: Optional[Dict[str, Adw.EntryRow]] = None,
    ):
        super().__init__()
        self._row = row
        self._sign_in_btn = sign_in_btn
        self._sign_out_btn = sign_out_btn
        self._var: OAuthFlowVar = var
        self._dynamic_entries = dynamic_entries or {}
        self._token_value: str = var.value or ""

        self._sign_in_btn.connect("clicked", self._on_sign_in)
        self._sign_out_btn.connect("clicked", self._on_sign_out)

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "OAuthFlowAdapter"]:
        oauth_var = var
        assert isinstance(oauth_var, OAuthFlowVar)

        needs_entries = any(
            getattr(oauth_var, key, None) is None for key, _ in cls._FIELD_DEFS
        )

        dynamic_entries: Dict[str, Adw.EntryRow] = {}
        if needs_entries:
            row = cls._create_expander_row(oauth_var, dynamic_entries)
        else:
            row = cls._create_action_row(oauth_var)

        # --- Buttons (suffixes on the main row) ---
        sign_in_btn = Gtk.Button(
            label=_("Sign In"),
            css_classes=["pill", "suggested-action"],
            valign=Gtk.Align.CENTER,
        )
        row.add_suffix(sign_in_btn)

        sign_out_btn = Gtk.Button(
            label=_("Sign Out"),
            css_classes=["pill", "destructive-action"],
            valign=Gtk.Align.CENTER,
            visible=False,
        )
        row.add_suffix(sign_out_btn)

        adapter = cls(
            row,
            sign_in_btn,
            sign_out_btn,
            oauth_var,
            dynamic_entries=dynamic_entries,
        )
        adapter._update_status()
        return row, adapter

    @classmethod
    def _create_action_row(cls, oauth_var: OAuthFlowVar) -> Adw.ActionRow:
        row = Adw.ActionRow(title=escape_title(oauth_var.label))
        if oauth_var.description:
            row.set_subtitle(oauth_var.description)
        return row

    @classmethod
    def _create_expander_row(
        cls,
        oauth_var: OAuthFlowVar,
        dynamic_entries: Dict[str, Adw.EntryRow],
    ) -> Adw.ExpanderRow:
        row = Adw.ExpanderRow(title=escape_title(oauth_var.label))
        if oauth_var.description:
            row.set_subtitle(oauth_var.description)

        for field_key, field_label in cls._FIELD_DEFS:
            if getattr(oauth_var, field_key, None) is None:
                entry_row = Adw.EntryRow(title=field_label)
                row.add_row(entry_row)
                dynamic_entries[field_key] = entry_row

        return row

    # ------------------------------------------------------------------
    # RowAdapter interface
    # ------------------------------------------------------------------

    def get_value(self) -> Optional[Any]:
        return self._token_value

    def set_value(self, value: Any) -> None:
        self._token_value = str(value) if value is not None else ""
        self._update_status()

    def update_from_var(self, var: Var) -> None:
        if not isinstance(var, OAuthFlowVar):
            return
        self._var = var
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(var.description)
        self._token_value = var.value or ""
        self._update_status()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_token_data(self) -> Optional[Dict[str, Any]]:
        if not self._token_value:
            return None
        try:
            return json.loads(self._token_value)
        except (json.JSONDecodeError, TypeError):
            return None

    def _update_status(self) -> None:
        tokens = self._get_token_data()
        if tokens and tokens.get("access_token"):
            if self._var._is_expired(tokens):
                self._row.set_subtitle(_("Token expired"))
                self._sign_in_btn.set_label(_("Refresh"))
                self._sign_out_btn.set_visible(True)
            else:
                self._row.set_subtitle(_("Authenticated"))
                self._sign_in_btn.set_label(_("Re-authorize"))
                self._sign_out_btn.set_visible(True)
        else:
            desc = self._var.description
            self._row.set_subtitle(desc if desc else _("Not connected"))
            self._sign_in_btn.set_label(_("Sign In"))
            self._sign_out_btn.set_visible(False)

    def _collect_overrides(self) -> Dict[str, str]:
        overrides: Dict[str, str] = {}
        for field_key, entry_row in self._dynamic_entries.items():
            text = entry_row.get_text().strip()
            if text:
                overrides[field_key] = text
        return overrides

    def _on_sign_in(self, _btn) -> None:
        refresh_token = self._var.get_refresh_token()
        if refresh_token:
            self._try_refresh(refresh_token)
        else:
            self._start_full_flow()

    def _try_refresh(self, refresh_token: str) -> None:
        config_dict = self._var.resolve_config(self._collect_overrides())
        token_url = config_dict.get("token_url")
        client_id = config_dict.get("client_id")
        if not token_url or not client_id:
            self._start_full_flow()
            return

        config = OAuthFlowConfig(
            authorize_url=config_dict.get("authorize_url", ""),
            token_url=token_url,
            client_id=client_id,
            client_secret=config_dict.get("client_secret"),
            redirect_port=config_dict.get("redirect_port", 8765),
        )
        flow = OAuthFlow(config)
        self._sign_in_btn.set_sensitive(False)
        self._sign_in_btn.set_label(_("Refreshing…"))

        def on_refresh_complete(result):
            GLib.idle_add(self._on_flow_complete, result)

        def on_refresh_error(error):
            logger.debug(
                "Token refresh failed, starting full flow: %s",
                error,
            )
            GLib.idle_add(self._start_full_flow)

        try:
            result = flow.refresh(refresh_token)
            self._on_flow_complete(result)
        except Exception:
            self._start_full_flow()

    def _start_full_flow(self) -> None:
        config_dict = self._var.resolve_config(self._collect_overrides())

        authorize_url = config_dict.get("authorize_url")
        token_url = config_dict.get("token_url")
        client_id = config_dict.get("client_id")

        if not authorize_url or not token_url or not client_id:
            logger.warning(
                "Cannot start OAuth flow: missing authorize_url, "
                "token_url, or client_id"
            )
            return

        config = OAuthFlowConfig(
            authorize_url=authorize_url,
            token_url=token_url,
            client_id=client_id,
            client_secret=config_dict.get("client_secret"),
            scopes=config_dict.get("scopes", []),
            redirect_port=config_dict.get("redirect_port", 8765),
        )

        flow = OAuthFlow(config)
        self._sign_in_btn.set_sensitive(False)
        self._sign_in_btn.set_label(_("Waiting…"))

        def on_complete(result):
            GLib.idle_add(self._on_flow_complete, result)

        def on_error(error):
            GLib.idle_add(self._on_flow_error, error)

        flow.start(on_complete, on_error)

    def _on_flow_complete(self, result) -> bool:
        self._sign_in_btn.set_sensitive(True)
        token_data: Dict[str, Any] = {
            "access_token": result.access_token,
        }
        if result.refresh_token:
            token_data["refresh_token"] = result.refresh_token
        if result.expires_at:
            token_data["expires_at"] = result.expires_at.isoformat()
        if result.scope:
            token_data["scope"] = result.scope

        self._token_value = json.dumps(token_data)
        self._update_status()
        self.changed.send(self)
        return False

    def _on_flow_error(self, error) -> bool:
        logger.error("OAuth flow failed: %s", error)
        self._sign_in_btn.set_sensitive(True)
        self._update_status()
        return False

    def _on_sign_out(self, _btn) -> None:
        self._token_value = ""
        self._update_status()
        self.changed.send(self)
