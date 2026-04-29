import json
import logging
import urllib.request
import urllib.error
from gettext import gettext as _
from typing import Any, Optional, Tuple

from gi.repository import Adw, GLib, Gtk

from ....core.varset import AppKeyVar, Var
from .base import RowAdapter, escape_title, register_adapter

logger = logging.getLogger(__name__)

_POLL_INTERVAL_MS = 2000
_REQUEST_TIMEOUT = 10


@register_adapter(AppKeyVar)
class AppKeyAdapter(RowAdapter):
    """
    Adapter that renders an AppKeyVar.

    Shows an ExpanderRow with a text entry for manual API key input
    and a "Request Access" button that initiates the decision-based
    key approval flow (probe → request → poll).
    """

    has_natural_commit = True

    def __init__(
        self,
        row: Adw.ExpanderRow,
        entry_row: Adw.EntryRow,
        request_btn: Gtk.Button,
        clear_btn: Gtk.Button,
        var: AppKeyVar,
    ):
        super().__init__()
        self._row = row
        self._entry_row = entry_row
        self._request_btn = request_btn
        self._clear_btn = clear_btn
        self._var: AppKeyVar = var
        self._key_value: str = var.value or ""
        self._poll_source_id: Optional[int] = None

        self._entry_row.connect("apply", self._on_entry_apply)
        self._request_btn.connect("clicked", self._on_request)
        self._clear_btn.connect("clicked", self._on_clear)

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "AppKeyAdapter"]:
        app_var = var
        assert isinstance(app_var, AppKeyVar)

        row = Adw.ExpanderRow(
            title=escape_title(app_var.label)
        )
        if app_var.description:
            row.set_subtitle(app_var.description)

        entry_row = Adw.EntryRow(
            title=_("API Key"),
        )
        entry_row.set_show_apply_button(True)
        row.add_row(entry_row)

        btn_box = Gtk.Box(
            spacing=6,
            valign=Gtk.Align.CENTER,
            margin_top=8,
            margin_bottom=8,
            margin_start=12,
            margin_end=12,
        )

        request_btn = Gtk.Button(
            label=_("Request Access"),
            css_classes=["pill", "suggested-action"],
        )
        btn_box.append(request_btn)

        clear_btn = Gtk.Button(
            label=_("Clear"),
            css_classes=["pill"],
            visible=False,
        )
        btn_box.append(clear_btn)

        row.add_row(btn_box)

        adapter = cls(
            row, entry_row, request_btn, clear_btn, app_var
        )
        adapter._update_status()
        return row, adapter

    def get_value(self) -> Optional[Any]:
        return self._key_value

    def set_value(self, value: Any) -> None:
        self._key_value = str(value) if value is not None else ""
        self._update_status()

    def update_from_var(self, var: Var) -> None:
        if not isinstance(var, AppKeyVar):
            return
        self._var = var
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(var.description)
        self._key_value = var.value or ""
        self._update_status()

    def _get_key(self) -> Optional[str]:
        if not self._key_value:
            return None
        try:
            data = json.loads(self._key_value)
            if isinstance(data, dict):
                return data.get("api_key")
            return str(data)
        except (json.JSONDecodeError, TypeError):
            return self._key_value.strip() or None

    def _update_status(self) -> None:
        api_key = self._get_key()
        if api_key:
            self._row.set_subtitle(_("API key configured"))
            self._clear_btn.set_visible(True)
            self._request_btn.set_label(_("Request New Key"))
            self._entry_row.set_text(api_key)
        else:
            desc = self._var.description
            self._row.set_subtitle(
                desc if desc else _("No API key configured")
            )
            self._clear_btn.set_visible(False)
            self._request_btn.set_label(_("Request Access"))
            self._entry_row.set_text("")

    def _on_entry_apply(self, entry_row) -> None:
        text = entry_row.get_text().strip()
        if text:
            self._key_value = json.dumps({"api_key": text})
            self._update_status()
            self.changed.send(self)

    def _on_clear(self, _btn) -> None:
        self._stop_polling()
        self._key_value = ""
        self._update_status()
        self.changed.send(self)

    def _resolve_base_url(self) -> Optional[str]:
        config = self._var.resolve_config()
        request_url = config.get("request_url")
        if not request_url:
            return None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(request_url)
            scheme = parsed.scheme or "http"
            return f"{scheme}://{parsed.netloc}"
        except Exception:
            return None

    def _on_request(self, _btn) -> None:
        config = self._var.resolve_config()
        probe_url = config.get("probe_url")
        request_url = config.get("request_url")
        app_name = config.get("app_name", "RayForge")

        if not request_url:
            self._row.set_subtitle(
                _("Hostname and port must be configured first")
            )
            return

        if probe_url:
            try:
                req = urllib.request.Request(
                    probe_url, method="GET"
                )
                with urllib.request.urlopen(
                    req, timeout=_REQUEST_TIMEOUT
                ):
                    pass
            except Exception:
                self._row.set_subtitle(
                    _(
                        "Device not reachable or does not support "
                        "automatic key requests"
                    )
                )
                return

        try:
            data = json.dumps({"app": app_name}).encode()
            req = urllib.request.Request(
                request_url,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(
                req, timeout=_REQUEST_TIMEOUT
            ) as resp:
                result = json.loads(resp.read().decode())
            app_token = result.get("app_token")
            if not app_token:
                self._row.set_subtitle(
                    _("Unexpected response from device")
                )
                return
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self._row.set_subtitle(
                    _("Too many requests. Try again later.")
                )
            else:
                self._row.set_subtitle(
                    _("Request failed: {code}").format(code=e.code)
                )
            return
        except Exception as e:
            self._row.set_subtitle(
                _("Connection failed: {err}").format(err=str(e))
            )
            return

        poll_url = config.get("poll_url")
        if not poll_url:
            return
        resolved_poll = poll_url.replace("{app_token}", app_token)

        self._row.set_subtitle(
            _("Waiting for approval on device…")
        )
        self._request_btn.set_sensitive(False)
        self._request_btn.set_label(_("Waiting…"))
        self._start_polling(resolved_poll)

    def _start_polling(self, poll_url: str) -> None:
        self._stop_polling()
        self._poll_url = poll_url
        self._poll_count = 0
        self._poll_source_id = GLib.timeout_add(
            _POLL_INTERVAL_MS, self._poll_tick
        )

    def _stop_polling(self) -> None:
        if self._poll_source_id is not None:
            GLib.source_remove(self._poll_source_id)
            self._poll_source_id = None
        self._request_btn.set_sensitive(True)
        self._request_btn.set_label(_("Request Access"))

    def _poll_tick(self) -> bool:
        self._poll_count += 1
        if self._poll_count > 150:
            self._row.set_subtitle(
                _("Approval timed out. Please try again.")
            )
            self._stop_polling()
            return False

        try:
            req = urllib.request.Request(
                self._poll_url, method="GET"
            )
            with urllib.request.urlopen(
                req, timeout=_REQUEST_TIMEOUT
            ) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self._row.set_subtitle(
                    _("Request denied or expired.")
                )
                self._stop_polling()
                return False
            return True
        except Exception:
            return True

        api_key = result.get("api_key")
        if api_key:
            self._key_value = json.dumps({"api_key": api_key})
            self._stop_polling()
            self._update_status()
            self.changed.send(self)
            return False

        return True
