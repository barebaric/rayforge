import logging
import threading
import webbrowser
from gettext import gettext as _
from typing import Callable, List, Optional, cast

from gi.repository import Adw, GLib, Gtk

from ...context import get_context
from ...license.gumroad_provider import GumroadProvider


logger = logging.getLogger(__name__)


class LicenseEntryDialog(Adw.MessageDialog):
    """
    Dialog for entering a license key.

    Product ID is hidden - user only sees the license key field.
    """

    def __init__(
        self,
        product_ids: List[str],
        addon_name: str,
        on_success: Optional[Callable[[], None]] = None,
    ):
        super().__init__()

        self.product_ids = product_ids
        self.addon_name = addon_name
        self.on_success_callback = on_success
        self._is_validating = False

        self.set_heading(_("Enter License Key"))
        self.set_body(self._get_body_text())
        self.license_key_entry = Adw.EntryRow(title=_("License Key"))
        self.set_extra_child(self.license_key_entry)

        self.add_response("cancel", _("Cancel"))
        self.add_response("add", _("Activate"))
        self.set_response_appearance("add", Adw.ResponseAppearance.SUGGESTED)
        self.set_default_response("add")
        self.set_close_response("cancel")

        self.connect("response", self._on_response)

    def _get_body_text(self) -> str:
        return _(
            "Enter the license key you received when purchasing {addon_name}."
        ).format(addon_name=self.addon_name)

    def _on_response(self, dialog, response_id):
        if response_id != "add" or self._is_validating:
            return

        license_key = self.license_key_entry.get_text().strip()
        if not license_key:
            self._show_error(_("Please enter a license key."))
            return

        self._start_validation(license_key)

    def _start_validation(self, license_key: str):
        self._is_validating = True
        self.set_body(_("Validating license..."))
        self.set_sensitive(False)

        thread = threading.Thread(
            target=self._validate_license,
            args=(license_key,),
            daemon=True,
        )
        thread.start()

    def _validate_license(self, license_key: str):
        validator = get_context().license_validator
        gumroad = validator.get_provider("gumroad")
        success = False
        error_message = None

        if isinstance(gumroad, GumroadProvider):
            for product_id in self.product_ids:
                result = gumroad.validate_key(product_id, license_key)

                if result.status.value == "valid":
                    validator.add_gumroad_license(product_id, license_key)
                    success = True
                    break

                error_message = result.message

        GLib.idle_add(self._on_validation_complete, success, error_message)

    def _on_validation_complete(
        self, success: bool, error_message: Optional[str]
    ):
        self._is_validating = False
        self.set_sensitive(True)
        self.set_body(self._get_body_text())

        if success:
            self._handle_success()
        else:
            self._show_error(error_message or _("License validation failed."))

    def _handle_success(self):
        if self.on_success_callback:
            self.on_success_callback()
        self.close()

    def _show_error(self, message: str):
        error_dialog = Adw.MessageDialog(
            transient_for=cast(Optional[Gtk.Window], self.get_transient_for()),
            modal=True,
            heading=_("License Invalid"),
            body=message,
        )
        error_dialog.add_response("ok", _("OK"))
        error_dialog.present()


class LicenseRequiredDialog(Adw.MessageDialog):
    """
    Dialog shown when trying to install/use a premium addon without license.

    Provides options to buy or enter a license key.
    """

    def __init__(
        self,
        addon_name: str,
        product_ids: List[str],
        purchase_url: Optional[str],
        on_license_added: Optional[Callable[[], None]] = None,
    ):
        super().__init__()

        self.product_ids = product_ids
        self.purchase_url = purchase_url
        self.addon_name = addon_name
        self.on_license_added = on_license_added

        self.set_heading(_("License Required"))
        self.set_body(
            _(
                "{addon_name} is a premium addon. Purchase a license "
                "to unlock it, or enter your license key if you "
                "already have one."
            ).format(addon_name=addon_name)
        )

        self.add_response("cancel", _("Cancel"))
        if purchase_url:
            self.add_response("buy", _("Buy License"))
            self.set_response_appearance(
                "buy", Adw.ResponseAppearance.SUGGESTED
            )
        self.add_response("enter", _("Enter License Key"))

        self.connect("response", self._on_response)

    def _on_response(self, dialog, response_id):
        handlers = {
            "buy": self._handle_buy,
            "enter": self._show_license_entry_dialog,
        }

        handler = handlers.get(response_id)
        if handler:
            handler()

    def _handle_buy(self):
        if self.purchase_url:
            webbrowser.open(self.purchase_url)
            self._show_license_entry_dialog()

    def _show_license_entry_dialog(self):
        parent = self.get_transient_for()
        self.close()

        entry_dialog = LicenseEntryDialog(
            product_ids=self.product_ids,
            addon_name=self.addon_name,
            on_success=self.on_license_added,
        )
        if parent:
            entry_dialog.set_transient_for(parent)
        entry_dialog.present()
