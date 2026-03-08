import logging
import webbrowser
from gettext import gettext as _
from typing import cast, Optional
from gi.repository import Adw, Gtk, GLib

from ..shared.preferences_page import TrackedPreferencesPage
from ...context import get_context


logger = logging.getLogger(__name__)


class LicenseSettingsPage(TrackedPreferencesPage):
    """Settings page for managing licenses."""

    key = "licenses"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(_("Licenses"))
        self.set_icon_name("license-symbolic")

        self._groups: list[Adw.PreferencesGroup] = []
        self._build_ui()

        validator = get_context().license_validator
        validator.changed.connect(self._on_license_changed)

    def _on_license_changed(self, sender):
        self._refresh_ui()

    def _build_ui(self):
        self._build_patreon_section()
        self._build_licenses_section()
        self._build_license_required_section()

    def _add_group(self, group: Adw.PreferencesGroup) -> None:
        self._groups.append(group)
        self.add(group)

    def _get_addons_for_product_id(self, product_id: str) -> list[str]:
        """Get list of addon names that use this product ID."""
        addon_mgr = get_context().addon_mgr
        result = []

        for addon_name, addon in addon_mgr.license_required_addons.items():
            if addon.metadata.license:
                product_ids = addon.metadata.license.get_all_product_ids()
                if product_id in product_ids:
                    display = addon.metadata.display_name or addon_name
                    result.append(display)

        return result

    def _build_patreon_section(self):
        validator = get_context().license_validator
        patreon = validator.get_provider("patreon")

        patreon_group = Adw.PreferencesGroup(
            title=_("Patreon"),
            description=_(
                "Link your Patreon account for early access to new addons."
            ),
        )
        self._add_group(patreon_group)

        if patreon and patreon.is_configured():
            row = Adw.ActionRow(
                title=_("Patreon Account Linked"),
                subtitle=_("Early access addons are unlocked"),
            )
            unlink_btn = Gtk.Button(label=_("Unlink"))
            unlink_btn.add_css_class("destructive-action")
            unlink_btn.set_valign(Gtk.Align.CENTER)
            unlink_btn.connect("clicked", self._on_unlink_patreon)
            row.add_suffix(unlink_btn)
            patreon_group.add(row)
        else:
            row = Adw.ActionRow(
                title=_("Link Patreon Account"),
                subtitle=_("Get early access to premium addons"),
            )
            link_btn = Gtk.Button(label=_("Link"))
            link_btn.add_css_class("suggested-action")
            link_btn.set_valign(Gtk.Align.CENTER)
            link_btn.connect("clicked", self._on_link_patreon)
            row.add_suffix(link_btn)
            patreon_group.add(row)

    def _build_licenses_section(self):
        validator = get_context().license_validator
        licenses = validator.get_gumroad_licenses()

        licenses_group = Adw.PreferencesGroup(
            title=_("Addon Licenses"),
            description=_("Manage your purchased license keys."),
        )
        self._add_group(licenses_group)

        if not licenses:
            empty_row = Adw.ActionRow(
                title=_("No licenses installed"),
                subtitle=_(
                    "Purchase a premium addon and enter the license "
                    "key during installation."
                ),
            )
            empty_row.set_sensitive(False)
            licenses_group.add(empty_row)
            return

        for product_id, license_key in licenses.items():
            masked = (
                f"****{license_key[-4:]}" if len(license_key) > 4 else "****"
            )

            addon_names = self._get_addons_for_product_id(product_id)
            if addon_names:
                max_show = 3
                if len(addon_names) > max_show:
                    subtitle = _("Unlocks: {addons} (+{count} more)").format(
                        addons=", ".join(addon_names[:max_show]),
                        count=len(addon_names) - max_show,
                    )
                else:
                    subtitle = _("Unlocks: {addons}").format(
                        addons=", ".join(addon_names)
                    )
            else:
                subtitle = masked

            row = Adw.ActionRow(title=product_id, subtitle=subtitle)
            remove_btn = Gtk.Button(label=_("Remove"))
            remove_btn.add_css_class("destructive-action")
            remove_btn.set_valign(Gtk.Align.CENTER)
            remove_btn.connect("clicked", self._on_remove_license, product_id)
            row.add_suffix(remove_btn)
            licenses_group.add(row)

    def _build_license_required_section(self):
        addon_mgr = get_context().addon_mgr
        license_required = addon_mgr.get_all_license_required_addons()

        if not license_required:
            return

        section = Adw.PreferencesGroup(
            title=_("Addons Requiring License"),
            description=_("These addons need a valid license to be activated"),
        )
        self._add_group(section)

        for addon_name, addon in license_required.items():
            display_name = addon.metadata.display_name or addon_name
            row = Adw.ActionRow(
                title=display_name,
                subtitle=addon.license_message or _("License required"),
            )

            if addon.purchase_url:
                buy_btn = Gtk.Button(label=_("Buy"))
                buy_btn.set_valign(Gtk.Align.CENTER)
                buy_btn.connect(
                    "clicked", self._on_buy_license, addon.purchase_url
                )
                row.add_suffix(buy_btn)

            section.add(row)

    def _on_link_patreon(self, btn):
        validator = get_context().license_validator

        def on_oauth_complete(success, error):
            if success:
                GLib.idle_add(self._refresh_ui)
            elif error:
                logger.error(f"Patreon OAuth failed: {error}")

        try:
            result = validator.start_patreon_oauth(on_oauth_complete)
            if result is None:
                logger.warning(
                    "Patreon integration not configured. "
                    "Set RAYFORGE_PATREON_CLIENT_ID environment variable."
                )
                return
            port, thread = result
            oauth_url = validator.get_patreon_oauth_url()
            if oauth_url:
                logger.info("Opening Patreon OAuth URL")
                webbrowser.open(oauth_url)
        except Exception as e:
            logger.error(f"Failed to start Patreon OAuth: {e}", exc_info=True)

    def _on_unlink_patreon(self, btn):
        validator = get_context().license_validator
        validator.unlink_patreon()
        self._refresh_ui()

    def _on_remove_license(self, btn, product_id):
        dialog = Adw.MessageDialog(
            transient_for=cast(
                Optional[Gtk.Window], self.get_ancestor(Gtk.Window)
            ),
            modal=True,
            heading=_("Remove License?"),
            body=_(
                "This license key will be removed. You may need to "
                "re-enter it to use licensed addons."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("remove", _("Remove"))
        dialog.set_response_appearance(
            "remove", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.set_default_response("cancel")

        dialog.connect(
            "response", self._on_remove_license_response, product_id
        )
        dialog.present()

    def _on_remove_license_response(
        self, dialog, response_id: str, product_id
    ):
        if response_id == "remove":
            validator = get_context().license_validator
            validator.remove_gumroad_license(product_id)
            self._refresh_ui()
        dialog.close()

    def _on_buy_license(self, btn, purchase_url):
        webbrowser.open(purchase_url)

    def _refresh_ui(self):
        for group in self._groups:
            self.remove(group)
        self._groups.clear()
        self._build_ui()
