"""Shared install/update action button for addon rows and dialogs."""

from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Gtk

from ... import __version__
from ...addon_mgr.addon import AddonMetadata
from ...addon_mgr.addon_manager import UpdateStatus
from ...context import get_context


def is_license_valid(addon: AddonMetadata) -> bool:
    """Check if the user already has a valid license for this addon."""
    license_config = addon.license
    if not license_config or not license_config.required:
        return True

    validator = get_context().license_validator
    if not validator:
        return False

    result = validator.validate(addon.name, license_config.to_dict())
    return result.status.value == "valid"


def get_product_ids(addon: AddonMetadata) -> list[str]:
    """Extract product IDs from addon license config."""
    if not addon.license:
        return []
    return addon.license.get_all_product_ids()


def build_action_button(
    addon: AddonMetadata,
    on_install: Callable[[AddonMetadata, bool], None],
    on_unlock: Callable[[AddonMetadata], None],
) -> Gtk.Button:
    """
    Builds the Install/Update/Unlock/Installed button for an addon.

    on_install(addon, is_update) runs when the user installs or
    updates the addon; on_unlock(addon) runs for premium addons
    without a valid license.
    """
    context = get_context()
    btn = Gtk.Button(valign=Gtk.Align.CENTER)
    status, local_ver = context.addon_mgr.check_update_status(addon)

    license_config = addon.license
    is_premium = license_config and license_config.required

    if status == UpdateStatus.NOT_INSTALLED:
        if is_premium and not is_license_valid(addon):
            btn.set_label(_("Unlock"))
            btn.add_css_class("suggested-action")
            btn.connect("clicked", lambda _b: on_unlock(addon))
        else:
            btn.set_label(_("Install"))
            btn.add_css_class("suggested-action")
            btn.connect("clicked", lambda _b: on_install(addon, False))
    elif status == UpdateStatus.UPDATE_AVAILABLE:
        btn.set_label(_("Update"))
        btn.add_css_class("suggested-action")
        btn.connect("clicked", lambda _b: on_install(addon, True))
    elif status == UpdateStatus.UP_TO_DATE:
        btn.set_label(_("Installed"))
        btn.set_sensitive(False)
        btn.set_tooltip_text(
            _("Version {v} already installed").format(v=local_ver)
        )
    elif status == UpdateStatus.INCOMPATIBLE:
        btn.set_label(_("Incompatible"))
        btn.set_sensitive(False)
        deps_str = ", ".join(addon.depends)
        btn.set_tooltip_text(
            _(
                "Requires {deps}, but current rayforge version is {current}"
            ).format(deps=deps_str, current=__version__)
        )

    if not addon.url:
        btn.set_label(_("Unavailable"))
        btn.set_sensitive(False)

    return btn
