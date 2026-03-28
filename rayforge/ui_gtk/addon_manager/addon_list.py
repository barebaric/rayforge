import logging
import threading
from typing import Callable, cast, Optional, Tuple
from gettext import gettext as _
from gi.repository import Adw, Gtk, GLib
from blinker import Signal
from ... import __version__
from ...context import get_context
from ...addon_mgr.addon import Addon, AddonMetadata
from ...addon_mgr.addon_manager import AddonState
from ...shared.util.versioning import UnknownVersion
from ..icons import get_icon
from ..shared.preferences_group import PreferencesGroupWithButton
from .addon_dialog import AddonRegistryDialog
from .license_dialog import LicenseRequiredDialog


logger = logging.getLogger(__name__)


class AddonRow(Gtk.Box):
    """A widget representing a single Addon in a ListBox."""

    def __init__(
        self,
        addon: Addon,
        state: str,
        on_delete,
        on_toggle: Optional[Callable] = None,
        on_unlock: Optional[Callable] = None,
        error_message: Optional[str] = None,
        is_builtin: bool = False,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.addon = addon
        self.state = state
        self.on_toggle = on_toggle
        self.on_unlock = on_unlock
        self.is_builtin = is_builtin

        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        is_asset_only = not addon.metadata.provides.backend

        if state == AddonState.LICENSE_REQUIRED.value:
            icon = get_icon("lock-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            icon.set_tooltip_text(_("License required"))
            self.append(icon)
        elif state == AddonState.LOAD_ERROR.value:
            icon = get_icon("error-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            if error_message:
                icon.set_tooltip_text(error_message)
            else:
                icon.set_tooltip_text(_("Failed to load this addon"))
            self.append(icon)
        elif state == AddonState.PENDING_UNLOAD.value:
            icon = get_icon("hourglass-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            icon.set_tooltip_text(
                _("This addon will be unloaded when active jobs finish")
            )
            self.append(icon)
        elif state == AddonState.INCOMPATIBLE.value:
            icon = get_icon("warning-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            icon.add_css_class("warning")
            icon.set_tooltip_text(
                _(
                    "This addon is incompatible with the current "
                    "version of Rayforge"
                )
            )
            self.append(icon)
        elif addon.metadata.license and addon.metadata.license.required:
            icon = get_icon("crown-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            icon.set_tooltip_text(_("Premium addon"))
            self.append(icon)
        else:
            if self.is_builtin:
                icon = get_icon("addon-builtin-symbolic")
                icon.set_tooltip_text(_("Built-in addon"))
            else:
                icon = get_icon("addon-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            self.append(icon)

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, hexpand=True
        )
        self.append(labels_box)

        title = Gtk.Label(
            label=addon.metadata.display_name or addon.metadata.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        if state in (
            AddonState.INCOMPATIBLE.value,
            AddonState.LOAD_ERROR.value,
            AddonState.LICENSE_REQUIRED.value,
        ):
            title.add_css_class("dim-label")
        labels_box.append(title)

        subtitle_text = self._get_subtitle()
        if state == AddonState.LICENSE_REQUIRED.value:
            subtitle_text = _("License required")
        subtitle = Gtk.Label(
            label=subtitle_text,
            halign=Gtk.Align.START,
            xalign=0,
        )
        subtitle.add_css_class("dim-label")
        labels_box.append(subtitle)

        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        if state == AddonState.LICENSE_REQUIRED.value:
            unlock_btn = Gtk.Button(label=_("Unlock"))
            unlock_btn.add_css_class("suggested-action")
            unlock_btn.set_valign(Gtk.Align.CENTER)
            unlock_btn.connect("clicked", self._on_unlock_clicked)
            suffix_box.append(unlock_btn)

        if not self.is_builtin:
            delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
            delete_button.add_css_class("flat")
            delete_button.set_tooltip_text(_("Uninstall Addon"))
            delete_button.connect("clicked", lambda w: on_delete(addon))
            suffix_box.append(delete_button)

        if (
            state != AddonState.LICENSE_REQUIRED.value
            and not is_asset_only
            and on_toggle
        ):
            self.enable_switch = Gtk.Switch(valign=Gtk.Align.CENTER)
            self.enable_switch.set_active(state == AddonState.ENABLED.value)
            self.enable_switch.set_sensitive(
                state
                not in (
                    AddonState.INCOMPATIBLE.value,
                    AddonState.LOAD_ERROR.value,
                    AddonState.PENDING_UNLOAD.value,
                    AddonState.LICENSE_REQUIRED.value,
                )
            )
            self.enable_switch.set_tooltip_text(
                _("Enable or disable this addon")
            )
            self.enable_switch.connect("state-set", self._on_toggle_clicked)
            suffix_box.append(self.enable_switch)

    def _get_subtitle(self) -> str:
        parts = []
        version = self.addon.metadata.version
        if version is UnknownVersion:
            if self.is_builtin:
                parts.append(__version__)
        elif version:
            parts.append(str(version))
        if self.addon.metadata.author.name:
            parts.append(self.addon.metadata.author.name)
        return " | ".join(parts)

    def _on_toggle_clicked(self, switch: Gtk.Switch, state: bool) -> bool:
        if self.on_toggle:
            self.on_toggle(self.addon.metadata.name, state)
        return False

    def _on_unlock_clicked(self, btn):
        if self.on_unlock:
            self.on_unlock(self.addon)


class AddonListWidget(PreferencesGroupWithButton):
    """Displays a list of addons and allows adding/deleting them."""

    def __init__(self, **kwargs):
        super().__init__(button_label=_("Install New Addon..."), **kwargs)

        placeholder = Gtk.Label(
            label=_("No addons installed."),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_show_separators(True)

        self.populate_addons()

        self.install_started = Signal()
        self.install_finished = Signal()

    def populate_addons(self):
        """Refreshes the list of addons."""
        context = get_context()
        am = context.addon_mgr

        addons: list[Tuple[Addon, str, Optional[str], bool]] = []

        for name, addon in am.loaded_addons.items():
            if name in am._pending_unloads:
                state = AddonState.PENDING_UNLOAD.value
            else:
                state = AddonState.ENABLED.value
            is_builtin = not addon.root_path.is_relative_to(am.install_dir)
            addons.append((addon, state, None, is_builtin))

        for name, addon in am.disabled_addons.items():
            is_builtin = not addon.root_path.is_relative_to(am.install_dir)
            addons.append((addon, AddonState.DISABLED.value, None, is_builtin))

        for name, addon in am.incompatible_addons.items():
            is_builtin = not addon.root_path.is_relative_to(am.install_dir)
            addons.append(
                (addon, AddonState.INCOMPATIBLE.value, None, is_builtin)
            )

        for name, addon in am.license_required_addons.items():
            is_builtin = not addon.root_path.is_relative_to(am.install_dir)
            addons.append(
                (addon, AddonState.LICENSE_REQUIRED.value, None, is_builtin)
            )

        for name, error in am._load_errors.items():
            addon = (
                am.loaded_addons.get(name)
                or am.disabled_addons.get(name)
                or am.incompatible_addons.get(name)
            )
            if addon:
                is_builtin = not addon.root_path.is_relative_to(am.install_dir)
                addons.append(
                    (addon, AddonState.LOAD_ERROR.value, error, is_builtin)
                )

        addons.sort(
            key=lambda a: (
                a[0].metadata.display_name or a[0].metadata.name
            ).lower(),
        )
        self.set_items(addons)

    def create_row_widget(self, item: tuple) -> Gtk.Widget:
        addon, state, error_message, is_builtin = item
        return AddonRow(
            addon,
            state,
            self._on_delete_addon,
            self._on_toggle_addon,
            self._on_unlock_addon,
            error_message,
            is_builtin,
        )

    def _on_add_clicked(self, button):
        """Opens the registry dialog."""
        root = cast(Gtk.Window, self.get_root())
        dialog = AddonRegistryDialog(root, self._install_addon)
        dialog.present()

    def _install_addon(self, install_info):
        """
        Installs and hot-loads the addon via backend.
        `install_info` can be AddonMetadata or a git_url string.
        """
        context = get_context()
        addon_id = None
        git_url = ""
        display_name = ""

        if isinstance(install_info, AddonMetadata):
            git_url = install_info.url
            addon_id = install_info.name
            display_name = install_info.display_name or install_info.name
        else:
            git_url = str(install_info)
            addon_id = None
            display_name = context.addon_mgr._extract_repo_name(git_url)

        self.list_box.set_sensitive(False)
        self.add_button.set_sensitive(False)
        self.install_started.send(
            self, message=_("Installing {name}...").format(name=display_name)
        )

        def _worker():
            return context.addon_mgr.install_addon(git_url, addon_id)

        def _done(result_path):
            self.list_box.set_sensitive(True)
            self.add_button.set_sensitive(True)
            self.install_finished.send(self)

            if result_path:
                self.populate_addons()
            else:
                self._show_error(_("Failed to install addon."))

        thread = threading.Thread(
            target=lambda: GLib.idle_add(_done, _worker()), daemon=True
        )
        thread.start()

    def _on_toggle_addon(self, addon_name: str, enable: bool):
        """Toggle addon enabled/disabled state."""
        context = get_context()
        am = context.addon_mgr

        if enable:
            missing = am.get_missing_dependencies(addon_name)
            if missing:
                missing_names = [name for name, _ in missing]
                missing_str = ", ".join(missing_names)
                root = cast(Gtk.Window, self.get_root())

                def on_response(dialog, response):
                    if response == "enable":
                        success, enabled = am.enable_addon_with_deps(
                            addon_name
                        )
                        if not success:
                            self._show_error(
                                _(
                                    "Failed to enable addon and its "
                                    "dependencies."
                                )
                            )
                        self.populate_addons()
                    else:
                        self.populate_addons()
                    dialog.close()

                dialog = Adw.MessageDialog(
                    transient_for=root,
                    heading=_("Enable Dependencies?"),
                    body=_(
                        "This addon requires: {deps}\n\nEnable them as well?"
                    ).format(deps=missing_str),
                )
                dialog.add_response("cancel", _("Cancel"))
                dialog.add_response("enable", _("Enable All"))
                dialog.connect("response", on_response)
                dialog.present()
                return

            success = am.enable_addon(addon_name)
            if not success:
                self._show_error(
                    _("Failed to enable addon. Check the logs for details.")
                )
            self.populate_addons()
        else:
            can_disable, reason = am.can_disable(addon_name)
            if not can_disable:
                self._show_warning(
                    _("Cannot Disable Addon"),
                    _("This addon cannot be disabled.\n\n{reason}").format(
                        reason=reason
                    ),
                )
                self.populate_addons()
                return

            success = am.disable_addon(addon_name)
            if not success:
                if am.has_pending_unloads():
                    self._show_info(
                        _("Addon will be disabled when active jobs complete.")
                    )
                else:
                    self._show_error(
                        _(
                            "Failed to disable addon. Check the logs "
                            "for details."
                        )
                    )

            self.populate_addons()

    def _on_unlock_addon(self, addon: Addon):
        """Handle unlock button click for license-required addon."""
        license_config = addon.metadata.license
        if not license_config:
            return

        product_ids = license_config.get_all_product_ids()
        purchase_url = license_config.purchase_url
        display_name = addon.metadata.display_name or addon.metadata.name
        addon_name = addon.metadata.name

        def on_license_added():
            context = get_context()
            context.addon_mgr.recheck_license(addon_name)
            self.populate_addons()

        root = cast(Gtk.Window, self.get_root())
        dialog = LicenseRequiredDialog(
            addon_name=display_name,
            product_ids=product_ids,
            purchase_url=purchase_url,
            on_license_added=on_license_added,
        )
        if root:
            dialog.set_transient_for(root)
        dialog.present()

    def _on_delete_addon(self, addon: Addon):
        """Confirm and delete the addon."""
        display_name = addon.metadata.display_name or addon.metadata.name
        root = cast(Gtk.Window, self.get_root())
        dialog = Adw.MessageDialog(
            transient_for=root,
            heading=_("Uninstall {name}?").format(name=display_name),
            body=_(
                "The addon files will be removed. "
                "Restart recommended to fully clear memory."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Uninstall"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )

        def _response_cb(dlg, response):
            if response == "delete":
                self._delete_addon(addon)
            dlg.close()

        dialog.connect("response", _response_cb)
        dialog.present()

    def _delete_addon(self, addon: Addon):
        """Triggers the backend to uninstall the addon."""
        context = get_context()
        success = context.addon_mgr.uninstall_addon(addon.metadata.name)

        if success:
            self.populate_addons()
        else:
            logger.error(
                f"UI failed to trigger uninstall for {addon.metadata.name}"
            )
            self._show_error(_("Error deleting addon."))

    def _show_error(self, message):
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, self.get_root()),
            heading=_("Error"),
            body=message,
        )
        dialog.add_response("ok", _("OK"))
        dialog.present()

    def _show_info(self, message):
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, self.get_root()),
            heading=_("Info"),
            body=message,
        )
        dialog.add_response("ok", _("OK"))
        dialog.present()

    def _show_warning(self, heading: str, message: str):
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, self.get_root()),
            heading=heading,
            body=message,
        )
        dialog.add_response("ok", _("OK"))
        dialog.present()
