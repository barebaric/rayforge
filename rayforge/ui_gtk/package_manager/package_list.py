import logging
import threading
from typing import Callable, cast, Optional, Tuple
from gettext import gettext as _
from gi.repository import Adw, Gtk, GLib
from blinker import Signal
from ... import __version__
from ...context import get_context
from ...package_mgr.package import Package, PackageMetadata
from ...package_mgr.package_manager import AddonState
from ...shared.util.versioning import UnknownVersion
from ..icons import get_icon
from ..shared.preferences_group import PreferencesGroupWithButton
from .dialog import PackageRegistryDialog

logger = logging.getLogger(__name__)


class PackageRow(Gtk.Box):
    """A widget representing a single Package in a ListBox."""

    def __init__(
        self,
        pkg: Package,
        state: str,
        on_delete,
        on_toggle: Optional[Callable] = None,
        error_message: Optional[str] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.pkg = pkg
        self.state = state
        self.on_toggle = on_toggle

        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        is_asset_only = not pkg.metadata.provides.backend

        if state == AddonState.LOAD_ERROR.value:
            icon = get_icon("error-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            if error_message:
                icon.set_tooltip_text(error_message)
            else:
                icon.set_tooltip_text(_("Failed to load this package"))
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
                    "This package is incompatible with the current "
                    "version of Rayforge"
                )
            )
            self.append(icon)
        else:
            icon = get_icon("addon-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            self.append(icon)

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, hexpand=True
        )
        self.append(labels_box)

        title = Gtk.Label(
            label=pkg.metadata.display_name or pkg.metadata.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        if state in (
            AddonState.INCOMPATIBLE.value,
            AddonState.LOAD_ERROR.value,
        ):
            title.add_css_class("dim-label")
        labels_box.append(title)

        subtitle = Gtk.Label(
            label=self._get_subtitle(),
            halign=Gtk.Align.START,
            xalign=0,
        )
        subtitle.add_css_class("dim-label")
        labels_box.append(subtitle)

        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        if not is_asset_only and on_toggle:
            self.enable_switch = Gtk.Switch(valign=Gtk.Align.CENTER)
            self.enable_switch.set_active(state == AddonState.ENABLED.value)
            self.enable_switch.set_sensitive(
                state
                not in (
                    AddonState.INCOMPATIBLE.value,
                    AddonState.LOAD_ERROR.value,
                    AddonState.PENDING_UNLOAD.value,
                )
            )
            self.enable_switch.set_tooltip_text(
                _("Enable or disable this addon")
            )
            self.enable_switch.connect("state-set", self._on_toggle_clicked)
            suffix_box.append(self.enable_switch)

        delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        delete_button.add_css_class("flat")
        delete_button.set_tooltip_text(_("Uninstall Package"))
        delete_button.connect("clicked", lambda w: on_delete(pkg))
        suffix_box.append(delete_button)

    def _get_subtitle(self) -> str:
        parts = []
        version = self.pkg.metadata.version
        if version is UnknownVersion:
            parts.append(__version__)
        elif version:
            parts.append(str(version))
        if self.pkg.metadata.author.name:
            parts.append(self.pkg.metadata.author.name)
        return " | ".join(parts)

    def _on_toggle_clicked(self, switch: Gtk.Switch, state: bool) -> bool:
        if self.on_toggle:
            self.on_toggle(self.pkg.metadata.name, state)
        return False


class PackageListWidget(PreferencesGroupWithButton):
    """Displays a list of packages and allows adding/deleting them."""

    def __init__(self, **kwargs):
        super().__init__(button_label=_("Install New Package..."), **kwargs)

        placeholder = Gtk.Label(
            label=_("No packages installed."),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_show_separators(True)

        self.populate_packages()

        self.install_started = Signal()
        self.install_finished = Signal()

    def populate_packages(self):
        """Refreshes the list of packages."""
        context = get_context()
        pm = context.package_mgr

        packages: list[Tuple[Package, str, Optional[str]]] = []

        for name, pkg in pm.loaded_packages.items():
            if name in pm._pending_unloads:
                state = AddonState.PENDING_UNLOAD.value
            else:
                state = AddonState.ENABLED.value
            packages.append((pkg, state, None))

        for name, pkg in pm.disabled_packages.items():
            packages.append((pkg, AddonState.DISABLED.value, None))

        for name, pkg in pm.incompatible_packages.items():
            packages.append((pkg, AddonState.INCOMPATIBLE.value, None))

        for name, error in pm._load_errors.items():
            pkg = (
                pm.loaded_packages.get(name)
                or pm.disabled_packages.get(name)
                or pm.incompatible_packages.get(name)
            )
            if pkg:
                packages.append((pkg, AddonState.LOAD_ERROR.value, error))

        packages.sort(
            key=lambda p: (
                p[0].metadata.display_name or p[0].metadata.name
            ).lower(),
        )
        self.set_items(packages)

    def create_row_widget(self, item: tuple) -> Gtk.Widget:
        pkg, state, error_message = item
        return PackageRow(
            pkg,
            state,
            self._on_delete_package,
            self._on_toggle_addon,
            error_message,
        )

    def _on_add_clicked(self, button):
        """Opens the registry dialog."""
        root = cast(Gtk.Window, self.get_root())
        dialog = PackageRegistryDialog(root, self._install_package)
        dialog.present()

    def _install_package(self, install_info):
        """
        Installs and hot-loads the package via backend.
        `install_info` can be PackageMetadata or a git_url string.
        """
        context = get_context()
        package_id = None
        git_url = ""
        display_name = ""

        if isinstance(install_info, PackageMetadata):
            git_url = install_info.url
            package_id = install_info.name
            display_name = install_info.display_name or install_info.name
        else:
            git_url = str(install_info)
            package_id = None
            display_name = context.package_mgr._extract_repo_name(git_url)

        self.list_box.set_sensitive(False)
        self.add_button.set_sensitive(False)
        self.install_started.send(
            self, message=_("Installing {name}...").format(name=display_name)
        )

        def _worker():
            return context.package_mgr.install_package(git_url, package_id)

        def _done(result_path):
            self.list_box.set_sensitive(True)
            self.add_button.set_sensitive(True)
            self.install_finished.send(self)

            if result_path:
                self.populate_packages()
            else:
                self._show_error(_("Failed to install package."))

        thread = threading.Thread(
            target=lambda: GLib.idle_add(_done, _worker()), daemon=True
        )
        thread.start()

    def _on_toggle_addon(self, addon_name: str, enable: bool):
        """Toggle addon enabled/disabled state."""
        context = get_context()
        pm = context.package_mgr

        if enable:
            missing = pm.get_missing_dependencies(addon_name)
            if missing:
                missing_names = [name for name, _ in missing]
                missing_str = ", ".join(missing_names)
                root = cast(Gtk.Window, self.get_root())

                def on_response(dialog, response):
                    if response == "enable":
                        success, enabled = pm.enable_addon_with_deps(
                            addon_name
                        )
                        if not success:
                            self._show_error(
                                _(
                                    "Failed to enable addon and its "
                                    "dependencies."
                                )
                            )
                        self.populate_packages()
                    else:
                        self.populate_packages()
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

            success = pm.enable_addon(addon_name)
            if not success:
                self._show_error(
                    _("Failed to enable addon. Check the logs for details.")
                )
            self.populate_packages()
        else:
            can_disable, reason = pm.can_disable(addon_name)
            if not can_disable:
                self._show_warning(
                    _("Cannot Disable Addon"),
                    _("This addon cannot be disabled.\n\n{reason}").format(
                        reason=reason
                    ),
                )
                self.populate_packages()
                return

            success = pm.disable_addon(addon_name)
            if not success:
                if pm.has_pending_unloads():
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

            self.populate_packages()

    def _on_delete_package(self, pkg: Package):
        """Confirm and delete the package."""
        display_name = pkg.metadata.display_name or pkg.metadata.name
        root = cast(Gtk.Window, self.get_root())
        dialog = Adw.MessageDialog(
            transient_for=root,
            heading=_("Uninstall {name}?").format(name=display_name),
            body=_(
                "The package files will be removed. "
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
                self._delete_package(pkg)
            dlg.close()

        dialog.connect("response", _response_cb)
        dialog.present()

    def _delete_package(self, pkg: Package):
        """Triggers the backend to uninstall the package."""
        context = get_context()
        success = context.package_mgr.uninstall_package(pkg.metadata.name)

        if success:
            self.populate_packages()
        else:
            logger.error(
                f"UI failed to trigger uninstall for {pkg.metadata.name}"
            )
            self._show_error(_("Error deleting package."))

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
