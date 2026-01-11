import logging
import threading
from typing import cast
from gi.repository import Adw, Gtk, GLib
from blinker import Signal
from ...context import get_context
from ...package_mgr.package import Package, PackageMetadata
from ..icons import get_icon
from ..shared.preferences_group import PreferencesGroupWithButton
from .dialog import PackageRegistryDialog

logger = logging.getLogger(__name__)


class PackageRow(Gtk.Box):
    """A widget representing a single Package in a ListBox."""

    def __init__(self, pkg: Package, on_delete):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.pkg = pkg

        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

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

        delete_button = Gtk.Button(icon_name="user-trash-symbolic")
        delete_button.add_css_class("flat")
        delete_button.add_css_class("destructive-action")
        delete_button.set_tooltip_text(_("Uninstall Package"))
        delete_button.connect("clicked", lambda w: on_delete(pkg))
        suffix_box.append(delete_button)

    def _get_subtitle(self) -> str:
        parts = []
        if self.pkg.metadata.version:
            parts.append(self.pkg.metadata.version)
        if self.pkg.metadata.author.name:
            parts.append(self.pkg.metadata.author.name)
        return " | ".join(parts)


class PackageListWidget(PreferencesGroupWithButton):
    """Displays a list of packages and allows adding/deleting them."""

    # Signals to communicate with the parent page
    install_started = Signal()
    install_finished = Signal()

    def __init__(self, **kwargs):
        super().__init__(button_label=_("Install New Package..."), **kwargs)

        # Set up the list box
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

    def populate_packages(self):
        """Refreshes the list of packages."""
        context = get_context()
        packages = sorted(
            context.package_mgr.loaded_packages.values(),
            key=lambda p: (p.metadata.display_name or p.metadata.name).lower(),
        )
        self.set_items(packages)

    def create_row_widget(self, item: Package) -> Gtk.Widget:
        return PackageRow(item, self._on_delete_package)

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

        # Update UI to show progress
        self.list_box.set_sensitive(False)
        self.add_button.set_sensitive(False)
        self.install_started.send(
            self, message=_("Installing {name}...").format(name=display_name)
        )

        def _worker():
            return context.package_mgr.install_package(git_url, package_id)

        def _done(result_path):
            # Restore UI to its normal state
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
