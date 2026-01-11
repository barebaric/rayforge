import logging
import threading
from typing import List

from gi.repository import Adw, Gdk, GLib, Gtk

from ... import __version__
from ...context import get_context
from ...package_mgr.package import PackageMetadata
from ...package_mgr.package_manager import UpdateStatus
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow

logger = logging.getLogger(__name__)


class PackageRegistryDialog(PatchedDialogWindow):
    """
    A dialog that fetches and lists available packages from the
    online registry via the PackageManager.
    """

    def __init__(self, parent_window, on_install_callback):
        super().__init__()
        self.set_transient_for(parent_window)
        self.set_modal(True)
        self.set_title(_("Package Registry"))
        self.set_default_size(600, 700)

        self.on_install_callback = on_install_callback

        # Main Layout
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(box)

        # Header
        header = Adw.HeaderBar()
        box.append(header)

        # Content area (Stack for Loading vs List)
        self.stack = Gtk.Stack()
        self.stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        box.append(self.stack)

        # 1. Loading Page
        loading_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            valign=Gtk.Align.CENTER,
            halign=Gtk.Align.CENTER,
        )
        spinner = Gtk.Spinner()
        spinner.set_size_request(32, 32)
        spinner.start()
        loading_box.append(spinner)
        loading_box.append(Gtk.Label(label=_("Fetching registry...")))
        self.stack.add_named(loading_box, "loading")

        # 2. List Page
        list_page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # Scrolled Window for the list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        clamp = Adw.Clamp(maximum_size=800)
        clamp.set_margin_top(24)
        clamp.set_margin_bottom(24)
        clamp.set_margin_start(12)
        clamp.set_margin_end(12)

        self.list_box = Gtk.ListBox()
        self.list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.list_box.get_style_context().add_class("boxed-list")

        clamp.set_child(self.list_box)
        scrolled.set_child(clamp)
        list_page_box.append(scrolled)

        # Manual Install Button Footer
        footer_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_bottom=12,
            margin_top=12,
        )
        manual_btn = Gtk.Button(label=_("Install from URL..."))
        manual_btn.get_style_context().add_class("flat")
        manual_btn.set_halign(Gtk.Align.CENTER)
        manual_btn.connect("clicked", self._on_manual_install_clicked)
        footer_box.append(manual_btn)
        list_page_box.append(footer_box)

        self.stack.add_named(list_page_box, "list")

        # 3. Error Page
        self.error_box = Adw.StatusPage()
        self.error_box.set_icon_name("network-error-symbolic")
        self.error_box.set_title(_("Connection Failed"))
        self.error_box.set_description(_("Could not reach the registry."))
        self.stack.add_named(self.error_box, "error")

        # Handle Escape key press to close the dialog
        controller = Gtk.EventControllerKey()
        controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(controller)

        # Start fetching
        self._fetch_registry()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Close the dialog when the Escape key is pressed."""
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True
        return False

    def _fetch_registry(self):
        """Fetches the registry using the PackageManager in a thread."""
        context = get_context()

        def _worker():
            return context.package_mgr.fetch_registry()

        def _done(data):
            self._populate_list(data)
            self.stack.set_visible_child_name("list")

        thread = threading.Thread(
            target=lambda: GLib.idle_add(_done, _worker()), daemon=True
        )
        thread.start()

    def _populate_list(self, data: List[PackageMetadata]):
        """Populates the list box with registry items."""
        while child := self.list_box.get_row_at_index(0):
            self.list_box.remove(child)

        if not data:
            empty_label = Gtk.Label(
                label=_("No packages found in registry."), margin_top=24
            )
            self.list_box.append(empty_label)
            return

        context = get_context()

        for package in data:
            row = Adw.ActionRow(
                title=package.display_name or package.name or "?",
                subtitle=package.description,
            )
            row.add_prefix(get_icon("addon-symbolic"))

            author_name = package.author.name
            if author_name:
                lbl = Gtk.Label(label=f"by {author_name}")
                lbl.get_style_context().add_class("dim-label")
                row.add_suffix(lbl)

            # --- Action Button Logic ---
            btn = Gtk.Button(valign=Gtk.Align.CENTER)
            status, local_ver = context.package_mgr.check_update_status(
                package
            )

            if status == UpdateStatus.NOT_INSTALLED:
                btn.set_label(_("Install"))
                btn.get_style_context().add_class("suggested-action")
                btn.connect("clicked", self._on_install_clicked, package)
            elif status == UpdateStatus.UPDATE_AVAILABLE:
                btn.set_label(_("Update"))
                btn.get_style_context().add_class("suggested-action")
                btn.connect("clicked", self._on_install_clicked, package)
            elif status == UpdateStatus.UP_TO_DATE:
                btn.set_label(_("Installed"))
                btn.set_sensitive(False)
                btn.set_tooltip_text(
                    _("Version {v} already installed").format(v=local_ver)
                )
            elif status == UpdateStatus.INCOMPATIBLE:
                btn.set_label(_("Incompatible"))
                btn.set_sensitive(False)
                deps_str = ", ".join(package.depends)
                btn.set_tooltip_text(
                    _(
                        "Requires {deps}, but current rayforge "
                        "version is {current}"
                    ).format(deps=deps_str, current=__version__)
                )

            if not package.url:  # Handle invalid registry entries
                btn.set_label(_("Unavailable"))
                btn.set_sensitive(False)

            row.add_suffix(btn)
            self.list_box.append(row)

    def _on_install_clicked(self, btn, package_meta: PackageMetadata):
        if package_meta:
            self.close()
            self.on_install_callback(package_meta)

    def _on_manual_install_clicked(self, btn):
        """Allows manual URL entry if not in registry."""
        dialog = Adw.MessageDialog(
            transient_for=self,
            heading=_("Manual Install"),
            body=_("Enter the Git URL."),
        )
        entry = Adw.EntryRow(title="URL")
        dialog.set_extra_child(entry)
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("install", _("Install"))

        def _cb(dlg, resp):
            if resp == "install":
                url = entry.get_text().strip()
                if url:
                    self.close()
                    self.on_install_callback(url)
            dlg.close()

        dialog.connect("response", _cb)
        dialog.present()
