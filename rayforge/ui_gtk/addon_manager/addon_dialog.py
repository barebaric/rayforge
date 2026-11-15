import logging
import threading
from gettext import gettext as _

from gi.repository import Adw, GLib, Gtk

from ...addon_mgr.addon import AddonMetadata, is_official_addon_url
from ...context import get_context
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from .action_button import build_action_button, get_product_ids
from .addon_detail_dialog import AddonDetailDialog
from .community_dialog import CommunityAddonDialog
from .license_dialog import LicenseRequiredDialog

logger = logging.getLogger(__name__)


class AddonRegistryDialog(PatchedDialogWindow):
    """
    A dialog that fetches and lists available addons from the
    online registry via the AddonManager.
    """

    def __init__(self, parent_window, on_install_callback):
        super().__init__()
        self.set_transient_for(parent_window)
        self.set_modal(True)
        self.set_title(_("Addon Registry"))
        self.set_default_size(600, 700)

        self.on_install_callback = on_install_callback
        self._row_addons: dict[Adw.ActionRow, AddonMetadata] = {}

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
        self.list_box.connect("row-activated", self._on_row_activated)

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
        self.error_box.set_icon_name("error-symbolic")
        self.error_box.set_title(_("Connection Failed"))
        self.error_box.set_description(_("Could not reach the registry."))
        self.stack.add_named(self.error_box, "error")

        # Start fetching
        self._fetch_registry()

    def _fetch_registry(self):
        """Fetches the registry using the AddonManager in a thread."""
        context = get_context()

        def _worker():
            logger.info("Fetching addon registry in background thread")
            result = context.addon_mgr.fetch_registry()
            logger.info(f"Registry fetch complete: {len(result)} items")
            return result

        def _done(data):
            if data is not None:
                self._populate_list(data)
                self.stack.set_visible_child_name("list")
            else:
                self.stack.set_visible_child_name("error")

        def _thread_target():
            result = _worker()
            GLib.idle_add(_done, result)

        thread = threading.Thread(target=_thread_target, daemon=True)
        thread.start()

    def _populate_list(self, data: list[AddonMetadata]):
        """Populates the list box with registry items."""
        self._row_addons.clear()
        while child := self.list_box.get_row_at_index(0):
            self.list_box.remove(child)

        if not data:
            empty_label = Gtk.Label(
                label=_("No addons found in registry."), margin_top=24
            )
            self.list_box.append(empty_label)
            return

        for addon in data:
            row = Adw.ActionRow(
                title=addon.display_name or addon.name or "?",
                subtitle=addon.description,
            )
            icon = get_icon("addon-symbolic")
            if not addon.is_official:
                icon = get_icon("addon-community-symbolic")
                icon.set_tooltip_text(_("Community addon"))
            row.add_prefix(icon)

            author_name = addon.author.name
            if author_name:
                lbl = Gtk.Label(label=f"by {author_name}")
                lbl.get_style_context().add_class("dim-label")
                row.add_suffix(lbl)

            btn = build_action_button(
                addon,
                on_install=self._on_install_clicked,
                on_unlock=self._on_unlock_clicked,
            )
            row.add_suffix(btn)
            self._row_addons[row] = addon
            self.list_box.append(row)

    def _start_install(self, addon_meta: AddonMetadata):
        self.close()
        self.on_install_callback(addon_meta)

    def _confirm_install_community(self, addon_meta: AddonMetadata):
        """Ask for confirmation before installing a community addon."""
        dialog = CommunityAddonDialog(
            addon_name=addon_meta.display_name or addon_meta.name,
            repository_url=addon_meta.url or None,
            author=addon_meta.author.name or None,
            on_install=lambda: self._start_install(addon_meta),
        )
        dialog.set_transient_for(self)
        dialog.present()

    def _on_row_activated(self, _list_box, row):
        """Opens the detail view when the user clicks an addon row."""
        addon = self._row_addons.get(row)
        if addon is None:
            return
        dialog = AddonDetailDialog(
            self,
            addon,
            on_install=self._on_install_clicked,
            on_unlock=self._on_unlock_clicked,
        )
        dialog.present()

    def _on_install_clicked(
        self, addon_meta: AddonMetadata, is_update: bool = False
    ):
        if not addon_meta:
            return
        if not is_update and not addon_meta.is_official:
            self._confirm_install_community(addon_meta)
            return
        self._start_install(addon_meta)

    def _on_unlock_clicked(self, addon_meta: AddonMetadata):
        """Handle click on Unlock button for premium addon."""
        product_ids = get_product_ids(addon_meta)
        purchase_url = None
        if addon_meta.license:
            purchase_url = addon_meta.license.purchase_url

        display_name = addon_meta.display_name or addon_meta.name

        def on_license_added():
            if not addon_meta.is_official:
                self._confirm_install_community(addon_meta)
            else:
                self._start_install(addon_meta)

        dialog = LicenseRequiredDialog(
            addon_name=display_name,
            product_ids=product_ids,
            purchase_url=purchase_url,
            on_license_added=on_license_added,
        )
        dialog.set_transient_for(self)
        dialog.present()

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
                    dlg.close()
                    self._install_from_url(url)
                    return
            dlg.close()

        dialog.connect("response", _cb)
        dialog.present()

    def _install_from_url(self, url: str):
        """Installs from a manually entered URL, confirming if needed."""
        if is_official_addon_url(url):
            self.close()
            self.on_install_callback(url)
            return

        context = get_context()
        addon_name = context.addon_mgr._extract_repo_name(url)

        def on_install():
            self.close()
            self.on_install_callback(url)

        dialog = CommunityAddonDialog(
            addon_name=addon_name,
            repository_url=url,
            on_install=on_install,
        )
        dialog.set_transient_for(self)
        dialog.present()
