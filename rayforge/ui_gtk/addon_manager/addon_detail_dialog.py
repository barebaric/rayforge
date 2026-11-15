"""Detail view shown when the user clicks an addon row."""

import logging
import threading
import webbrowser
from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw, GLib, Gtk, Pango

from ...addon_mgr.addon import AddonMetadata
from ...context import get_context
from ...shared.util.versioning import UnknownVersion
from ..shared.async_image import AsyncImage
from ..shared.markdown import MarkdownView
from ..shared.patched_dialog_window import PatchedDialogWindow
from .action_button import build_action_button

logger = logging.getLogger(__name__)

CAROUSEL_HEIGHT = 360


class AddonDetailDialog(PatchedDialogWindow):
    """
    Shows an addon's description and screenshots.

    Opens from both the registry dialog and the installed addons
    list. Screenshots and the long description are fetched from the
    registry in the background, so addons without registry data (for
    example manually installed ones) still show their local metadata.
    """

    def __init__(
        self,
        parent_window,
        addon: AddonMetadata,
        on_install: Callable[[AddonMetadata, bool], None] | None = None,
        on_unlock: Callable[[AddonMetadata], None] | None = None,
    ):
        super().__init__()
        self.set_transient_for(parent_window)
        self.set_modal(True)
        self.set_title(addon.display_name or addon.name)
        self.set_default_size(640, 720)

        self._addon = addon
        self._on_install = on_install
        self._on_unlock = on_unlock
        self._destroyed = False
        self.connect("destroy", self._on_destroy)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(box)

        header = Adw.HeaderBar()
        box.append(header)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        box.append(scrolled)

        clamp = Adw.Clamp(maximum_size=800)
        clamp.set_margin_top(24)
        clamp.set_margin_bottom(24)
        clamp.set_margin_start(12)
        clamp.set_margin_end(12)
        scrolled.set_child(clamp)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        clamp.set_child(content)

        self._carousel_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6
        )
        self._carousel_box.set_visible(False)
        content.append(self._carousel_box)

        title = Gtk.Label(
            label=addon.display_name or addon.name,
            halign=Gtk.Align.START,
            xalign=0,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
        )
        title.add_css_class("title-1")
        content.append(title)

        self._meta_label = Gtk.Label(
            label=self._meta_text(addon),
            halign=Gtk.Align.START,
            xalign=0,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
        )
        self._meta_label.add_css_class("dim-label")
        content.append(self._meta_label)

        self._description_label = Gtk.Label(
            label=addon.description,
            halign=Gtk.Align.START,
            xalign=0,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
        )
        content.append(self._description_label)

        self._markdown_view = MarkdownView()
        content.append(self._markdown_view)
        self._set_description(addon.long_description)

        if addon.url:
            content.append(self._build_repository_link(addon.url))

        self._action_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        self._action_box.set_halign(Gtk.Align.END)
        box.append(self._action_box)
        self._set_action(addon)

        self._fetch_registry_entry()

    def _on_destroy(self, _widget):
        self._destroyed = True

    @staticmethod
    def _meta_text(addon: AddonMetadata) -> str:
        parts = []
        if addon.author.name:
            parts.append(_("by {author}").format(author=addon.author.name))
        if addon.version and addon.version is not UnknownVersion:
            parts.append(str(addon.version))
        if addon.license_name:
            parts.append(addon.license_name)
        return " · ".join(parts)

    def _set_description(self, long_description: str):
        """Shows the long description as Markdown, or the short one."""
        if long_description:
            self._markdown_view.set_text(long_description)
            self._markdown_view.set_visible(True)
            self._description_label.set_visible(False)
        else:
            self._markdown_view.set_visible(False)
            self._description_label.set_visible(True)

    def _build_repository_link(self, repository_url: str) -> Gtk.Label:
        """Create a clickable link to the addon's repository."""
        label = Gtk.Label(
            use_markup=True,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
            halign=Gtk.Align.START,
            xalign=0,
        )
        label.set_markup(
            '<a href="{url}" title="{tooltip}">{text}</a>'.format(
                url=GLib.markup_escape_text(repository_url),
                tooltip=GLib.markup_escape_text(
                    _("Open the repository in your browser")
                ),
                text=GLib.markup_escape_text(repository_url),
            )
        )
        label.connect("activate-link", self._on_activate_link)
        return label

    def _on_activate_link(self, _label, uri: str) -> bool:
        webbrowser.open(uri)
        return True

    def _set_action(self, addon: AddonMetadata):
        while child := self._action_box.get_first_child():
            self._action_box.remove(child)
        # Addons without a repository URL (for example built-ins) are
        # not managed through the registry, so there is no action.
        if (
            self._on_install is None and self._on_unlock is None
        ) or not addon.url:
            self._action_box.set_visible(False)
            return
        self._action_box.set_visible(True)
        btn = build_action_button(
            addon,
            on_install=self._on_install or (lambda _a, _u: None),
            on_unlock=self._on_unlock or (lambda _a: None),
        )
        self._action_box.append(btn)

    def _build_carousel(self, urls: list[str]):
        while child := self._carousel_box.get_first_child():
            self._carousel_box.remove(child)
        if not urls:
            self._carousel_box.set_visible(False)
            return

        carousel = Adw.Carousel()
        carousel.set_spacing(12)
        carousel.set_size_request(-1, CAROUSEL_HEIGHT)
        for url in urls:
            carousel.append(AsyncImage(url, min_height=CAROUSEL_HEIGHT))
        self._carousel_box.append(carousel)

        dots = Adw.CarouselIndicatorDots()
        dots.set_carousel(carousel)
        self._carousel_box.append(dots)
        self._carousel_box.set_visible(True)

    def _fetch_registry_entry(self):
        """Looks up the addon in the registry to enrich local data."""

        def _worker():
            try:
                entries = get_context().addon_mgr.fetch_registry_cached()
            except Exception as e:  # noqa: BLE001 - background enrichment
                logger.warning("Failed to fetch registry for details: %s", e)
                return None
            for entry in entries:
                if entry.name == self._addon.name:
                    return entry
            return None

        def _done(entry):
            if self._destroyed or entry is None:
                return
            if entry.screenshots:
                self._addon.screenshots = entry.screenshots
                self._build_carousel(entry.screenshots)
            if entry.long_description:
                self._addon.long_description = entry.long_description
                self._set_description(entry.long_description)
            self._set_action(entry)

        def _thread_target():
            entry = _worker()
            GLib.idle_add(_done, entry)

        thread = threading.Thread(target=_thread_target, daemon=True)
        thread.start()
