import logging
import platform
import sys
import webbrowser
from importlib.metadata import PackageNotFoundError, version

from gi.repository import Adw, GLib, Gtk

from .. import __version__
from .icons import get_icon
from .shared.patched_dialog_window import PatchedDialogWindow

logger = logging.getLogger(__name__)
_not_found_str = _("Not found")


def _get_version(package_name: str) -> str:
    """Safely retrieves the version of a Python package."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return _not_found_str


def get_dependency_info() -> dict:
    """
    Gathers version information for the application's key dependencies.
    This function only includes information that can be dynamically queried
    at runtime.
    """
    info = {}

    # System
    info[_("System")] = [
        ("Python", sys.version.split(" ")[0]),
        (
            "Platform",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
        ),
    ]

    # UI Toolkit
    try:
        import gi

        pygobject_ver = gi.__version__
    except (ImportError, AttributeError):
        pygobject_ver = _not_found_str

    ui_deps = [
        (
            "GTK",
            f"{Gtk.get_major_version()}."
            f"{Gtk.get_minor_version()}."
            f"{Gtk.get_micro_version()}",
        ),
        (
            "LibAdwaita",
            f"{Adw.get_major_version()}."
            f"{Adw.get_minor_version()}."
            f"{Adw.get_micro_version()}",
        ),
    ]
    if pygobject_ver != _not_found_str:
        ui_deps.append(("PyGObject", pygobject_ver))
    info[_("UI Toolkit")] = ui_deps

    # Graphics & Imaging
    graphics_deps = []
    try:
        import cairo

        graphics_deps.append(("PyCairo", cairo.version))
        graphics_deps.append(("libcairo", cairo.cairo_version_string()))
    except ImportError:
        pass

    try:
        import pyvips

        pyvips_ver = _get_version("pyvips")
        graphics_deps.append(("pyvips", pyvips_ver))
    except Exception as e:
        msg = f"failed to find pyvips version: {e}"
        logger.warning(msg)
        graphics_deps.append(("pyvips", msg))
        pyvips = None

    try:
        libvips_ver = pyvips.version(0) if pyvips else None
        graphics_deps.append(("libvips", libvips_ver))
    except Exception as e:
        msg = f"failed to find libvips version: {e}"
        logger.warning(msg)
        graphics_deps.append(("libvips", msg))

    for pkg_name, display_name in [
        ("opencv-python", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("vtracer", "vtracer"),
    ]:
        ver = _get_version(pkg_name)
        graphics_deps.append((display_name, ver))

    if graphics_deps:
        info[_("Graphics & Imaging")] = graphics_deps

    comm_deps = []
    for pkg in [
        "ezdxf",
        "pypdf",
        "PyYAML",
        "pyserial-asyncio",
        "aiohttp",
        "websockets",
    ]:
        ver = _get_version(pkg)
        comm_deps.append((pkg, ver))

    if comm_deps:
        info[_("File Formats & Communication")] = comm_deps

    return {k: v for k, v in info.items() if v}


def get_supporters() -> list[tuple[str, str | None]]:
    """
    Returns a list of supporters who donated to the app.
    Each entry is a tuple of (name, optional_url).
    """
    return [
        ("Anonymous Supporter", None),
    ]


class AboutDialog(PatchedDialogWindow):
    """
    A custom 'About' dialog that uses a ViewStack to navigate between
    the main page and a detailed system information page.
    """

    def __init__(self, **kwargs):
        super().__init__(modal=True, **kwargs)
        self.set_default_size(500, 700)
        self.set_hide_on_close(True)

        self._build_ui()

    def _on_copy_info_clicked(self, button: Gtk.Button):
        lines = [f"## Rayforge {__version__ or _not_found_str}", ""]
        dep_info = get_dependency_info()
        for category, deps in dep_info.items():
            lines.append(f"### {category}")
            for name, ver in deps:
                lines.append(f"{name}: {ver}")
            lines.append("")
        full_text = "\n".join(lines).strip()
        clipboard = self.get_display().get_clipboard()
        clipboard.set(full_text)
        button.set_child(get_icon("check-symbolic"))
        GLib.timeout_add(
            2000,
            lambda: button.set_child(get_icon("copy-symbolic"))
            and GLib.SOURCE_REMOVE,
        )
        # Also give feedback on the headerbar copy button if it exists
        if hasattr(self, "header_copy_button"):
            self.header_copy_button.set_child(get_icon("check-symbolic"))
            GLib.timeout_add(
                2000,
                lambda: self.header_copy_button.set_child(
                    get_icon("copy-symbolic")
                )
                and GLib.SOURCE_REMOVE,
            )

    def _on_copy_version_clicked(self, button: Gtk.Button):
        """Copy the version information to clipboard."""
        clipboard = self.get_display().get_clipboard()
        clipboard.set(__version__ or _not_found_str)
        button.set_child(get_icon("check-symbolic"))
        GLib.timeout_add(
            2000,
            lambda: button.set_child(get_icon("copy-symbolic"))
            and GLib.SOURCE_REMOVE,
        )

    def _build_main_page(self):
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.set_halign(Gtk.Align.FILL)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(24)

        hero_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        hero_box.set_vexpand(True)
        hero_box.set_valign(Gtk.Align.CENTER)
        hero_box.set_halign(Gtk.Align.CENTER)
        content_box.append(hero_box)

        icon = get_icon("org.rayforge.rayforge")
        icon.set_pixel_size(128)
        hero_box.append(icon)

        title = Gtk.Label()
        title.set_markup("<span size='xx-large' weight='bold'>Rayforge</span>")
        title.set_margin_top(6)
        hero_box.append(title)

        copyright_label = Gtk.Label(label="© 2025 Samuel Abels")
        hero_box.append(copyright_label)

        links_box = Gtk.Box(halign=Gtk.Align.CENTER, margin_top=12)
        links_box.add_css_class("linked")
        hero_box.append(links_box)

        website_button = Gtk.Button.new_with_label(_("Website"))
        website_button.connect(
            "clicked",
            lambda _: webbrowser.open("https://github.com/barebaric/rayforge"),
        )
        links_box.append(website_button)

        issues_button = Gtk.Button.new_with_label(_("Report an Issue"))
        issues_button.connect(
            "clicked",
            lambda _: webbrowser.open(
                "https://github.com/barebaric/rayforge/issues"
            ),
        )
        links_box.append(issues_button)

        donate_button = Gtk.Button.new_with_label(_("Donate"))
        donate_button.connect(
            "clicked",
            lambda _: webbrowser.open("https://www.patreon.com/c/knipknap"),
        )
        links_box.append(donate_button)

        prefgroup = Adw.PreferencesGroup()
        content_box.append(prefgroup)

        # Version row with copy button
        version_row = Adw.ActionRow(title=_("Version"))
        version_row.set_subtitle(__version__ or _not_found_str)

        copy_button = Gtk.Button(child=get_icon("copy-symbolic"))
        copy_button.set_valign(Gtk.Align.CENTER)
        copy_button.add_css_class("flat")
        copy_button.set_tooltip_text(_("Copy Version"))
        copy_button.connect("clicked", self._on_copy_version_clicked)
        version_row.add_suffix(copy_button)

        prefgroup.add(version_row)

        dev_row = Adw.ActionRow(
            title=_("Lead Developer"), subtitle="Samuel Abels"
        )
        prefgroup.add(dev_row)

        license_row = Adw.ActionRow(title=_("License"), subtitle="MIT X11")
        license_row.set_activatable(True)
        license_row.add_suffix(get_icon("open-in-new-symbolic"))
        license_row.connect(
            "activated",
            lambda _: webbrowser.open("https://opensource.org/license/mit"),
        )
        prefgroup.add(license_row)

        sys_info_row = Adw.ActionRow(
            title=_("System Information"),
            subtitle=_("Versions of libraries and components"),
        )
        sys_info_row.set_activatable(True)

        self.inline_copy_button = Gtk.Button(child=get_icon("copy-symbolic"))
        self.inline_copy_button.set_valign(Gtk.Align.CENTER)
        self.inline_copy_button.add_css_class("flat")
        self.inline_copy_button.set_tooltip_text(_("Copy System Information"))
        self.inline_copy_button.connect("clicked", self._on_copy_info_clicked)
        sys_info_row.add_suffix(self.inline_copy_button)

        sys_info_row.add_suffix(get_icon("go-next-symbolic"))
        sys_info_row.connect(
            "activated",
            lambda w: self.view_stack.set_visible_child_name("sysinfo"),
        )
        prefgroup.add(sys_info_row)

        supporters = get_supporters()
        if supporters:
            supporters_row = Adw.ActionRow(
                title=_("Supporters"),
                subtitle=_("People who donated to the project"),
            )
            supporters_row.set_activatable(True)
            supporters_row.add_suffix(get_icon("go-next-symbolic"))
            supporters_row.connect(
                "activated",
                lambda w: self.view_stack.set_visible_child_name("supporters"),
            )
            prefgroup.add(supporters_row)

        return content_box

    def _build_sysinfo_page(self):
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(24)
        scrolled_window.set_child(content_box)

        dep_info = get_dependency_info()
        for category, deps in dep_info.items():
            escaped_category = GLib.markup_escape_text(category)
            group = Adw.PreferencesGroup()
            group.set_title(escaped_category)
            content_box.append(group)

            for name, ver in deps:
                row = Adw.ActionRow(title=name, subtitle=str(ver))
                group.add(row)

        return scrolled_window

    def _build_supporters_page(self):
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(24)
        scrolled_window.set_child(content_box)

        thank_you_label = Gtk.Label()
        thank_you_label.set_markup(
            "<i>"
            + _(
                "Special thanks go to everyone who has donated to support "
                "Rayforge! You keep the coffee and the AI tokens flowing!"
            )
            + "</i>"
        )
        thank_you_label.set_margin_top(6)
        thank_you_label.set_margin_bottom(6)
        thank_you_label.set_wrap(True)
        thank_you_label.set_max_width_chars(60)
        content_box.append(thank_you_label)

        supporters_group = Adw.PreferencesGroup()
        supporters_group.set_title(_("Supporters"))
        content_box.append(supporters_group)

        for name, url in get_supporters():
            row = Adw.ActionRow(title=name)
            if url:
                row.set_activatable(True)
                row.add_suffix(get_icon("open-in-new-symbolic"))
                row.connect("activated", lambda _, u=url: webbrowser.open(u))
            supporters_group.add(row)

        return scrolled_window

    def _on_view_changed(self, stack, param):
        visible_page = stack.get_visible_child_name()
        is_main = visible_page == "main"

        self.back_button.set_visible(not is_main)
        self.header_copy_button.set_visible(visible_page == "sysinfo")

        if visible_page == "supporters":
            self.header_bar.set_title_widget(self.supporters_title)
        elif visible_page == "sysinfo":
            self.header_bar.set_title_widget(self.sysinfo_title)
        else:
            self.header_bar.set_title_widget(self.main_title)

    def _build_ui(self):
        self.header_bar = Adw.HeaderBar()
        self.main_title = Adw.WindowTitle(title=_("About Rayforge"))
        self.sysinfo_title = Adw.WindowTitle(title=_("System Information"))
        self.supporters_title = Adw.WindowTitle(title=_("Supporters"))

        self.back_button = Gtk.Button(child=get_icon("go-previous-symbolic"))
        self.back_button.connect(
            "clicked", lambda w: self.view_stack.set_visible_child_name("main")
        )
        self.header_bar.pack_start(self.back_button)

        self.header_copy_button = Gtk.Button(child=get_icon("copy-symbolic"))
        self.header_copy_button.set_tooltip_text(_("Copy System Information"))
        self.header_copy_button.connect("clicked", self._on_copy_info_clicked)
        self.header_bar.pack_end(self.header_copy_button)

        self.view_stack = Adw.ViewStack()
        self.view_stack.add_named(self._build_main_page(), "main")
        self.view_stack.add_named(self._build_sysinfo_page(), "sysinfo")
        self.view_stack.add_named(self._build_supporters_page(), "supporters")
        self.view_stack.connect(
            "notify::visible-child-name", self._on_view_changed
        )

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.append(self.header_bar)
        main_box.append(self.view_stack)
        self.set_content(main_box)
        self._on_view_changed(self.view_stack, None)  # Set initial state
